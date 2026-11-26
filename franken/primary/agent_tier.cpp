// agent_tier.cpp — Primary-side FrankenStrata agent tier implementation.

#include "franken/primary/agent_tier.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace franken {
namespace primary {

namespace fw = franken::wire;
namespace fi = franken::ipc;

bool parse_agent_spec(const std::string& arg, AgentSpec& out, std::string& err) {
    out = AgentSpec{};
    if (arg.empty()) {
        err = "--franken-agent requires <shm_name>[,slots=N]";
        return false;
    }
    const size_t comma = arg.find(',');
    out.shm_name = comma == std::string::npos ? arg : arg.substr(0, comma);
    if (out.shm_name.empty()) {
        err = "--franken-agent: empty shm_name in '" + arg + "'";
        return false;
    }
    if (comma != std::string::npos) {
        size_t pos = comma + 1;
        while (pos < arg.size()) {
            const size_t next = arg.find(',', pos);
            const std::string kv = next == std::string::npos ? arg.substr(pos)
                                                             : arg.substr(pos, next - pos);
            if (kv.rfind("slots=", 0) == 0) {
                const std::string val = kv.substr(6);
                char* end = nullptr;
                const unsigned long n = std::strtoul(val.c_str(), &end, 10);
                if (end == val.c_str() || *end != '\0' || n == 0) {
                    err = "--franken-agent: invalid slots= value in '" + arg + "'";
                    return false;
                }
                out.max_slots = uint32_t(n);
            } else if (!kv.empty()) {
                err = "--franken-agent: unknown option '" + kv + "' in '" + arg + "'";
                return false;
            }
            if (next == std::string::npos) break;
            pos = next + 1;
        }
    }
    return true;
}

AgentTier::~AgentTier() { close(); }

const char* AgentTier::vendor_str() const {
    switch (vendor()) {
        case fw::AgentVendor::Sycl: return "SYCL";
        case fw::AgentVendor::Hip: return "HIP";
        case fw::AgentVendor::Cuda: return "CUDA";
        case fw::AgentVendor::Cpu: return "CPU";
        default: return "Unknown";
    }
}

void AgentTier::mark_dead(const char* reason) {
    if (!alive_) return;
    alive_ = false;
    std::fprintf(stderr, "strata franken-agent [%s/%s]: marked Dead (%s); falling back to CPU\n",
                 seg_.name().c_str(), vendor_str(), reason);
}

void AgentTier::drain_error_acks() {
    (void)poll_control_acks();
}

bool AgentTier::install_bitmap(uint64_t bitmap_offset, uint32_t gen,
                               std::vector<uint8_t>* claimed_out, std::string& err) {
    const uint64_t bm_words = fi::bitmap_words(geom_.n_layers, geom_.n_expert);
    const uint64_t bm_bytes = bm_words * sizeof(uint64_t);
    if (!seg_.var_range_ok(bitmap_offset, bm_bytes)) {
        err = "franken-agent [" + seg_.name() + "]: bitmap_offset out of bounds";
        return false;
    }
    resident_.assign(size_t(n_layers_ * n_expert_), 0);
    resident_count_ = 0;
    const auto* bm = reinterpret_cast<const uint64_t*>(seg_.var_ptr(bitmap_offset));
    for (int64_t l = 0; l < n_layers_; ++l) {
        for (int64_t e = 0; e < n_expert_; ++e) {
            const uint64_t idx = uint64_t(l) * uint64_t(n_expert_) + uint64_t(e);
            if ((bm[idx / 64] >> (idx % 64)) & 1ull) {
                resident_[size_t(idx)] = 1;
                if (claimed_out != nullptr && size_t(idx) < claimed_out->size()) {
                    (*claimed_out)[size_t(idx)] = 1;
                }
                ++resident_count_;
            }
        }
    }
    residency_gen_ = gen;
    seg_.control()->residency_gen.value.store(gen, std::memory_order_release);
    return true;
}

bool AgentTier::has_pending_promote(int64_t layer, int32_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    const uint64_t key = uint64_t(layer) * uint64_t(n_expert_) + uint64_t(expert);
    return pending_promotes_.find(key) != pending_promotes_.end();
}

bool AgentTier::send_demote(int32_t layer, int32_t expert) {
    if (!alive_ || !seg_.valid()) return false;
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;

    const uint64_t idx = uint64_t(layer) * uint64_t(n_expert_) + uint64_t(expert);
    // Primary-authoritative & immediate for routing (DESIGN §6.3).
    if (resident_[size_t(idx)] != 0) {
        resident_[size_t(idx)] = 0;
        if (resident_count_ > 0) --resident_count_;
    }
    pending_promotes_.erase(idx);

    const uint32_t next_gen = ++residency_gen_;
    seg_.control()->residency_gen.value.store(next_gen, std::memory_order_release);

    fw::CtrlMsg msg{};
    msg.type = uint32_t(fw::CtrlType::Demote);
    msg.seq = next_ctrl_seq_++;
    msg.demote = {layer, expert, next_gen};
    if (!fi::ring_push(*seg_.ctrl_ring(), msg)) {
        needs_resync_ = true;
        return false;
    }
    fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);
    ++demotes_sent_;
    return true;
}

bool AgentTier::send_promote(int32_t layer, int32_t expert) {
    if (!alive_ || !seg_.valid()) return false;
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;

    const uint64_t idx = uint64_t(layer) * uint64_t(n_expert_) + uint64_t(expert);
    if (resident_[size_t(idx)] != 0) return false;
    if (pending_promotes_.find(idx) != pending_promotes_.end()) return false;

    const uint32_t next_gen = ++residency_gen_;
    fw::CtrlMsg msg{};
    msg.type = uint32_t(fw::CtrlType::Promote);
    msg.seq = next_ctrl_seq_++;
    msg.promote = {layer, expert, next_gen};
    if (!fi::ring_push(*seg_.ctrl_ring(), msg)) {
        needs_resync_ = true;
        return false;
    }
    // Safe by construction: stays non-routable until matching PromoteAck lands.
    pending_promotes_[idx] = PendingPromote{layer, expert, next_gen, msg.seq, fi::now_ns()};
    fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);
    ++promotes_sent_;
    return true;
}

bool AgentTier::poll_control_acks() {
    if (!seg_.valid()) return true;
    fw::AckMsg ack{};
    while (fi::ring_pop(*seg_.ack_ring(), ack)) {
        switch (static_cast<fw::AckType>(ack.type)) {
            case fw::AckType::PromoteAck: {
                const int32_t l = ack.promote_ack.layer;
                const int32_t e = ack.promote_ack.expert;
                if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) {
                    needs_resync_ = true;
                    break;
                }
                const uint64_t idx = uint64_t(l) * uint64_t(n_expert_) + uint64_t(e);
                auto it = pending_promotes_.find(idx);
                if (it == pending_promotes_.end()) {
                    // Could have been immediately demoted before PromoteAck arrived; ignore if gen <= residency_gen_.
                    if (ack.promote_ack.gen > residency_gen_) needs_resync_ = true;
                    break;
                }
                if (it->second.gen != ack.promote_ack.gen) {
                    needs_resync_ = true;
                    pending_promotes_.erase(it);
                    break;
                }
                const uint64_t now = fi::now_ns();
                // Subtract batch queueing delay when multiple promotes were enqueued together.
                const uint64_t eff_start = std::max(it->second.send_time_ns, last_promote_ack_ns_);
                if (now >= eff_start) {
                    const double move_ms = double(now - eff_start) / 1e6;
                    if (move_samples_ < 16) {
                        c_move_ms_ += (move_ms - c_move_ms_) / double(++move_samples_);
                    } else {
                        c_move_ms_ += 0.1 * (move_ms - c_move_ms_);
                        ++move_samples_;
                    }
                }
                last_promote_ack_ns_ = now;
                pending_promotes_.erase(it);
                if (resident_[size_t(idx)] == 0) {
                    resident_[size_t(idx)] = 1;
                    ++resident_count_;
                }
                ++promotes_acked_;
                seg_.control()->residency_gen.value.store(residency_gen_, std::memory_order_release);
                break;
            }
            case fw::AckType::DemoteAck: {
                if (ack.demote_ack.gen > residency_gen_) needs_resync_ = true;
                break;
            }
            case fw::AckType::OpenAck: {
                std::string ierr;
                if (install_bitmap(ack.open_ack.bitmap_offset, ack.open_ack.gen, nullptr, ierr)) {
                    pending_promotes_.clear();
                    needs_resync_ = false;
                }
                break;
            }
            case fw::AckType::Error: {
                const std::string txt = seg_.read_error_text(ack.error.text_offset, ack.error.text_len);
                std::fprintf(stderr, "strata franken-agent [%s/%s] %s error: %s\n",
                             seg_.name().c_str(), vendor_str(),
                             ack.error.fatal ? "FATAL" : "warning", txt.c_str());
                // Remove any pending promote whose ctrl seq errored out.
                for (auto it = pending_promotes_.begin(); it != pending_promotes_.end(); ++it) {
                    if (it->second.ctrl_seq == ack.seq) {
                        pending_promotes_.erase(it);
                        break;
                    }
                }
                if (ack.error.fatal) mark_dead("agent reported fatal ERROR");
                break;
            }
            default:
                break;
        }
    }
    return !needs_resync_;
}

bool AgentTier::resync(uint32_t timeout_ms) {
    if (!seg_.valid()) return false;
    const fw::ControlBlock* cb = seg_.control();
    if (cb->agent_state.value.load(std::memory_order_acquire) == uint32_t(fw::AgentState::Dead)) {
        return false;
    }
    if (hb_.check_stale(cb->heartbeat.value.load(std::memory_order_acquire))) {
        return false;
    }

    const uint32_t next_gen = ++residency_gen_;
    fw::CtrlMsg msg{};
    msg.type = uint32_t(fw::CtrlType::Resync);
    msg.seq = next_ctrl_seq_++;
    msg.resync = {next_gen, 0, seg_.plan().primary_pairs_offset};
    if (!fi::ring_push(*seg_.ctrl_ring(), msg)) return false;
    fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);

    const uint64_t deadline = fi::now_ns() + uint64_t(timeout_ms) * uint64_t(1000000);
    fw::AckMsg ack{};
    while (fi::now_ns() < deadline) {
        if (fi::ring_pop(*seg_.ack_ring(), ack)) {
            if (ack.type == uint32_t(fw::AckType::OpenAck) && ack.seq == msg.seq) {
                std::string ierr;
                if (!install_bitmap(ack.open_ack.bitmap_offset, ack.open_ack.gen, nullptr, ierr)) {
                    return false;
                }
                pending_promotes_.clear();
                needs_resync_ = false;
                consecutive_salvages_ = 0;
                alive_ = true;
                ++resyncs_done_;
                return true;
            }
            if (ack.type == uint32_t(fw::AckType::Error) && ack.error.fatal) {
                mark_dead("fatal error during RESYNC");
                return false;
            }
        }
        ::usleep(1000);
    }
    return false;
}

bool AgentTier::check_heartbeat_alive() {
    if (!alive_ || !seg_.valid()) return false;
    const fw::ControlBlock* cb = seg_.control();
    const uint32_t st = cb->agent_state.value.load(std::memory_order_acquire);
    if (st == uint32_t(fw::AgentState::Dead)) {
        mark_dead("agent_state == Dead");
        return false;
    }
    const uint64_t beat = cb->heartbeat.value.load(std::memory_order_acquire);
    if (hb_.check_stale(beat)) {
        mark_dead("heartbeat stale > 250 ms");
        return false;
    }
    return true;
}

uint64_t AgentTier::compute_deadline_ns() const {
    if (fixed_deadline_ms_ > 0.0) {
        return uint64_t(fixed_deadline_ms_ * 1e6);
    }
    const uint64_t floor_ns = uint64_t(fw::kMinDeadlineMs) * 1000000ull;
    if (recent_lat_ns_.size() < 8) {
        // Warmup grace before rolling p95 is populated (first JIT/AOT launches can be slower).
        return std::max<uint64_t>(floor_ns, 50000000ull);  // 50 ms warmup
    }
    std::vector<uint64_t> sorted(recent_lat_ns_.begin(), recent_lat_ns_.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t p95_idx = std::min(sorted.size() - 1, (sorted.size() * 95) / 100);
    const uint64_t p95_ns = sorted[p95_idx];
    return std::max<uint64_t>(floor_ns, uint64_t(fw::kDeadlineP95Multiplier) * p95_ns);
}

void AgentTier::record_latency_ns(uint64_t lat_ns) {
    constexpr size_t kWindow = 128;
    if (recent_lat_ns_.size() >= kWindow) recent_lat_ns_.pop_front();
    recent_lat_ns_.push_back(lat_ns);
}

bool AgentTier::open(const AgentSpec& spec,
                     const wire::Geometry& geom_offer,
                     const std::vector<uint64_t>& blob_bytes,
                     const std::vector<std::pair<int32_t, int32_t>>& ranked,
                     std::vector<uint8_t>& claimed,
                     double fixed_deadline_ms,
                     uint32_t hello_timeout_ms,
                     uint32_t open_timeout_ms,
                     std::string& err) {
    close();

    if (!seg_.create(spec.shm_name, geom_offer, blob_bytes, err)) {
        return false;
    }
    geom_ = seg_.control()->geometry;
    n_layers_ = geom_.n_layers;
    n_expert_ = geom_.n_expert;
    n_embd_ = geom_.n_embd;
    fixed_deadline_ms_ = fixed_deadline_ms;

    if (claimed.size() < size_t(n_layers_ * n_expert_)) {
        claimed.assign(size_t(n_layers_ * n_expert_), 0);
    }

    std::fprintf(stderr, "strata franken-agent [%s]: created shm (%.2f MiB); waiting for HELLO...\n",
                 seg_.name().c_str(), double(seg_.size()) / (1024.0 * 1024.0));

    // 1. Wait for HELLO on ack_ring.
    const uint64_t hello_deadline = fi::now_ns() + uint64_t(hello_timeout_ms) * 1000000ull;
    fw::AckMsg ack{};
    bool got_hello = false;
    while (!got_hello) {
        if (fi::ring_pop(*seg_.ack_ring(), ack)) {
            if (ack.type == uint32_t(fw::AckType::Hello)) {
                caps_ = ack.hello.caps;
                got_hello = true;
                break;
            }
            if (ack.type == uint32_t(fw::AckType::Error)) {
                const std::string txt = seg_.read_error_text(ack.error.text_offset, ack.error.text_len);
                err = "franken-agent [" + seg_.name() + "] error before HELLO: " + txt;
                close();
                return false;
            }
        }
        if (fi::now_ns() >= hello_deadline) {
            err = "franken-agent [" + seg_.name() + "]: timed out waiting for HELLO";
            close();
            return false;
        }
        ::usleep(5000);
    }

    max_tokens_ = std::min(geom_.max_tokens, caps_.max_tokens_per_window);
    max_entries_ = std::min(geom_.max_entries, caps_.max_entries_per_window);
    if (max_tokens_ == 0 || max_entries_ == 0) {
        err = "franken-agent [" + seg_.name() + "]: invalid caps (max_tokens or max_entries == 0)";
        close();
        return false;
    }

    const bool no_q8 = std::getenv("STRATA_FRANKEN_NO_Q8_1") != nullptr &&
                       std::atoi(std::getenv("STRATA_FRANKEN_NO_Q8_1")) != 0;
    const bool no_compact = std::getenv("STRATA_FRANKEN_NO_COMPACT") != nullptr &&
                            std::atoi(std::getenv("STRATA_FRANKEN_NO_COMPACT")) != 0;
    use_q8_1_ = !no_q8 && ((caps_.feature_flags & fw::kCapQ8_1Activations) != 0);
    use_compact_rows_ = !no_compact && ((caps_.feature_flags & fw::kCapCompactRows) != 0);

    const uint32_t slot_budget =
        spec.max_slots > 0 ? std::min(caps_.slot_count, spec.max_slots) : caps_.slot_count;

    // 2. Build disjoint fill list: next-hottest unclaimed experts from `ranked`,
    //    or layer-early fallback if no profile was provided.
    std::vector<std::pair<int32_t, int32_t>> fill;
    fill.reserve(slot_budget);
    std::vector<uint8_t> picked(size_t(n_layers_ * n_expert_), 0);

    auto try_add = [&](int32_t l, int32_t e) {
        if (fill.size() >= slot_budget) return;
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) return;
        const size_t idx = size_t(l) * size_t(n_expert_) + size_t(e);
        if (claimed[idx] || picked[idx]) return;
        picked[idx] = 1;
        fill.emplace_back(l, e);
    };

    if (!ranked.empty()) {
        for (const auto& [l, e] : ranked) {
            try_add(l, e);
            if (fill.size() >= slot_budget) break;
        }
    }
    // Layer-early fallback (when no profile or profile had fewer entries than budget).
    for (int32_t l = 0; l < n_layers_ && fill.size() < slot_budget; ++l) {
        for (int32_t e = 0; e < n_expert_ && fill.size() < slot_budget; ++e) {
            try_add(l, e);
        }
    }

    auto* dst_pairs = reinterpret_cast<int32_t*>(seg_.var_ptr(seg_.plan().primary_pairs_offset));
    for (size_t i = 0; i < fill.size(); ++i) {
        dst_pairs[2 * i] = fill[i].first;
        dst_pairs[2 * i + 1] = fill[i].second;
    }

    fw::CtrlMsg open_msg{};
    open_msg.type = uint32_t(fw::CtrlType::Open);
    open_msg.seq = next_ctrl_seq_++;
    open_msg.open.gen = 1;
    open_msg.open.n_fill = uint32_t(fill.size());
    open_msg.open.fill_offset = seg_.plan().primary_pairs_offset;

    if (!fi::ring_push(*seg_.ctrl_ring(), open_msg)) {
        err = "franken-agent [" + seg_.name() + "]: ctrl ring full on OPEN";
        close();
        return false;
    }
    const uint64_t open_t0_ns = fi::now_ns();
    fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);

    std::fprintf(stderr,
                 "strata franken-agent [%s/%s]: HELLO ok (advertised %u slots, q8_1=%d compact=%d); sent OPEN (%zu experts)...\n",
                 seg_.name().c_str(), vendor_str(), caps_.slot_count,
                 use_q8_1_ ? 1 : 0, use_compact_rows_ ? 1 : 0, fill.size());

    // 3. Wait for OPEN_ACK while monitoring agent heartbeat / fatal errors.
    hb_.reset(seg_.control()->heartbeat.value.load(std::memory_order_acquire));
    const uint64_t open_deadline = fi::now_ns() + uint64_t(open_timeout_ms) * 1000000ull;
    bool got_open_ack = false;
    fw::OpenAckPayload open_ack{};

    while (!got_open_ack) {
        if (fi::ring_pop(*seg_.ack_ring(), ack)) {
            if (ack.type == uint32_t(fw::AckType::OpenAck) && ack.seq == open_msg.seq) {
                open_ack = ack.open_ack;
                got_open_ack = true;
                break;
            }
            if (ack.type == uint32_t(fw::AckType::Error)) {
                const std::string txt = seg_.read_error_text(ack.error.text_offset, ack.error.text_len);
                err = "franken-agent [" + seg_.name() + "] error during OPEN: " + txt;
                close();
                return false;
            }
        }
        const uint32_t st = seg_.control()->agent_state.value.load(std::memory_order_acquire);
        if (st == uint32_t(fw::AgentState::Dead)) {
            err = "franken-agent [" + seg_.name() + "] transitioned to Dead during OPEN";
            close();
            return false;
        }
        if (hb_.check_stale(seg_.control()->heartbeat.value.load(std::memory_order_acquire))) {
            err = "franken-agent [" + seg_.name() + "] heartbeat went stale during OPEN";
            close();
            return false;
        }
        if (fi::now_ns() >= open_deadline) {
            err = "franken-agent [" + seg_.name() + "]: timed out waiting for OPEN_ACK";
            close();
            return false;
        }
        ::usleep(5000);
    }

    // 4. Install resident bitmap from OPEN_ACK and mark those experts claimed.
    const uint64_t open_t1_ns = fi::now_ns();
    open_fill_ms_ = double(open_t1_ns - open_t0_ns) / 1e6;
    if (!install_bitmap(open_ack.bitmap_offset, open_ack.gen, &claimed, err)) {
        close();
        return false;
    }
    open_fill_bytes_ = 0;
    for (int64_t l = 0; l < n_layers_; ++l) {
        const uint64_t bb = blob_bytes[size_t(l)];
        for (int64_t e = 0; e < n_expert_; ++e) {
            if (resident_[size_t(l * n_expert_ + e)] != 0) open_fill_bytes_ += bb;
        }
    }
    allocated_slots_ = uint32_t(resident_count_);
    pending_promotes_.clear();
    needs_resync_ = false;
    hb_.reset(seg_.control()->heartbeat.value.load(std::memory_order_acquire));
    alive_ = true;
    launched_ = false;
    consecutive_salvages_ = 0;
    next_doorbell_seq_ = seg_.control()->doorbell_seq.value.load(std::memory_order_acquire);

    std::fprintf(stderr, "strata franken-agent [%s/%s]: Ready with %lld resident experts (gen=%u)\n",
                 seg_.name().c_str(), vendor_str(), (long long)resident_count_, open_ack.gen);
    return true;
}

void AgentTier::close() {
    if (seg_.valid()) {
        if (alive_) {
            fw::CtrlMsg bye{};
            bye.type = uint32_t(fw::CtrlType::Bye);
            bye.seq = next_ctrl_seq_++;
            (void)fi::ring_push(*seg_.ctrl_ring(), bye);
            fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);

            // Give agent up to 200 ms to acknowledge Dead cleanly before unlinking.
            const uint64_t t_end = fi::now_ns() + 200000000ull;
            while (fi::now_ns() < t_end) {
                if (seg_.control()->agent_state.value.load(std::memory_order_acquire) ==
                    uint32_t(fw::AgentState::Dead)) {
                    break;
                }
                ::usleep(2000);
            }
        }
        seg_.close();
    }
    alive_ = false;
    launched_ = false;
    needs_resync_ = false;
    residency_gen_ = 0;
    resident_.clear();
    resident_count_ = 0;
    allocated_slots_ = 0;
    launched_entries_.clear();
    pending_promotes_.clear();
    recent_lat_ns_.clear();
}

bool AgentTier::launch(int64_t layer,
                       const float* x,
                       const int32_t* ids,
                       int64_t n_tok,
                       int64_t k,
                       const int32_t* kind,
                       bool allow_q8_1) {
    launched_ = false;
    launched_entries_.clear();
    active_distinct_ = 0;
    active_compact_out_ = false;
    if (!alive_) return false;

    drain_error_acks();
    if (!check_heartbeat_alive()) return false;
    if (!can_dispatch(n_tok, k)) return false;

    const int64_t n = n_tok * k;
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 3) continue;
        launched_entries_.push_back(int32_t(i));
        bool seen = false;
        for (int64_t j = 0; j < i; ++j) {
            if (kind[j] == 3 && ids[j] == ids[i]) { seen = true; break; }
        }
        if (!seen) ++active_distinct_;
    }
    if (launched_entries_.empty()) return false;

    const uint64_t t0 = fi::now_ns();
    const uint64_t seq = ++next_doorbell_seq_;
    fw::DescriptorHeader* dh = seg_.descriptor(seq);

    const bool send_q8_1 = use_q8_1_ && allow_q8_1 && ((n_embd_ & 31) == 0);
    const bool compact_out = use_compact_rows_;

    const uint64_t budget_ns = compute_deadline_ns();
    dh->payload_kind = uint32_t(fw::PayloadKind::ExpertRows);
    dh->seq = seq;
    dh->layer = int32_t(layer);
    dh->n_tok = int32_t(n_tok);
    dh->k = int32_t(k);
    dh->n_agent_entries = int32_t(launched_entries_.size());
    dh->deadline_hint_ns = budget_ns;
    dh->flags = (send_q8_1 ? fw::kDescFlagActQ8_1 : 0u) |
                (compact_out ? fw::kDescFlagCompactOut : 0u);

    std::memcpy(fw::slot_ids(dh), ids, size_t(n) * sizeof(int32_t));
    std::memcpy(fw::slot_kind(dh), kind, size_t(n) * sizeof(int32_t));
    if (send_q8_1) {
        fi::quantize_q8_1_rows_host(x, int32_t(n_tok), int32_t(n_embd_),
                                    reinterpret_cast<uint8_t*>(fw::slot_x(dh)));
    } else {
        std::memcpy(fw::slot_x(dh), x, size_t(n_tok) * size_t(n_embd_) * sizeof(float));
    }

    dh->state.store(uint32_t(fw::SlotState::Published), std::memory_order_release);
    seg_.control()->doorbell_seq.value.store(seq, std::memory_order_release);
    fi::futex_wake_u64_low(&seg_.control()->doorbell_seq.value);

    const uint64_t t1 = fi::now_ns();
    ms_launch_ += double(t1 - t0) / 1e6;
    active_seq_ = seq;
    active_compact_out_ = compact_out;
    launch_start_ns_ = t1;
    active_deadline_ns_ = t1 + budget_ns;
    launched_ = true;
    ++rounds_launched_;
    return true;
}

bool AgentTier::finish(float* out, std::vector<int32_t>& salvaged_entries) {
    if (!launched_) return true;
    launched_ = false;

    const uint64_t tw0 = fi::now_ns();
    fw::DescriptorHeader* dh = seg_.descriptor(active_seq_);
    const fi::SlotWaitResult wr = fi::wait_slot_or_salvage(
        dh, active_deadline_ns_, fw::kSpinPrimaryUs,
        [&]() { return !check_heartbeat_alive(); });
    const uint64_t tw1 = fi::now_ns();
    ms_wait_ += double(tw1 - tw0) / 1e6;

    drain_error_acks();

    if (wr == fi::SlotWaitResult::Done) {
        const uint64_t round_ns = tw1 - launch_start_ns_;
        record_latency_ns(round_ns);
        consecutive_salvages_ = 0;

        const float* row_arena = seg_.row_slot(active_seq_);
        const size_t row_bytes = size_t(n_embd_) * sizeof(float);
        if (active_compact_out_) {
            for (size_t r = 0; r < launched_entries_.size(); ++r) {
                std::memcpy(out + size_t(launched_entries_[r]) * size_t(n_embd_),
                            row_arena + r * size_t(n_embd_),
                            row_bytes);
            }
        } else {
            for (int32_t entry : launched_entries_) {
                std::memcpy(out + size_t(entry) * size_t(n_embd_),
                            row_arena + size_t(entry) * size_t(n_embd_),
                            row_bytes);
            }
        }
        const uint64_t tc1 = fi::now_ns();
        ms_copy_ += double(tc1 - tw1) / 1e6;
        last_round_ms_ = double(tc1 - launch_start_ns_) / 1e6;
        last_round_entries_ = int32_t(launched_entries_.size());
        last_round_distinct_ = active_distinct_;
        ++rounds_done_;
        entries_served_ += uint64_t(launched_entries_.size());
        return true;
    }

    // Salvaged: deadline expired or agent died. Return every entry to CPU fallback.
    ++rounds_salvaged_;
    ++consecutive_salvages_;
    salvaged_entries.insert(salvaged_entries.end(),
                            launched_entries_.begin(), launched_entries_.end());

    if (alive_ && consecutive_salvages_ >= fw::kMaxSalvagesBeforeDead) {
        mark_dead("exceeded consecutive salvage threshold");
    }
    return false;
}

}  // namespace primary
}  // namespace franken
