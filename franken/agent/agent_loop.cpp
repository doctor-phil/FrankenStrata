// agent_loop.cpp — Vendor-neutral FrankenStrata agent runtime loop.

#include "franken/agent/agent_loop.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace franken {
namespace agent {

namespace {

std::atomic<bool> g_signal_stop{false};

void signal_handler(int) {
    g_signal_stop.store(true, std::memory_order_release);
}

const char* vendor_name(uint32_t v) {
    switch (static_cast<wire::AgentVendor>(v)) {
        case wire::AgentVendor::Sycl: return "SYCL";
        case wire::AgentVendor::Hip: return "HIP";
        case wire::AgentVendor::Cuda: return "CUDA";
        case wire::AgentVendor::Cpu: return "CPU";
        default: return "Unknown";
    }
}

void push_error_ack(ipc::ShmSegment& seg, uint64_t seq, const std::string& msg, bool fatal) {
    uint64_t off = 0;
    uint32_t len = 0;
    seg.write_error_text(msg, off, len);
    wire::AckMsg ack{};
    ack.type = uint32_t(wire::AckType::Error);
    ack.seq = seq;
    ack.error.text_offset = off;
    ack.error.text_len = len;
    ack.error.fatal = fatal ? 1u : 0u;
    (void)ipc::ring_push(*seg.ack_ring(), ack);
}

void write_resident_bitmap(ipc::ShmSegment& seg,
                           int32_t n_expert,
                           const std::vector<std::pair<int32_t, int32_t>>& loaded) {
    auto* bm = reinterpret_cast<uint64_t*>(seg.var_ptr(seg.plan().agent_bitmap_offset));
    std::memset(bm, 0, size_t(seg.plan().agent_bitmap_bytes));
    for (const auto& [l, e] : loaded) {
        const uint64_t idx = uint64_t(l) * uint64_t(n_expert) + uint64_t(e);
        bm[idx / 64] |= (1ull << (idx % 64));
    }
}

}  // namespace

void install_agent_signal_handlers() {
    g_signal_stop.store(false, std::memory_order_release);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

bool run_agent_loop(const AgentLoopConfig& cfg,
                    AgentBackend& backend,
                    AgentLoopStats& stats,
                    std::string& err) {
    stats = AgentLoopStats{};

    // 1. Wait for and attach to the primary's shared-memory segment.
    ipc::ShmSegment seg;
    const uint64_t attach_deadline =
        ipc::now_ns() + uint64_t(cfg.attach_timeout_ms) * 1000000ull;
    while (true) {
        std::string aerr;
        if (seg.attach(cfg.shm_name, aerr)) break;
        if (g_signal_stop.load(std::memory_order_acquire)) {
            err = "interrupted while waiting for shm segment " + cfg.shm_name;
            return false;
        }
        if (ipc::now_ns() >= attach_deadline) {
            err = "timed out waiting for shm segment " + cfg.shm_name + ": " + aerr;
            return false;
        }
        ::usleep(20000);
    }

    wire::ControlBlock* cb = seg.control();
    const wire::Geometry g = cb->geometry;
    const uint64_t* blob_bytes = seg.blob_bytes_table();

    if (cfg.verbose) {
        std::fprintf(stderr, "franken-agent (%s): attached to %s (L=%d E=%d k=%d H=%d max_t=%u max_e=%u)\n",
                     vendor_name(uint32_t(backend.vendor())), seg.name().c_str(),
                     g.n_layers, g.n_expert, g.k, g.n_embd, g.max_tokens, g.max_entries);
    }

    // 2. Initialize backend, verify PackIdentity, publish Caps + HELLO.
    wire::Caps caps{};
    wire::PackIdentity local_pack{};
    if (!backend.init(g, blob_bytes, caps, local_pack, err)) {
        push_error_ack(seg, 0, err, /*fatal=*/true);
        cb->agent_state.value.store(uint32_t(wire::AgentState::Dead), std::memory_order_release);
        return false;
    }

    if (!ipc::pack_identity_matches(g.pack, local_pack)) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "pack identity mismatch: primary (fmt=%u size=%llu hash=%016llx) != "
                      "agent (fmt=%u size=%llu hash=%016llx)",
                      g.pack.format, (unsigned long long)g.pack.experts_bin_size,
                      (unsigned long long)g.pack.gguf_shard_list_hash,
                      local_pack.format, (unsigned long long)local_pack.experts_bin_size,
                      (unsigned long long)local_pack.gguf_shard_list_hash);
        err = buf;
        push_error_ack(seg, 0, err, /*fatal=*/true);
        cb->agent_state.value.store(uint32_t(wire::AgentState::Dead), std::memory_order_release);
        backend.shutdown();
        return false;
    }

    caps.max_tokens_per_window = std::min(g.max_tokens, caps.max_tokens_per_window);
    caps.max_entries_per_window = std::min(g.max_entries, caps.max_entries_per_window);
    caps.vendor = uint32_t(backend.vendor());
    cb->caps = caps;

    // Start 10 ms heartbeat thread before transitioning out of Absent so the
    // primary immediately sees a live heartbeat during OPEN fill.
    std::mutex hb_mu;
    std::condition_variable hb_cv;
    bool hb_stop = false;
    cb->heartbeat.value.fetch_add(1, std::memory_order_release);
    std::thread hb_thread([&] {
        std::unique_lock<std::mutex> lk(hb_mu);
        while (!hb_stop) {
            cb->heartbeat.value.fetch_add(1, std::memory_order_release);
            hb_cv.wait_for(lk, std::chrono::milliseconds(wire::kHeartbeatPeriodMs),
                           [&] { return hb_stop; });
        }
    });

    auto stop_heartbeat = [&] {
        {
            std::lock_guard<std::mutex> lk(hb_mu);
            hb_stop = true;
        }
        hb_cv.notify_all();
        if (hb_thread.joinable()) hb_thread.join();
    };

    cb->agent_state.value.store(uint32_t(wire::AgentState::Loading), std::memory_order_release);

    wire::AckMsg hello{};
    hello.type = uint32_t(wire::AckType::Hello);
    hello.seq = 0;
    hello.hello.caps = caps;
    hello.hello.pack_size =
        local_pack.format == uint32_t(wire::PackFormat::ExpertsBin)
            ? local_pack.experts_bin_size
            : local_pack.gguf_shard_list_hash;
    if (!ipc::ring_push(*seg.ack_ring(), hello)) {
        err = "ack ring full when posting HELLO";
        stop_heartbeat();
        cb->agent_state.value.store(uint32_t(wire::AgentState::Dead), std::memory_order_release);
        backend.shutdown();
        return false;
    }

    if (cfg.verbose) {
        std::fprintf(stderr, "franken-agent (%s): posted HELLO (slots=%u max_t=%u max_e=%u caps=0x%x)\n",
                     vendor_name(caps.vendor), caps.slot_count,
                     caps.max_tokens_per_window, caps.max_entries_per_window, caps.feature_flags);
    }

    // Background non-blocking PROMOTE refill thread (Phase 3): ensures blob I/O
    // and H2D copies never stall compute doorbells on the main loop thread.
    struct PromoteTask {
        int32_t layer = 0;
        int32_t expert = 0;
        uint32_t gen = 0;
        uint64_t ctrl_seq = 0;
    };
    std::mutex ack_mu;
    std::mutex pairs_mu;
    std::vector<std::pair<int32_t, int32_t>> resident_pairs;

    std::mutex refill_mu;
    std::condition_variable refill_cv;
    std::deque<PromoteTask> refill_queue;
    bool refill_stop = false;
    bool refill_busy = false;
    std::pair<int32_t, int32_t> refill_active{-1, -1};
    std::atomic<uint64_t> async_promotes_done{0};

    auto push_ack_safe = [&](const wire::AckMsg& msg) {
        std::lock_guard<std::mutex> lk(ack_mu);
        (void)ipc::ring_push(*seg.ack_ring(), msg);
    };
    auto push_error_safe = [&](uint64_t seq, const std::string& msg, bool fatal) {
        std::lock_guard<std::mutex> lk(ack_mu);
        push_error_ack(seg, seq, msg, fatal);
    };

    std::thread refill_thread([&] {
        std::unique_lock<std::mutex> lk(refill_mu);
        while (true) {
            refill_cv.wait(lk, [&] { return refill_stop || !refill_queue.empty(); });
            if (refill_stop && refill_queue.empty()) break;
            const PromoteTask task = refill_queue.front();
            refill_queue.pop_front();
            refill_busy = true;
            refill_active = {task.layer, task.expert};
            lk.unlock();

            std::string perr;
            if (backend.promote(task.layer, task.expert, perr)) {
                {
                    std::lock_guard<std::mutex> plk(pairs_mu);
                    const auto pr = std::make_pair(task.layer, task.expert);
                    if (std::find(resident_pairs.begin(), resident_pairs.end(), pr) ==
                        resident_pairs.end()) {
                        resident_pairs.push_back(pr);
                    }
                }
                wire::AckMsg ack{};
                ack.type = uint32_t(wire::AckType::PromoteAck);
                ack.seq = task.ctrl_seq;
                ack.promote_ack = {task.layer, task.expert, task.gen};
                push_ack_safe(ack);
                async_promotes_done.fetch_add(1, std::memory_order_relaxed);
            } else {
                push_error_safe(task.ctrl_seq, perr, /*fatal=*/false);
            }

            lk.lock();
            refill_busy = false;
            refill_active = {-1, -1};
            refill_cv.notify_all();
        }
    });

    auto drain_refill = [&] {
        std::unique_lock<std::mutex> lk(refill_mu);
        refill_cv.wait(lk, [&] { return refill_queue.empty() && !refill_busy; });
    };
    auto stop_refill = [&] {
        {
            std::lock_guard<std::mutex> lk(refill_mu);
            refill_stop = true;
            refill_queue.clear();
        }
        refill_cv.notify_all();
        if (refill_thread.joinable()) refill_thread.join();
    };

    std::vector<float> staging_out(size_t(caps.max_entries_per_window) * size_t(g.n_embd), 0.0f);
    uint64_t last_seq = cb->doorbell_seq.value.load(std::memory_order_acquire);
    bool running = true;
    bool ok = true;

    auto should_stop = [&]() { return g_signal_stop.load(std::memory_order_acquire); };

    // 3. Main event loop: control messages + doorbell rounds.
    while (running && !should_stop()) {
        ipc::wait_agent_wakeup(seg, last_seq, cfg.spin_us, cfg.park_slice_us, should_stop);
        if (should_stop()) break;

        // 3a. Drain control ring.
        wire::CtrlMsg cm{};
        while (running && ipc::ring_pop(*seg.ctrl_ring(), cm)) {
            switch (static_cast<wire::CtrlType>(cm.type)) {
                case wire::CtrlType::Open: {
                    drain_refill();
                    const uint64_t pair_bytes = uint64_t(cm.open.n_fill) * 2ull * sizeof(int32_t);
                    if (!seg.var_range_ok(cm.open.fill_offset, pair_bytes)) {
                        err = "OPEN fill_offset out of segment bounds";
                        push_error_safe(cm.seq, err, /*fatal=*/true);
                        ok = false;
                        running = false;
                        break;
                    }
                    const auto* raw_pairs =
                        reinterpret_cast<const int32_t*>(seg.var_ptr(cm.open.fill_offset));
                    std::vector<std::pair<int32_t, int32_t>> req;
                    req.reserve(cm.open.n_fill);
                    for (uint32_t i = 0; i < cm.open.n_fill; ++i) {
                        req.emplace_back(raw_pairs[2 * i], raw_pairs[2 * i + 1]);
                    }
                    const uint64_t t0 = ipc::now_ns();
                    std::vector<std::pair<int32_t, int32_t>> loaded;
                    std::string ferr;
                    if (!backend.open_fill(req, loaded, ferr)) {
                        err = "OPEN fill failed: " + ferr;
                        push_error_safe(cm.seq, err, /*fatal=*/true);
                        ok = false;
                        running = false;
                        break;
                    }
                    {
                        std::lock_guard<std::mutex> plk(pairs_mu);
                        resident_pairs = std::move(loaded);
                        write_resident_bitmap(seg, g.n_expert, resident_pairs);
                    }
                    wire::AckMsg ack{};
                    ack.type = uint32_t(wire::AckType::OpenAck);
                    ack.seq = cm.seq;
                    ack.open_ack.gen = cm.open.gen;
                    ack.open_ack.n_resident = uint32_t(resident_pairs.size());
                    ack.open_ack.bitmap_offset = seg.plan().agent_bitmap_offset;
                    push_ack_safe(ack);
                    cb->agent_state.value.store(uint32_t(wire::AgentState::Ready),
                                                std::memory_order_release);
                    if (cfg.verbose) {
                        const double fill_s = double(ipc::now_ns() - t0) / 1e9;
                        std::fprintf(stderr,
                                     "franken-agent (%s): OPEN_ACK gen=%u loaded %zu / %u experts in %.2f s -> Ready\n",
                                     vendor_name(caps.vendor), cm.open.gen,
                                     resident_pairs.size(), cm.open.n_fill, fill_s);
                    }
                    break;
                }
                case wire::CtrlType::Promote: {
                    // Non-blocking enqueue to background refill worker (Phase 3).
                    {
                        std::lock_guard<std::mutex> lk(refill_mu);
                        refill_queue.push_back(PromoteTask{
                            cm.promote.layer, cm.promote.expert, cm.promote.gen, cm.seq});
                    }
                    refill_cv.notify_one();
                    break;
                }
                case wire::CtrlType::Demote: {
                    const auto target = std::make_pair(cm.demote.layer, cm.demote.expert);
                    // Cancel any queued promote for this exact expert; wait only if it is actively loading right now.
                    {
                        std::unique_lock<std::mutex> lk(refill_mu);
                        refill_queue.erase(
                            std::remove_if(refill_queue.begin(), refill_queue.end(),
                                           [&](const PromoteTask& t) {
                                               return t.layer == target.first && t.expert == target.second;
                                           }),
                            refill_queue.end());
                        refill_cv.wait(lk, [&] { return !refill_busy || refill_active != target; });
                    }
                    std::string derr;
                    if (backend.demote(cm.demote.layer, cm.demote.expert, derr)) {
                        {
                            std::lock_guard<std::mutex> plk(pairs_mu);
                            resident_pairs.erase(
                                std::remove(resident_pairs.begin(), resident_pairs.end(), target),
                                resident_pairs.end());
                        }
                        wire::AckMsg ack{};
                        ack.type = uint32_t(wire::AckType::DemoteAck);
                        ack.seq = cm.seq;
                        ack.demote_ack = {cm.demote.layer, cm.demote.expert, cm.demote.gen};
                        push_ack_safe(ack);
                    } else {
                        push_error_safe(cm.seq, derr, /*fatal=*/false);
                    }
                    break;
                }
                case wire::CtrlType::Resync: {
                    drain_refill();
                    uint32_t n_res = 0;
                    {
                        std::lock_guard<std::mutex> plk(pairs_mu);
                        write_resident_bitmap(seg, g.n_expert, resident_pairs);
                        n_res = uint32_t(resident_pairs.size());
                    }
                    wire::AckMsg ack{};
                    ack.type = uint32_t(wire::AckType::OpenAck);
                    ack.seq = cm.seq;
                    ack.open_ack.gen = cm.resync.gen;
                    ack.open_ack.n_resident = n_res;
                    ack.open_ack.bitmap_offset = seg.plan().agent_bitmap_offset;
                    push_ack_safe(ack);
                    cb->agent_state.value.store(uint32_t(wire::AgentState::Ready),
                                                std::memory_order_release);
                    break;
                }
                case wire::CtrlType::Bye: {
                    if (cfg.verbose) {
                        std::fprintf(stderr, "franken-agent (%s): received BYE, shutting down cleanly\n",
                                     vendor_name(caps.vendor));
                    }
                    running = false;
                    break;
                }
                default: {
                    push_error_safe(cm.seq, "unknown CtrlType", /*fatal=*/false);
                    break;
                }
            }
        }
        if (!running) break;

        // 3b. Process any newly published doorbell rounds.
        const uint64_t cur_seq = cb->doorbell_seq.value.load(std::memory_order_acquire);
        while (last_seq < cur_seq && running && !should_stop()) {
            ++last_seq;
            ++stats.rounds_seen;
            wire::DescriptorHeader* dh = seg.descriptor(last_seq);
            if (!ipc::try_claim_slot(dh)) {
                ++stats.rounds_salvaged_before_claim;
                continue;
            }
            if (dh->seq != last_seq ||
                dh->payload_kind != uint32_t(wire::PayloadKind::ExpertRows) ||
                dh->n_tok <= 0 || dh->k <= 0 ||
                uint32_t(dh->n_tok) > caps.max_tokens_per_window ||
                uint32_t(dh->n_tok * dh->k) > caps.max_entries_per_window ||
                dh->layer < 0 || dh->layer >= g.n_layers) {
                err = "invalid descriptor header on seq " + std::to_string(last_seq);
                push_error_safe(last_seq, err, /*fatal=*/true);
                ok = false;
                running = false;
                break;
            }

            const int32_t n_entries = dh->n_tok * dh->k;
            const uint32_t dflags = dh->flags;
            const bool compact_out = (dflags & wire::kDescFlagCompactOut) != 0;
            const int32_t* ids = wire::slot_ids(dh);
            const int32_t* kind = wire::slot_kind(dh);
            const void* x_data = static_cast<const void*>(wire::slot_x(dh));

            if (dh->n_agent_entries > 0) {
                const uint64_t tc0 = ipc::now_ns();
                std::string cerr;
                if (!backend.compute_slot_ex(dh->layer, dh->n_tok, dh->k, ids, kind, x_data,
                                             dflags, staging_out.data(), cerr)) {
                    err = "compute_slot failed on layer " + std::to_string(dh->layer) + ": " + cerr;
                    push_error_safe(last_seq, err, /*fatal=*/true);
                    ok = false;
                    running = false;
                    break;
                }
                const uint64_t tc1 = ipc::now_ns();
                stats.compute_ms += double(tc1 - tc0) / 1e6;
                ++stats.rounds_computed;
                stats.entries_computed += uint64_t(dh->n_agent_entries);
                if ((dflags & wire::kDescFlagActQ8_1) != 0) ++stats.q8_1_rounds;
                if (compact_out) ++stats.compact_rounds;

                float* row_arena = seg.row_slot(last_seq);
                const size_t row_bytes = size_t(g.n_embd) * sizeof(float);
                const int32_t n_agent = dh->n_agent_entries;
                const bool committed = ipc::commit_slot_if_claimed(dh, [&] {
                    if (compact_out) {
                        std::memcpy(row_arena, staging_out.data(), size_t(n_agent) * row_bytes);
                    } else {
                        for (int32_t e = 0; e < n_entries; ++e) {
                            if (kind[e] != 3) continue;
                            std::memcpy(row_arena + size_t(e) * size_t(g.n_embd),
                                        staging_out.data() + size_t(e) * size_t(g.n_embd),
                                        row_bytes);
                        }
                    }
                });
                const uint64_t tc2 = ipc::now_ns();
                stats.commit_ms += double(tc2 - tc1) / 1e6;
                if (!committed) {
                    ++stats.rounds_salvaged_before_commit;
                }
            } else {
                (void)ipc::commit_slot_if_claimed(dh, {});
            }
        }
    }

    stop_refill();
    stats.async_promotes_completed = async_promotes_done.load(std::memory_order_relaxed);
    stop_heartbeat();
    cb->agent_state.value.store(uint32_t(wire::AgentState::Dead), std::memory_order_release);
    backend.shutdown();

    if (cfg.verbose) {
        std::fprintf(stderr,
                     "franken-agent (%s): exit (rounds_seen=%llu computed=%llu entries=%llu "
                     "q8_1=%llu compact=%llu async_promotes=%llu salvaged=%llu/%llu "
                     "avg_compute=%.3f ms avg_commit=%.3f ms)\n",
                     vendor_name(caps.vendor),
                     (unsigned long long)stats.rounds_seen,
                     (unsigned long long)stats.rounds_computed,
                     (unsigned long long)stats.entries_computed,
                     (unsigned long long)stats.q8_1_rounds,
                     (unsigned long long)stats.compact_rounds,
                     (unsigned long long)stats.async_promotes_completed,
                     (unsigned long long)stats.rounds_salvaged_before_claim,
                     (unsigned long long)stats.rounds_salvaged_before_commit,
                     stats.rounds_computed > 0 ? stats.compute_ms / double(stats.rounds_computed) : 0.0,
                     stats.rounds_computed > 0 ? stats.commit_ms / double(stats.rounds_computed) : 0.0);
    }
    return ok;
}

}  // namespace agent
}  // namespace franken
