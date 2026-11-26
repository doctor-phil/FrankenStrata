// franken_agent_sycl.cpp — Intel Arc (SYCL) foreign-vendor expert tier agent.
//
// Runs on Intel Arc GPUs (validated target: Arc Pro B70 32 GB) inside the SYCL
// runtime (or Docker container sharing /dev/shm + /dev/dri). Connects to the
// primary's POSIX shm segment, loads its assigned experts directly from the
// pack/GGUF shards into SYCL ExpertCache VRAM, and computes kind == 3 entries
// using native_expert_grouped (native IQ packs, with Hadamard support) or
// moe_grouped_s2 (canonical s2 packs).

#include "franken/agent/agent_loop.hpp"
#include "franken/wire/pack_identity.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/copy_queue.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fw = franken::wire;
namespace fi = franken::ipc;
namespace fa = franken::agent;
namespace cpu = strata::kernels::cpu;

// Coherency self-test for host-pinned USM staging (analogous to p2p_copies_data
// in src/core/peer_experts.cpp): round-trips a pseudo-random pattern H2D -> D2H
// through USM host memory and verifies bit-exact equality before relying on it.
bool verify_usm_staging_coherency(sycl::queue* q) {
    constexpr size_t kN = 16384;  // 64 KiB
    float* h = nullptr;
    float* d = nullptr;
    bool ok = false;
    try {
        h = sycl::malloc_host<float>(kN, *q);
        d = sycl::malloc_device<float>(kN, *q);
        if (h != nullptr && d != nullptr) {
            std::vector<float> ref(kN);
            for (size_t i = 0; i < kN; ++i) {
                ref[i] = 1.0f + float((i * 2654435761u) % 4093u);
                h[i] = ref[i];
            }
            q->memcpy(d, h, kN * sizeof(float));
            q->wait_and_throw();
            std::memset(h, 0, kN * sizeof(float));
            q->memcpy(h, d, kN * sizeof(float));
            q->wait_and_throw();
            ok = (std::memcmp(h, ref.data(), kN * sizeof(float)) == 0);
        }
    } catch (...) {
        ok = false;
    }
    if (d != nullptr) sycl::free(d, *q);
    if (h != nullptr) sycl::free(h, *q);
    return ok;
}

// Single contiguous metadata upload per round (mirrors PeerExperts::Meta).
struct SyclMeta {
    unsigned long long ptr[fw::kDefaultMaxEntries];
    int32_t start[fw::kDefaultMaxEntries + 1];
    int32_t dst[fw::kDefaultMaxEntries];
    int32_t tok[fw::kDefaultMaxEntries];
    int32_t count[4];
};

class SyclAgentBackend final : public fa::AgentBackend {
public:
    SyclAgentBackend(std::string pack_dir, std::string native_gguf,
                     uint32_t max_slots, uint32_t reserve_mib)
        : pack_dir_(std::move(pack_dir)),
          native_gguf_(std::move(native_gguf)),
          max_slots_(max_slots),
          reserve_mib_(reserve_mib) {}

    ~SyclAgentBackend() override { shutdown(); }

    fw::AgentVendor vendor() const override { return fw::AgentVendor::Sycl; }

    bool init(const fw::Geometry& g,
              const uint64_t* blob_bytes,
              fw::Caps& out_caps,
              fw::PackIdentity& out_pack,
              std::string& err) override {
        g_ = g;
        if (!cpu::expert_layout_load(pack_dir_, g.n_layers, g.n_expert, err)) {
            return false;
        }
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        for (int32_t l = 0; l < g.n_layers; ++l) {
            if (lay.blob_bytes(l) != blob_bytes[l]) {
                err = "SyclAgentBackend: blob_bytes mismatch at layer " + std::to_string(l);
                return false;
            }
        }
        if (!fi::compute_pack_identity(pack_dir_, native_gguf_, lay, out_pack, err)) {
            return false;
        }

        const bool has_experts_bin =
            std::filesystem::exists(std::filesystem::path(pack_dir_) / "experts.bin");
        if (lay.native && !has_experts_bin) {
            if (native_gguf_.empty()) {
                err = "SyclAgentBackend: native pack has no experts.bin; pass --native <shard1.gguf>";
                return false;
            }
            gguf_src_ = std::make_unique<strata::core::GgufExpertSource>();
            if (!gguf_src_->open(native_gguf_, g.n_layers, g.n_expert, err)) {
                gguf_src_.reset();
                return false;
            }
            src_ = gguf_src_.get();
        } else {
            file_src_ = std::make_unique<strata::core::FileExpertSource>();
            if (!native_gguf_.empty()) file_src_->set_gguf(native_gguf_);
            if (!file_src_->open(pack_dir_, g.n_layers, g.n_expert, err)) {
                file_src_.reset();
                return false;
            }
            src_ = file_src_.get();
        }

        q_ = &dpct::get_in_order_queue();
        const std::string dev_name = q_->get_device().get_info<sycl::info::device::name>();

        // Verify USM host-pinned staging coherency before enabling it.
        usm_staging_ok_ = verify_usm_staging_coherency(q_);
        std::fprintf(stderr, "franken-agent-sycl: USM host staging coherency self-test: %s\n",
                     usm_staging_ok_ ? "PASS" : "FAIL (falling back to pageable staging)");

        // Dedicated background copy queue for non-blocking async PROMOTE refills.
        refill_q_ = strata::make_copy_engine_queue("franken-promote");
        if (refill_q_ == nullptr) {
            refill_q_ = dpct::get_current_device().create_queue(true);
            owns_refill_q_ = true;
        }
        std::fprintf(stderr, "franken-agent-sycl: async PROMOTE refill queue: %s\n",
                     owns_refill_q_ ? "dedicated compute-engine queue" : "Level Zero copy engine");

        // Allocate compute workspace BEFORE measuring remaining VRAM for expert slots.
        cap_tok_ = std::min<uint32_t>(g.max_tokens, fw::kDefaultMaxTokens);
        cap_ent_ = std::min<uint32_t>(g.max_entries, fw::kDefaultMaxEntries);
        int64_t ff = cpu::FF;
        if (lay.native) {
            for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
        }
        const size_t scratch_bytes = std::max<size_t>(
            size_t(strata::kernels::moe_hit_grouped_scratch_bytes(cap_ent_, g.n_embd, ff)),
            strata::kernels::native_expert_scratch_bytes(cap_ent_, ff));
        const size_t q8_bytes = std::max<size_t>(
            size_t(cap_tok_) * size_t(g.n_embd / 32) * 36,
            size_t(cap_tok_) * size_t(g.n_embd / 32) * 34);

        try {
            if (usm_staging_ok_) {
                h_x_ = sycl::malloc_host<float>(size_t(cap_tok_) * size_t(g.n_embd), *q_);
                h_q8_ = sycl::malloc_host<uint8_t>(q8_bytes, *q_);
                h_out_ = sycl::malloc_host<float>(size_t(cap_ent_) * size_t(g.n_embd), *q_);
                h_meta_ = sycl::malloc_host<SyclMeta>(1, *q_);
                refill_bounce_usm_ = sycl::malloc_host<uint8_t>(size_t(lay.max_blob), *q_);
            } else {
                h_x_vec_.resize(size_t(cap_tok_) * size_t(g.n_embd));
                h_q8_vec_.resize(q8_bytes);
                h_out_vec_.resize(size_t(cap_ent_) * size_t(g.n_embd));
                h_meta_vec_.resize(1);
                h_x_ = h_x_vec_.data();
                h_q8_ = h_q8_vec_.data();
                h_out_ = h_out_vec_.data();
                h_meta_ = h_meta_vec_.data();
            }
            if (refill_bounce_usm_ == nullptr) refill_bounce_vec_.resize(size_t(lay.max_blob));
            d_x_ = sycl::malloc_device<float>(size_t(cap_tok_) * size_t(g.n_embd), *q_);
            d_out_ = sycl::malloc_device<float>(size_t(cap_ent_) * size_t(g.n_embd), *q_);
            d_meta_ = sycl::malloc_device<SyclMeta>(1, *q_);
            d_q8_ = sycl::malloc_device<uint8_t>(q8_bytes, *q_);
            d_scales_ = sycl::malloc_device<float>(size_t(cap_tok_) * size_t(g.n_embd / 32), *q_);
            d_scratch_ = sycl::malloc_device(scratch_bytes, *q_);
        } catch (const std::exception& ex) {
            err = std::string("SyclAgentBackend: workspace alloc failed: ") + ex.what();
            shutdown();
            return false;
        }
        if (!h_x_ || !h_q8_ || !h_out_ || !h_meta_ || !d_x_ || !d_out_ ||
            !d_meta_ || !d_q8_ || !d_scales_ || !d_scratch_) {
            err = "SyclAgentBackend: SYCL malloc returned nullptr";
            shutdown();
            return false;
        }

        if (cpu::hadamard_spec().any()) {
            d_sx_ = strata::kernels::hadamard_device_signs(cpu::hadamard_signs(g.n_embd), g.n_embd);
            d_sh_ = strata::kernels::hadamard_device_signs(cpu::hadamard_signs(int(ff)), int(ff));
        }

        // Estimate available slot budget from device_free_bytes() minus reserve.
        const size_t free_b = strata::core::device_free_bytes();
        const uint64_t reserve_b = uint64_t(std::max<uint32_t>(reserve_mib_, 256)) << 20;
        vram_budget_ = free_b > reserve_b ? uint64_t(free_b) - reserve_b : 0;

        uint64_t avg_blob = 0;
        for (int32_t l = 0; l < g.n_layers; ++l) {
            const uint64_t b = lay.blob_bytes(l);
            avg_blob += lay.native ? ((b + 255) / 256 * 256) : lay.max_blob;
        }
        avg_blob = std::max<uint64_t>(1, avg_blob / uint64_t(g.n_layers));

        const uint32_t total_experts = uint32_t(g.n_layers * g.n_expert);
        uint32_t est_slots = uint32_t(std::min<uint64_t>(vram_budget_ / avg_blob, total_experts));
        if (max_slots_ > 0) est_slots = std::min(est_slots, max_slots_);

        std::fprintf(stderr,
                     "franken-agent-sycl: device '%s', free %.2f GiB, reserve %u MiB, "
                     "budget %.2f GiB -> advertising %u slots\n",
                     dev_name.c_str(), double(free_b) / 1073741824.0, reserve_mib_,
                     double(vram_budget_) / 1073741824.0, est_slots);

        out_caps.max_tokens_per_window = cap_tok_;
        out_caps.max_entries_per_window = cap_ent_;
        out_caps.slot_count = est_slots;
        out_caps.vendor = uint32_t(fw::AgentVendor::Sycl);
        out_caps.feature_flags = fw::kCapQ8_1Activations | fw::kCapCompactRows;
        return true;
    }

    bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                   std::vector<std::pair<int32_t, int32_t>>& out_loaded,
                   std::string& err) override {
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        cache_.close();
        res_.assign(size_t(g_.n_layers) * size_t(g_.n_expert), strata::core::kNotResident);
        free_slots_.clear();
        out_loaded.clear();

        // Filter duplicates and pack as many pairs as fit the exact per-layer VRAM budget
        // (and optional --slots cap). Short loads truncate cleanly (§4.3).
        std::vector<std::pair<int32_t, int32_t>> pick;
        std::vector<int64_t> sizes;
        std::vector<uint8_t> seen(size_t(g_.n_layers) * size_t(g_.n_expert), 0);
        uint64_t used = 0;
        const uint64_t max_b256 = (lay.max_blob + 255) / 256 * 256;

        for (const auto& [l, e] : pairs) {
            if (l < 0 || l >= g_.n_layers || e < 0 || e >= g_.n_expert) continue;
            const size_t idx = size_t(l) * size_t(g_.n_expert) + size_t(e);
            if (seen[idx]) continue;
            const uint64_t b = lay.blob_bytes(l);
            const uint64_t b256 = lay.native ? ((b + 255) / 256 * 256) : lay.max_blob;
            if (used + b256 > vram_budget_) break;
            if (max_slots_ > 0 && pick.size() >= size_t(max_slots_)) break;
            seen[idx] = 1;
            used += b256;
            pick.emplace_back(l, e);
            sizes.push_back(int64_t(b));
        }

        if (pick.empty()) {
            return true;  // 0 slots loaded is a valid degenerate short-load
        }

        // Allocate uniform max_blob slots when they fit within VRAM budget so dynamic
        // PROMOTE across layers can reuse any freed slot; otherwise use open_sized and
        // enforce slot_room(slot) >= blob_bytes(layer) in promote().
        const bool use_uniform = !lay.native || (uint64_t(pick.size()) * max_b256 <= vram_budget_);
        const bool opened = use_uniform
            ? cache_.open(int64_t(pick.size()), g_.n_layers, g_.n_expert, int64_t(max_b256), err)
            : cache_.open_sized(sizes, g_.n_layers, g_.n_expert, err);
        if (!opened) return false;

        for (const auto& [l, e] : pick) {
            const int32_t slot = cache_.admit(l, e);
            const uint8_t* b = src_->blob(l, e);
            if (slot < 0 || b == nullptr) {
                err = "SyclAgentBackend::open_fill: blob read or admit failed";
                return false;
            }
            if (!cache_.fill_slot_blocking(slot, b, err, int64_t(lay.blob_bytes(l)))) {
                return false;
            }
            res_[size_t(l) * size_t(g_.n_expert) + size_t(e)] = slot;
            out_loaded.emplace_back(l, e);
        }

        // Verify slot 0 byte-for-byte after fill.
        const auto [l0, e0] = out_loaded.front();
        const int32_t s0 = res_[size_t(l0) * size_t(g_.n_expert) + size_t(e0)];
        if (!cache_.verify_slot(s0, src_->blob(l0, e0), err, int64_t(lay.blob_bytes(l0)))) {
            return false;
        }
        std::fprintf(stderr, "franken-agent-sycl: verified slot 0, resident %zu experts (%.2f GiB VRAM, %s slots)\n",
                     out_loaded.size(), cache_.gib(), use_uniform ? "uniform" : "sized");
        return true;
    }

    bool promote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "SyclAgentBackend::promote: (layer, expert) out of range";
            return false;
        }
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        const uint64_t need = lay.blob_bytes(layer);

        // 1. Claim a freed slot with room >= need under res_mu_, then release lock
        //    before performing blob I/O and H2D copy on refill_q_.
        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lk(res_mu_);
            if (res_[idx] >= 0) return true;  // already resident
            size_t best_pos = free_slots_.size();
            uint64_t best_room = ~0ull;
            for (size_t i = 0; i < free_slots_.size(); ++i) {
                const int32_t s = free_slots_[i];
                const uint64_t room = cache_.slot_room(s);
                if (room >= need && room < best_room) {
                    best_room = room;
                    best_pos = i;
                }
            }
            if (best_pos == free_slots_.size()) {
                err = "SyclAgentBackend::promote: no freed slot with room >= " +
                      std::to_string(need) + " B for layer " + std::to_string(layer);
                return false;
            }
            slot = free_slots_[best_pos];
            free_slots_.erase(free_slots_.begin() + ptrdiff_t(best_pos));
        }

        // 2. Thread-safe blob read into dedicated refill bounce buffer + H2D on refill_q_.
        uint8_t* bounce = refill_bounce_usm_ != nullptr ? refill_bounce_usm_ : refill_bounce_vec_.data();
        bool read_ok = false;
        if (gguf_src_) {
            read_ok = gguf_src_->read_into(layer, expert, bounce, size_t(need));
        } else if (src_) {
            read_ok = src_->copy_blob(layer, expert, bounce);
        }
        if (!read_ok) {
            std::lock_guard<std::mutex> lk(res_mu_);
            free_slots_.push_back(slot);
            err = "SyclAgentBackend::promote: thread-safe blob read failed";
            return false;
        }

        try {
            refill_q_->memcpy(cache_.device_slot(slot), bounce, size_t(need)).wait_and_throw();
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(res_mu_);
            free_slots_.push_back(slot);
            err = std::string("SyclAgentBackend::promote: H2D copy failed: ") + ex.what();
            return false;
        }

        // 3. Publish slot in residency table only AFTER copy has completed.
        {
            std::lock_guard<std::mutex> lk(res_mu_);
            res_[idx] = slot;
        }
        return true;
    }

    bool demote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "SyclAgentBackend::demote: (layer, expert) out of range";
            return false;
        }
        std::lock_guard<std::mutex> lk(res_mu_);
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        const int32_t slot = res_[idx];
        if (slot < 0) return true;  // idempotent
        res_[idx] = strata::core::kNotResident;
        free_slots_.push_back(slot);
        return true;
    }

    bool compute_slot(int32_t layer,
                      int32_t n_tok,
                      int32_t k,
                      const int32_t* ids,
                      const int32_t* kind,
                      const float* x,
                      float* staging_out,
                      std::string& err) override {
        return compute_slot_ex(layer, n_tok, k, ids, kind, x, 0u, staging_out, err);
    }

    bool compute_slot_ex(int32_t layer,
                         int32_t n_tok,
                         int32_t k,
                         const int32_t* ids,
                         const int32_t* kind,
                         const void* x_data,
                         uint32_t desc_flags,
                         float* staging_out,
                         std::string& err) override {
        const int64_t n = int64_t(n_tok) * int64_t(k);
        if (n_tok > int32_t(cap_tok_) || n > int64_t(cap_ent_)) {
            err = "SyclAgentBackend: window exceeds agent caps";
            return false;
        }

        const bool act_q8_1 = (desc_flags & fw::kDescFlagActQ8_1) != 0;
        const bool compact_out = (desc_flags & fw::kDescFlagCompactOut) != 0;

        // Map each kind == 3 entry to its router-order compact output index [0..n_agent_entries).
        int32_t compact_of[fw::kDefaultMaxEntries];
        int32_t n_compact = 0;
        for (int64_t i = 0; i < n; ++i) {
            compact_of[i] = (kind[i] == 3) ? n_compact++ : -1;
        }

        SyclMeta& m = *h_meta_;
        row_to_entry_.clear();
        int32_t groups = 0, rows = 0;

        {
            std::lock_guard<std::mutex> lk(res_mu_);
            for (int64_t i = 0; i < n; ++i) {
                if (kind[i] != 3) continue;
                const int32_t e = ids[i];
                bool seen = false;
                for (int64_t j = 0; j < i; ++j) {
                    if (kind[j] == 3 && ids[j] == e) { seen = true; break; }
                }
                if (seen) continue;
                const int32_t slot =
                    (e >= 0 && e < g_.n_expert)
                        ? res_[size_t(layer) * size_t(g_.n_expert) + size_t(e)]
                        : strata::core::kNotResident;
                if (slot < 0) {
                    err = "SyclAgentBackend: routed kind==3 expert (" + std::to_string(layer) +
                          "," + std::to_string(e) + ") not resident";
                    return false;
                }
                m.ptr[groups] = reinterpret_cast<unsigned long long>(cache_.device_slot(slot));
                m.start[groups] = rows;
                for (int64_t j = i; j < n; ++j) {
                    if (kind[j] == 3 && ids[j] == e) {
                        // With compact_out, the kernel writes directly into router-order
                        // compact row `compact_of[j]`, eliminating host-side scatter!
                        m.dst[rows] = compact_out ? compact_of[j] : rows;
                        m.tok[rows] = int32_t(j / k);
                        if (!compact_out) row_to_entry_.push_back(int32_t(j));
                        ++rows;
                    }
                }
                ++groups;
            }
        }
        if (groups == 0) return true;

        m.start[groups] = rows;
        m.count[0] = groups;
        m.count[1] = rows;
        m.count[2] = 0;
        m.count[3] = 0;

        const cpu::ExpertLayout& lay = cpu::expert_layout();
        const size_t q8_1_bytes = size_t(n_tok) * size_t(g_.n_embd / 32) * fw::kQ8_1BlockBytes;
        const size_t fp32_bytes = size_t(n_tok) * size_t(g_.n_embd) * sizeof(float);

        if (act_q8_1 && lay.native) {
            std::memcpy(h_q8_, x_data, q8_1_bytes);
        } else {
            std::memcpy(h_x_, x_data, fp32_bytes);
        }

        try {
            q_->memcpy(d_meta_, h_meta_, sizeof(SyclMeta));

            if (lay.native) {
                const cpu::NativeFmt& f = lay.fmt[size_t(layer)];
                if (act_q8_1 && !f.had_gu) {
                    // Phase 3 fast path: upload pre-quantized q8_1 directly, skip quantize_q8_1_rows!
                    q_->memcpy(d_q8_, h_q8_, q8_1_bytes);
                } else {
                    q_->memcpy(d_x_, h_x_, fp32_bytes);
                    if (f.had_gu) {
                        strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, g_.n_embd,
                                                       f.had_block, d_sx_, q_);
                    }
                    strata::kernels::quantize_q8_1_rows(d_x_, n_tok, g_.n_embd, d_q8_, q_);
                }
                auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                if (f.had_d) {
                    L.had_block = f.had_block;
                    L.d_signs = d_sh_;
                }
                strata::kernels::native_expert_grouped(
                    L, d_meta_->ptr, d_meta_->start, d_meta_->count,
                    d_meta_->dst, d_meta_->tok, groups, rows,
                    d_q8_, d_scratch_, d_out_, q_);
            } else {
                q_->memcpy(d_x_, h_x_, fp32_bytes);
                strata::kernels::quantize_q8_0_scaled(d_x_, d_q8_, d_scales_,
                                                      int64_t(n_tok) * int64_t(g_.n_embd), q_);
                strata::kernels::moe_grouped_s2(
                    d_meta_->ptr, d_meta_->start, d_meta_->count,
                    d_meta_->dst, d_meta_->tok, groups, rows,
                    d_q8_, d_scales_, d_scratch_, d_out_, q_);
            }

            q_->memcpy(h_out_, d_out_, size_t(rows) * size_t(g_.n_embd) * sizeof(float));
            // Strictly synchronize before returning (DESIGN §4.4 step 4).
            q_->wait_and_throw();
        } catch (const std::exception& ex) {
            err = std::string("SyclAgentBackend: SYCL execution error: ") + ex.what();
            return false;
        }

        const size_t row_bytes = size_t(g_.n_embd) * sizeof(float);
        if (compact_out) {
            std::memcpy(staging_out, h_out_, size_t(rows) * row_bytes);
        } else {
            for (int32_t r = 0; r < rows; ++r) {
                std::memcpy(staging_out + size_t(row_to_entry_[size_t(r)]) * size_t(g_.n_embd),
                            h_out_ + size_t(r) * size_t(g_.n_embd),
                            row_bytes);
            }
        }
        return true;
    }

    void shutdown() override {
        if (refill_q_ != nullptr) {
            try { refill_q_->wait_and_throw(); } catch (...) {}
            if (owns_refill_q_) {
                dpct::get_current_device().destroy_queue(refill_q_);
                owns_refill_q_ = false;
            }
            refill_q_ = nullptr;
        }
        if (q_ != nullptr) {
            try { q_->wait_and_throw(); } catch (...) {}
            auto free_ptr = [&](auto*& p) {
                if (p != nullptr) { sycl::free(p, *q_); p = nullptr; }
            };
            if (usm_staging_ok_) {
                free_ptr(h_x_);
                free_ptr(h_q8_);
                free_ptr(h_out_);
                free_ptr(h_meta_);
                free_ptr(refill_bounce_usm_);
            } else {
                h_x_ = nullptr;
                h_q8_ = nullptr;
                h_out_ = nullptr;
                h_meta_ = nullptr;
            }
            free_ptr(d_x_);
            free_ptr(d_out_);
            free_ptr(d_meta_);
            free_ptr(d_q8_);
            free_ptr(d_scales_);
            free_ptr(d_scratch_);
        }
        h_x_vec_.clear();
        h_q8_vec_.clear();
        h_out_vec_.clear();
        h_meta_vec_.clear();
        refill_bounce_vec_.clear();
        cache_.close();
        res_.clear();
        free_slots_.clear();
        gguf_src_.reset();
        file_src_.reset();
        src_ = nullptr;
        d_sx_ = nullptr;
        d_sh_ = nullptr;
        q_ = nullptr;
    }

private:
    std::string pack_dir_;
    std::string native_gguf_;
    uint32_t max_slots_ = 0;
    uint32_t reserve_mib_ = 1024;
    fw::Geometry g_{};
    uint32_t cap_tok_ = 0, cap_ent_ = 0;
    uint64_t vram_budget_ = 0;
    bool usm_staging_ok_ = false;

    std::unique_ptr<strata::core::GgufExpertSource> gguf_src_;
    std::unique_ptr<strata::core::FileExpertSource> file_src_;
    strata::core::ExpertSource* src_ = nullptr;

    sycl::queue* q_ = nullptr;
    sycl::queue* refill_q_ = nullptr;
    bool owns_refill_q_ = false;
    strata::core::ExpertCache cache_;
    std::mutex res_mu_;
    std::vector<int32_t> res_;
    std::vector<int32_t> free_slots_;

    float* h_x_ = nullptr;
    uint8_t* h_q8_ = nullptr;
    float* h_out_ = nullptr;
    SyclMeta* h_meta_ = nullptr;
    uint8_t* refill_bounce_usm_ = nullptr;
    std::vector<float> h_x_vec_;
    std::vector<uint8_t> h_q8_vec_;
    std::vector<float> h_out_vec_;
    std::vector<SyclMeta> h_meta_vec_;
    std::vector<uint8_t> refill_bounce_vec_;
    float* d_x_ = nullptr;
    float* d_out_ = nullptr;
    SyclMeta* d_meta_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    const float* d_sx_ = nullptr;
    const float* d_sh_ = nullptr;
    std::vector<int32_t> row_to_entry_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string shm_name, pack_dir, native_gguf;
    uint32_t max_slots = 0;
    uint32_t reserve_mib = 1024;
    uint32_t wait_ms = 30000;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--shm") shm_name = argv[i + 1];
        else if (a == "--pack") pack_dir = argv[i + 1];
        else if (a == "--native") native_gguf = argv[i + 1];
        else if (a == "--slots") max_slots = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--reserve-mib") reserve_mib = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--wait-ms") wait_ms = uint32_t(std::stoul(argv[i + 1]));
        else {
            std::fprintf(stderr, "franken-agent-sycl: unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (shm_name.empty() || pack_dir.empty()) {
        std::fprintf(stderr,
                     "usage: franken-agent-sycl --shm <name> --pack <dir> "
                     "[--native <shard1.gguf>] [--slots <N>] [--reserve-mib <M>] [--wait-ms <ms>]\n");
        return 2;
    }

    fa::install_agent_signal_handlers();
    SyclAgentBackend backend(pack_dir, native_gguf, max_slots, reserve_mib);
    fa::AgentLoopConfig cfg{};
    cfg.shm_name = shm_name;
    cfg.attach_timeout_ms = wait_ms;
    fa::AgentLoopStats stats{};
    std::string err;
    if (!fa::run_agent_loop(cfg, backend, stats, err)) {
        std::fprintf(stderr, "franken-agent-sycl: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
