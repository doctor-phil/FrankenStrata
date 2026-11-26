// agent_spike.cpp — FrankenStrata Phase 0 agent spike (DESIGN §7, Phase 0).
//
// THE SPIKE SHAPE (§7): "a minimal agent skeleton (no protocol) that opens the
// validation pack with the SYCL ports, fills the SYCL ExpertCache, and
// reproduces CPU-computed rows for recorded (activation, ids) inputs. This is
// the extraction-risk burn-down (§11)."
//
// What this file is:
//   - The extraction skeleton for the SYCL agent: pack source, cache fill,
//     grouped compute, and the §8.2 row-comparison statistics, structured
//     around the upstream SYCL ports listed in DESIGN §6.2.
//   - The recorded-input replay harness that Phase 0 uses for the divergence
//     control (§8.2 step 1) and that grows into Phase 1's validation harness.
//   - Deliberately NO protocol: no segment, no doorbell, no kind 3. The full
//     agent loop (§6.1) arrives in Phase 1 around franken/wire/franken_wire.hpp.
//
// What it is not: a working agent. Every body that needs the SYCL runtime or
// an upstream engine class is a TODO stub that fails loudly (returns false /
// throws), so the skeleton can never silently produce numbers. The host-side
// pieces — recorded-window loading, the fill-list builder, the epsilon
// statistics — are real code.
//
// Exit criterion this spike serves (§7 Phase 0): "spike rows within the SYCL
// parity tests' own tolerances against the CPU path" — i.e. its epsilon
// distribution must sit inside what sycl/src/kernels/s2_expert_grouped_parity
// .cpp already accepts for the same kernels, measured against CPU rows
// (kind -1), the engine's reference implementation of an expert (§8.2).
//
// Build note: this file compiles standalone (host C++17, no SYCL) so its
// structure and statistics can be exercised in CI / on GPU-less machines —
// the §14 conformance-floor idea applied to the spike. The real build links
// it into the SYCL tree and fills the TODO bodies with the ports below.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(STRATA_FRANKEN_SYCL_SPIKE)
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl_queue.hpp"
#endif

namespace spike {

// ---------------------------------------------------------------------------
// Recorded inputs (§7 Phase 0, §8.2 step 1): windows captured from real
// engine sessions — per MoE layer, the fp32 activations the session
// published and the routed ids it chose. The spike replays them offline.
//
// File format (little-endian, written by the capture side — Phase 0 work in
// the primary tree, format owned here):
//   char[4]  magic "FRKW"
//   u32      format version (1)
//   u32      n_layers (L)          u32 n_expert (E)
//   u32      k                     u32 n_embd (H)
//   then per window:
//     u32 layer  u32 n_tok
//     i32 ids[n_tok * k]
//     f32 x[n_tok * H]
// ---------------------------------------------------------------------------

struct Window {
    int32_t layer = 0;
    int32_t n_tok = 0;
    std::vector<int32_t> ids;  // [n_tok * k]
    std::vector<float> x;      // [n_tok * H]
};

struct RecordedSet {
    int32_t n_layers = 0;
    int32_t n_expert = 0;
    int32_t k = 0;
    int32_t n_embd = 0;
    std::vector<Window> windows;
};

// Real code: strict loader; any truncation or magic mismatch is an error,
// never a partial read (a half-loaded window would corrupt the statistics).
inline RecordedSet load_recorded(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open recorded inputs: " + path);
    auto read = [&](void* dst, size_t bytes) {
        in.read(reinterpret_cast<char*>(dst), std::streamsize(bytes));
        if (!in) throw std::runtime_error("truncated recorded inputs: " + path);
    };
    char magic[4];
    read(magic, 4);
    if (std::memcmp(magic, "FRKW", 4) != 0)
        throw std::runtime_error("bad magic in recorded inputs: " + path);
    uint32_t version = 0;
    read(&version, 4);
    if (version != 1) throw std::runtime_error("unsupported recorded format version");
    RecordedSet rs;
    read(&rs.n_layers, 4);
    read(&rs.n_expert, 4);
    read(&rs.k, 4);
    read(&rs.n_embd, 4);
    while (in.peek() != std::char_traits<char>::eof()) {
        Window w;
        uint32_t layer = 0, n_tok = 0;
        read(&layer, 4);
        read(&n_tok, 4);
        w.layer = int32_t(layer);
        w.n_tok = int32_t(n_tok);
        w.ids.resize(size_t(w.n_tok) * size_t(rs.k));
        w.x.resize(size_t(w.n_tok) * size_t(rs.n_embd));
        read(w.ids.data(), w.ids.size() * sizeof(int32_t));
        read(w.x.data(), w.x.size() * sizeof(float));
        rs.windows.push_back(std::move(w));
    }
    return rs;
}

// ---------------------------------------------------------------------------
// Pack source — the extraction seam over the SYCL FileExpertSource ports
// (DESIGN §6.2: sycl/src/core/expert_source.cpp and
// sycl/src/core/gguf_expert_source.cpp, FileExpertSource ports over
// experts.bin and native GGUF role slices).
// ---------------------------------------------------------------------------

class PackSource {
public:
    // Opens the pack directory and verifies it against the expected geometry
    // (the spike's stand-in for the §4.3 pack-identity check).
    bool open(const std::string& pack_dir, const std::string& native_gguf,
              int32_t n_layers, int32_t n_expert, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        if (!strata::kernels::cpu::expert_layout_load(pack_dir, n_layers, n_expert, err)) {
            return false;
        }
        const auto& lay = strata::kernels::cpu::expert_layout();
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        const bool has_experts_bin =
            std::filesystem::exists(std::filesystem::path(pack_dir) / "experts.bin");
        if (lay.native && !has_experts_bin) {
            if (native_gguf.empty()) {
                err = "PackSource::open: native pack has no experts.bin; pass --native <shard1.gguf>";
                return false;
            }
            gguf_src_ = std::make_unique<strata::core::GgufExpertSource>();
            if (!gguf_src_->open(native_gguf, n_layers, n_expert, err)) {
                gguf_src_.reset();
                return false;
            }
            src_ = gguf_src_.get();
        } else {
            file_src_ = std::make_unique<strata::core::FileExpertSource>();
            if (!native_gguf.empty()) file_src_->set_gguf(native_gguf);
            if (!file_src_->open(pack_dir, n_layers, n_expert, err)) {
                file_src_.reset();
                return false;
            }
            src_ = file_src_.get();
        }
        return true;
#else
        err = "PackSource::open: SYCL FileExpertSource port not linked (compile with STRATA_FRANKEN_SYCL_SPIKE)";
        (void)pack_dir; (void)native_gguf; (void)n_layers; (void)n_expert;
        return false;
#endif
    }

    // Bytes of one expert blob, host-readable (the CPU path's view).
    const uint8_t* blob(int32_t layer, int32_t expert) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        return src_ ? src_->blob(layer, expert) : nullptr;
#else
        (void)layer; (void)expert;
        return nullptr;
#endif
    }

    uint64_t blob_bytes(int32_t layer) const {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        return strata::kernels::cpu::expert_layout().blob_bytes(layer);
#else
        (void)layer;
        return 0;
#endif
    }

    bool is_native() const {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        return strata::kernels::cpu::expert_layout().native;
#else
        return false;
#endif
    }

#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    strata::core::ExpertSource* raw() { return src_; }
#endif

private:
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    int32_t n_layers_ = 0;
    int32_t n_expert_ = 0;
    std::unique_ptr<strata::core::GgufExpertSource> gguf_src_;
    std::unique_ptr<strata::core::FileExpertSource> file_src_;
    strata::core::ExpertSource* src_ = nullptr;
#endif
};

// ---------------------------------------------------------------------------
// Fill list (the spike's static fill; DESIGN §4.3 step 3 computes the same
// list for OPEN, and §13 Q5 fixes the policy): ranked (layer, expert) pairs,
// hottest first, from the expert profile; absent a profile, degrade to
// layer-early.
// ---------------------------------------------------------------------------

using RankedFill = std::vector<std::pair<int32_t, int32_t>>;

inline RankedFill layer_early_fill(int32_t n_layers, int32_t n_expert,
                                   int64_t slot_budget) {
    RankedFill out;
    for (int32_t l = 0; l < n_layers && int64_t(out.size()) < slot_budget; ++l)
        for (int32_t e = 0; e < n_expert && int64_t(out.size()) < slot_budget; ++e)
            out.emplace_back(l, e);
    return out;
}

inline RankedFill load_profile_fill(const std::string& profile_bin,
                                    int32_t n_layers, int32_t n_expert,
                                    int64_t slot_budget, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    RankedFill ranked;
    int64_t pslots = 0;
    if (!strata::core::read_expert_profile(profile_bin, n_layers, n_expert, ranked, pslots, err)) {
        return {};
    }
    if (slot_budget > 0 && int64_t(ranked.size()) > slot_budget) {
        ranked.resize(size_t(slot_budget));
    }
    return ranked;
#else
    (void)profile_bin; (void)n_layers; (void)n_expert; (void)slot_budget;
    err = "load_profile_fill: SYCL ExpertCache profile reader not linked";
    return {};
#endif
}

// ---------------------------------------------------------------------------
// Agent cache — the extraction seam over the SYCL ExpertCache port
// (DESIGN §6.2: sycl/src/core/expert_cache.cpp; note its open_sized applies
// no per-slot guard, expert_cache.hpp:130 — the spike therefore validates
// slot counts itself before filling).
// ---------------------------------------------------------------------------

class AgentCache {
public:
    AgentCache() = default;
    ~AgentCache() { close(); }
    AgentCache(const AgentCache&) = delete;
    AgentCache& operator=(const AgentCache&) = delete;

    void close() {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        if (q_ != nullptr) {
            try { q_->wait_and_throw(); } catch (...) {}
            auto free_dev = [&](auto*& p) {
                if (p != nullptr) { sycl::free(p, *q_); p = nullptr; }
            };
            free_dev(d_x_);
            free_dev(d_q8_);
            free_dev(d_scales_);
            free_dev(d_scratch_);
            free_dev(d_out_);
            free_dev(d_ptr_);
            free_dev(d_start_);
            free_dev(d_count_);
            free_dev(d_dst_);
            free_dev(d_tok_);
        }
        cache_.close();
        res_.clear();
        d_sx_ = nullptr;
        d_sh_ = nullptr;
        q_ = nullptr;
#endif
    }

    bool open_for_fill(const RankedFill& fill, int32_t n_layers, int32_t n_expert,
                       int32_t max_tokens, int32_t max_entries, int32_t H,
                       std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        close();
        if (fill.empty()) {
            err = "AgentCache::open_for_fill: empty fill list";
            return false;
        }
        q_ = &dpct::get_in_order_queue();
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        max_tokens_ = max_tokens;
        max_entries_ = max_entries;
        H_ = H;

        const auto& lay = strata::kernels::cpu::expert_layout();
        int64_t ff = strata::kernels::cpu::FF;
        if (lay.native) {
            for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
        }
        const size_t scratch_bytes = std::max<size_t>(
            size_t(strata::kernels::moe_hit_grouped_scratch_bytes(max_entries_, H_, ff)),
            strata::kernels::native_expert_scratch_bytes(max_entries_, ff));
        const size_t q8_bytes = std::max<size_t>(
            size_t(max_tokens_) * size_t(H_ / 32) * 36,
            size_t(max_tokens_) * size_t(H_ / 32) * 34);

        try {
            d_x_ = sycl::malloc_device<float>(size_t(max_tokens_) * size_t(H_), *q_);
            d_q8_ = sycl::malloc_device<uint8_t>(q8_bytes, *q_);
            d_scales_ = sycl::malloc_device<float>(size_t(max_tokens_) * size_t(H_ / 32), *q_);
            d_scratch_ = sycl::malloc_device(scratch_bytes, *q_);
            d_out_ = sycl::malloc_device<float>(size_t(max_entries_) * size_t(H_), *q_);
            d_ptr_ = sycl::malloc_device<unsigned long long>(size_t(max_entries_), *q_);
            d_start_ = sycl::malloc_device<int32_t>(size_t(max_entries_) + 1, *q_);
            d_count_ = sycl::malloc_device<int32_t>(4, *q_);
            d_dst_ = sycl::malloc_device<int32_t>(size_t(max_entries_), *q_);
            d_tok_ = sycl::malloc_device<int32_t>(size_t(max_entries_), *q_);
        } catch (const std::exception& ex) {
            err = std::string("AgentCache: device workspace alloc failed: ") + ex.what();
            close();
            return false;
        }
        if (!d_x_ || !d_q8_ || !d_scales_ || !d_scratch_ || !d_out_ ||
            !d_ptr_ || !d_start_ || !d_count_ || !d_dst_ || !d_tok_) {
            err = "AgentCache: device workspace malloc_device returned nullptr";
            close();
            return false;
        }
        if (strata::kernels::cpu::hadamard_spec().any()) {
            d_sx_ = strata::kernels::hadamard_device_signs(
                strata::kernels::cpu::hadamard_signs(H_), H_);
            d_sh_ = strata::kernels::hadamard_device_signs(
                strata::kernels::cpu::hadamard_signs(int(ff)), int(ff));
        }

        // Validate slot count against free VRAM (leaving 512 MiB headroom).
        const size_t free_b = strata::core::device_free_bytes();
        constexpr uint64_t kReserve = 512ull << 20;
        const uint64_t budget = free_b > kReserve ? uint64_t(free_b) - kReserve : 0;

        uint64_t need = 0;
        std::vector<int64_t> sizes;
        sizes.reserve(fill.size());
        for (const auto& [l, e] : fill) {
            if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
                err = "AgentCache: fill pair out of range";
                close();
                return false;
            }
            const uint64_t b = lay.blob_bytes(l);
            const uint64_t b256 = lay.native ? ((b + 255) / 256 * 256) : lay.max_blob;
            if (budget > 0 && need + b256 > budget) {
                char msg[256];
                std::snprintf(msg, sizeof msg,
                              "AgentCache: requested %zu slots (%.2f GiB) exceeds free VRAM budget (%.2f GiB)",
                              fill.size(), double(need + b256) / 1073741824.0,
                              double(budget) / 1073741824.0);
                err = msg;
                close();
                return false;
            }
            need += b256;
            sizes.push_back(int64_t(b));
        }

        const bool ok = lay.native
            ? cache_.open_sized(sizes, n_layers, n_expert, err)
            : cache_.open(int64_t(fill.size()), n_layers, n_expert, int64_t(lay.max_blob), err);
        if (!ok) {
            close();
            return false;
        }
        res_.assign(size_t(n_layers) * size_t(n_expert), strata::core::kNotResident);
        return true;
#else
        err = "AgentCache::open_for_fill: SYCL ExpertCache port not linked (compile with STRATA_FRANKEN_SYCL_SPIKE)";
        (void)fill; (void)n_layers; (void)n_expert; (void)max_tokens; (void)max_entries; (void)H;
        return false;
#endif
    }

    // Loads one blob from the pack into a free slot (page-cache read; no
    // bytes cross any segment — §4.3 step 4).
    bool load(PackSource& src, int32_t layer, int32_t expert, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        const int32_t slot = cache_.admit(layer, expert);
        if (slot < 0) {
            err = "AgentCache::load: cache full during admit";
            return false;
        }
        const uint8_t* b = src.blob(layer, expert);
        if (b == nullptr) {
            err = "AgentCache::load: PackSource::blob returned nullptr";
            return false;
        }
        const int64_t bb = int64_t(src.blob_bytes(layer));
        if (!cache_.fill_slot_blocking(slot, b, err, bb)) return false;
        res_[size_t(layer) * size_t(n_expert_) + size_t(expert)] = slot;
        return true;
#else
        (void)src; (void)layer; (void)expert;
        err = "AgentCache::load: SYCL ExpertCache port not linked";
        return false;
#endif
    }

    bool verify_first(PackSource& src, int32_t layer, int32_t expert, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        const int32_t slot = slot_of(layer, expert);
        if (slot < 0) { err = "verify_first: not resident"; return false; }
        const uint8_t* b = src.blob(layer, expert);
        return cache_.verify_slot(slot, b, err, int64_t(src.blob_bytes(layer)));
#else
        (void)src; (void)layer; (void)expert; (void)err;
        return false;
#endif
    }

    bool has(int32_t layer, int32_t expert) const {
        return slot_of(layer, expert) >= 0;
    }

    int32_t slot_of(int32_t layer, int32_t expert) const {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return -1;
        return res_[size_t(layer) * size_t(n_expert_) + size_t(expert)];
#else
        (void)layer; (void)expert;
        return -1;
#endif
    }

#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    sycl::queue* queue() const { return q_; }
    const uint8_t* device_slot(int32_t slot) const { return cache_.device_slot(slot); }
    float* d_x() const { return d_x_; }
    uint8_t* d_q8() const { return d_q8_; }
    float* d_scales() const { return d_scales_; }
    void* d_scratch() const { return d_scratch_; }
    float* d_out() const { return d_out_; }
    unsigned long long* d_ptr() const { return d_ptr_; }
    int32_t* d_start() const { return d_start_; }
    int32_t* d_count() const { return d_count_; }
    int32_t* d_dst() const { return d_dst_; }
    int32_t* d_tok() const { return d_tok_; }
    const float* d_sx() const { return d_sx_; }
    const float* d_sh() const { return d_sh_; }
    double gib() const { return cache_.gib(); }
    int64_t resident() const { return cache_.resident(); }
#endif

private:
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    sycl::queue* q_ = nullptr;
    strata::core::ExpertCache cache_;
    std::vector<int32_t> res_;
    int32_t n_layers_ = 0, n_expert_ = 0;
    int32_t max_tokens_ = 0, max_entries_ = 0, H_ = 0;
    float* d_x_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    float* d_out_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    int32_t* d_start_ = nullptr;
    int32_t* d_count_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    const float* d_sx_ = nullptr;
    const float* d_sh_ = nullptr;
#endif
};

// ---------------------------------------------------------------------------
// Row computation. One entry = one (token, routed slot) pair; rows are fp32
// of width H and land at out[entry] in router order (§4.4 step 3).
// ---------------------------------------------------------------------------

// Agent side: grouped expert compute over the agent cache's resident blobs.
// Grouped/deduplicated by expert exactly like the primary's first_of pass
// (§4.4 step 3). Returns the number of resident entries computed via `computed`.
inline bool compute_rows_agent(AgentCache& cache, const Window& w,
                               int32_t k, int32_t H,
                               float* out, std::vector<uint8_t>& computed_mask,
                               double& elapsed_ms, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    const int64_t n = int64_t(w.n_tok) * int64_t(k);
    computed_mask.assign(size_t(n), 0);
    std::vector<unsigned long long> h_ptr;
    std::vector<int32_t> h_start, h_dst, h_tok, row_to_entry;
    h_ptr.reserve(size_t(n));
    h_start.reserve(size_t(n) + 1);
    h_dst.reserve(size_t(n));
    h_tok.reserve(size_t(n));
    row_to_entry.reserve(size_t(n));

    int32_t groups = 0, rows = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = w.ids[size_t(i)];
        const int32_t slot = cache.slot_of(w.layer, e);
        if (slot < 0) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j) {
            if (w.ids[size_t(j)] == e && cache.slot_of(w.layer, w.ids[size_t(j)]) >= 0) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        h_ptr.push_back(reinterpret_cast<unsigned long long>(cache.device_slot(slot)));
        h_start.push_back(rows);
        for (int64_t j = i; j < n; ++j) {
            if (w.ids[size_t(j)] == e) {
                h_dst.push_back(rows);
                h_tok.push_back(int32_t(j / k));
                row_to_entry.push_back(int32_t(j));
                computed_mask[size_t(j)] = 1;
                ++rows;
            }
        }
        ++groups;
    }
    if (groups == 0) {
        elapsed_ms = 0.0;
        return true;
    }
    h_start.push_back(rows);
    const int32_t counts[4] = {groups, rows, 0, 0};

    sycl::queue* q = cache.queue();
    std::vector<float> h_out(size_t(rows) * size_t(H));
    const auto t0 = std::chrono::steady_clock::now();
    try {
        q->memcpy(cache.d_x(), w.x.data(), size_t(w.n_tok) * size_t(H) * sizeof(float));
        q->memcpy(cache.d_ptr(), h_ptr.data(), size_t(groups) * sizeof(unsigned long long));
        q->memcpy(cache.d_start(), h_start.data(), size_t(groups + 1) * sizeof(int32_t));
        q->memcpy(cache.d_count(), counts, sizeof(counts));
        q->memcpy(cache.d_dst(), h_dst.data(), size_t(rows) * sizeof(int32_t));
        q->memcpy(cache.d_tok(), h_tok.data(), size_t(rows) * sizeof(int32_t));

        const auto& lay = strata::kernels::cpu::expert_layout();
        if (lay.native) {
            const auto& f = lay.fmt[size_t(w.layer)];
            if (f.had_gu) {
                strata::kernels::hadamard_rows(cache.d_x(), cache.d_x(), w.n_tok, H,
                                               f.had_block, cache.d_sx(), q);
            }
            strata::kernels::quantize_q8_1_rows(cache.d_x(), w.n_tok, H, cache.d_q8(), q);
            auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            if (f.had_d) {
                L.had_block = f.had_block;
                L.d_signs = cache.d_sh();
            }
            strata::kernels::native_expert_grouped(
                L, cache.d_ptr(), cache.d_start(), cache.d_count(),
                cache.d_dst(), cache.d_tok(), groups, rows,
                cache.d_q8(), cache.d_scratch(), cache.d_out(), q);
        } else {
            strata::kernels::quantize_q8_0_scaled(cache.d_x(), cache.d_q8(), cache.d_scales(),
                                                  int64_t(w.n_tok) * int64_t(H), q);
            strata::kernels::moe_grouped_s2(
                cache.d_ptr(), cache.d_start(), cache.d_count(),
                cache.d_dst(), cache.d_tok(), groups, rows,
                cache.d_q8(), cache.d_scales(), cache.d_scratch(), cache.d_out(), q);
        }
        q->memcpy(h_out.data(), cache.d_out(), size_t(rows) * size_t(H) * sizeof(float));
        q->wait_and_throw();
    } catch (const std::exception& ex) {
        err = std::string("compute_rows_agent: SYCL error: ") + ex.what();
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int32_t r = 0; r < rows; ++r) {
        std::memcpy(out + size_t(row_to_entry[size_t(r)]) * size_t(H),
                    h_out.data() + size_t(r) * size_t(H),
                    size_t(H) * sizeof(float));
    }
    return true;
#else
    (void)cache; (void)w; (void)k; (void)H; (void)out; (void)computed_mask; (void)elapsed_ms;
    err = "compute_rows_agent: SYCL grouped kernels not linked (compile with STRATA_FRANKEN_SYCL_SPIKE)";
    return false;
#endif
}

// Reference side: the CPU-row path (kind -1) — the engine's own reference
// implementation of an expert (§8.1, §8.2: "All comparisons are against the
// CPU-row path"). Uses ExpertPool::run_split_multi(_native) over PackSource blobs.
inline bool compute_rows_cpu(PackSource& src,
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
                             strata::kernels::cpu::ExpertPool& pool,
#endif
                             const Window& w, int32_t k, int32_t H,
                             const std::vector<uint8_t>& mask,
                             float* out, double& elapsed_ms, std::string& err) {
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
    using namespace strata::kernels::cpu;
    const auto t0 = std::chrono::steady_clock::now();
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    const int64_t n = int64_t(w.n_tok) * int64_t(k);

    std::vector<ActQ> act_multi(size_t(w.n_tok));
    std::vector<uint8_t> nact_multi(native ? size_t(w.n_tok) * kNativeActBytes : 0);
    for (int32_t t = 0; t < w.n_tok; ++t) {
        if (native && q2_native_kernels(lay.fmt[size_t(w.layer)].gu_type)) {
            act_quant_any(w.x.data() + size_t(t) * size_t(H), H, act_multi[size_t(t)]);
        } else if (native) {
            native_quant_act(lay.fmt[size_t(w.layer)], w.x.data() + size_t(t) * size_t(H),
                             nact_multi.data() + size_t(t) * kNativeActBytes);
        } else {
            act_quant_q8_1(w.x.data() + size_t(t) * size_t(H), H, act_multi[size_t(t)]);
        }
    }

    // Group entries by expert (up to MAXT tokens per ExpertJobMulti), holding copies of blobs
    // so ring-backed sources (GgufExpertSource) cannot recycle pointers mid-batch.
    const size_t bb = size_t(src.blob_bytes(w.layer));
    std::vector<std::vector<uint8_t>> blob_copies;
    std::vector<ExpertJobMulti> jobs;
    std::vector<int32_t> distinct_experts;

    for (int64_t i = 0; i < n; ++i) {
        if (!mask[size_t(i)]) continue;
        const int32_t e = w.ids[size_t(i)];
        bool seen = false;
        for (int32_t prev : distinct_experts) {
            if (prev == e) { seen = true; break; }
        }
        if (seen) continue;
        distinct_experts.push_back(e);
        const uint8_t* b = src.blob(w.layer, e);
        if (b == nullptr) {
            err = "compute_rows_cpu: PackSource::blob returned nullptr";
            return false;
        }
        blob_copies.emplace_back(b, b + bb);
        const uint8_t* copy_ptr = blob_copies.back().data();

        ExpertJobMulti cur{};
        cur.blob = copy_ptr;
        cur.nt = 0;
        for (int64_t j = i; j < n; ++j) {
            if (!mask[size_t(j)] || w.ids[size_t(j)] != e) continue;
            if (cur.nt == MAXT) {
                jobs.push_back(cur);
                cur = ExpertJobMulti{};
                cur.blob = copy_ptr;
                cur.nt = 0;
            }
            const int32_t tok = int32_t(j / k);
            cur.act[cur.nt] = &act_multi[size_t(tok)];
            cur.nact[cur.nt] = native ? nact_multi.data() + size_t(tok) * kNativeActBytes : nullptr;
            cur.out[cur.nt] = out + size_t(j) * size_t(H);
            ++cur.nt;
        }
        if (cur.nt > 0) jobs.push_back(cur);
    }

    if (!jobs.empty()) {
        for (size_t off = 0; off < jobs.size(); off += size_t(ExpertPool::kMaxSplitMulti)) {
            const int chunk = int(std::min<size_t>(size_t(ExpertPool::kMaxSplitMulti), jobs.size() - off));
            if (native) {
                pool.run_split_multi_native(lay.fmt[size_t(w.layer)], jobs.data() + off, chunk);
            } else {
                pool.run_split_multi(jobs.data() + off, chunk);
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
#else
    (void)src; (void)w; (void)k; (void)H; (void)mask; (void)out; (void)elapsed_ms;
    err = "compute_rows_cpu: CPU reference path not linked (compile with STRATA_FRANKEN_SYCL_SPIKE)";
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Row comparison statistics (DESIGN §8.2 step 1): for every sampled entry,
// relative L2 ||a-b|| / ||b||, max-abs error normalized by row RMS, and
// cosine similarity; reported as median / p99 / max per model. Real code.
// ---------------------------------------------------------------------------

struct RowStats {
    double rel_l2 = 0;
    double max_abs_over_rms = 0;
    double cosine = 1;
};

inline RowStats compare_rows(const float* a, const float* b, int32_t h) {
    double num = 0, den = 0, dot = 0, na = 0, nb = 0, max_abs = 0, sq = 0;
    for (int32_t i = 0; i < h; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
        max_abs = std::max(max_abs, std::fabs(d));
        sq += double(b[i]) * double(b[i]);
    }
    RowStats s;
    s.rel_l2 = den > 0 ? std::sqrt(num / den) : 0;
    const double rms = h > 0 ? std::sqrt(sq / h) : 0;
    s.max_abs_over_rms = rms > 0 ? max_abs / rms : 0;
    s.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1;
    return s;
}

struct Summary {
    double median = 0, p99 = 0, max = 0;
};

inline Summary summarize(std::vector<double> v) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    auto q = [&](double p) {
        const size_t i = std::min(v.size() - 1, size_t(p * double(v.size())));
        return v[i];
    };
    return {q(0.50), q(0.99), v.back()};
}

inline void print_summary(const char* name, const Summary& s) {
    std::printf("  %-18s median %.3e  p99 %.3e  max %.3e\n", name, s.median, s.p99, s.max);
}

}  // namespace spike

// ---------------------------------------------------------------------------
// main: open pack -> build fill -> fill cache -> replay recorded windows ->
// compare agent rows vs CPU rows -> print the §8.2 statistics.
// Mirrors §7's spike description step for step; stops at the first unlinked
// TODO with a nonzero exit, so a green run always means real numbers.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string pack_dir, profile_bin, recorded_path, native_gguf;
    int64_t slot_budget = 0;
    int32_t rand_windows = 0;
    int32_t rand_tok = 4;
    int32_t rand_layers = 48;
    int32_t rand_experts = 256;
    int32_t rand_k = 10;
    int32_t rand_embd = 2560;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack_dir = argv[i + 1];
        else if (a == "--native") native_gguf = argv[i + 1];
        else if (a == "--profile") profile_bin = argv[i + 1];
        else if (a == "--recorded") recorded_path = argv[i + 1];
        else if (a == "--slots") slot_budget = std::stoll(argv[i + 1]);
        else if (a == "--random-windows") rand_windows = std::stoi(argv[i + 1]);
        else if (a == "--random-tok") rand_tok = std::stoi(argv[i + 1]);
        else if (a == "--layers") rand_layers = std::stoi(argv[i + 1]);
        else if (a == "--experts") rand_experts = std::stoi(argv[i + 1]);
        else if (a == "--k") rand_k = std::stoi(argv[i + 1]);
        else if (a == "--embd") rand_embd = std::stoi(argv[i + 1]);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (pack_dir.empty() || (recorded_path.empty() && rand_windows <= 0) || slot_budget <= 0) {
        std::fprintf(stderr,
                     "usage: franken-agent-spike --pack <dir> --slots <n> "
                     "(--recorded <file.frkw> | --random-windows <N>) "
                     "[--native <shard1.gguf>] [--profile <profile.bin>] "
                     "[--random-tok <T>] [--layers <L>] [--experts <E>] [--k <K>] [--embd <H>]\n");
        return 2;
    }

    using namespace spike;
    try {
        RecordedSet rs;
        if (!recorded_path.empty()) {
            rs = load_recorded(recorded_path);
            std::printf("recorded: %zu windows, L=%d E=%d k=%d H=%d\n", rs.windows.size(),
                        rs.n_layers, rs.n_expert, rs.k, rs.n_embd);
        } else {
            rs.n_layers = rand_layers;
            rs.n_expert = rand_experts;
            rs.k = rand_k;
            rs.n_embd = rand_embd;
            std::mt19937 rng(42);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int32_t wi = 0; wi < rand_windows; ++wi) {
                Window w;
                w.layer = wi % rs.n_layers;
                w.n_tok = rand_tok;
                w.ids.resize(size_t(w.n_tok) * size_t(rs.k));
                w.x.resize(size_t(w.n_tok) * size_t(rs.n_embd));
                for (int32_t t = 0; t < w.n_tok; ++t) {
                    for (int32_t j = 0; j < rs.k; ++j) {
                        w.ids[size_t(t) * size_t(rs.k) + size_t(j)] =
                            int32_t((wi * 17 + t * 31 + j * 7) % rs.n_expert);
                    }
                }
                for (float& v : w.x) v = nd(rng);
                rs.windows.push_back(std::move(w));
            }
            std::printf("synthetic: %zu windows, L=%d E=%d k=%d H=%d n_tok=%d\n",
                        rs.windows.size(), rs.n_layers, rs.n_expert, rs.k, rs.n_embd, rand_tok);
        }

        std::string err;
        PackSource src;
        if (!src.open(pack_dir, native_gguf, rs.n_layers, rs.n_expert, err)) {
            std::fprintf(stderr, "spike stop: %s\n", err.c_str());
            return 1;
        }

        const RankedFill fill = profile_bin.empty()
                                    ? layer_early_fill(rs.n_layers, rs.n_expert, slot_budget)
                                    : load_profile_fill(profile_bin, rs.n_layers, rs.n_expert,
                                                        slot_budget, err);
        if (fill.empty()) {
            std::fprintf(stderr, "spike stop: %s\n",
                         err.empty() ? "empty fill list" : err.c_str());
            return 1;
        }

        int32_t max_tok = 1;
        for (const auto& w : rs.windows) max_tok = std::max(max_tok, w.n_tok);
        const int32_t max_entries = max_tok * rs.k;

        const auto tf0 = std::chrono::steady_clock::now();
        AgentCache cache;
        if (!cache.open_for_fill(fill, rs.n_layers, rs.n_expert, max_tok, max_entries,
                                 rs.n_embd, err)) {
            std::fprintf(stderr, "spike stop: %s\n", err.c_str());
            return 1;
        }
        for (const auto& [l, e] : fill) {
            if (!cache.load(src, l, e, err)) {
                std::fprintf(stderr, "spike stop: %s\n", err.c_str());
                return 1;
            }
        }
        if (!cache.verify_first(src, fill.front().first, fill.front().second, err)) {
            std::fprintf(stderr, "spike stop: %s\n", err.c_str());
            return 1;
        }
        const double fill_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
        std::printf("cache filled: %zu slots in %.2f s\n", fill.size(), fill_s);

#if defined(STRATA_FRANKEN_SYCL_SPIKE)
        strata::kernels::cpu::ExpertPool cpu_pool(0, /*pin=*/false, /*host_works=*/true);
#endif

        // Replay: sampling per §8.2 step 1 — every layer of the first 20
        // windows, then a 1% deterministic sample thereafter (plus all windows
        // when there are <= 100 windows).
        std::vector<double> rel_l2, max_abs, cosine;
        std::vector<double> sycl_ms_list, cpu_ms_list;
        std::vector<float> rows_agent, rows_cpu;
        std::vector<uint8_t> computed_mask;
        size_t total_entries = 0, resident_entries = 0;

        for (size_t wi = 0; wi < rs.windows.size(); ++wi) {
            const Window& w = rs.windows[wi];
            const size_t entries = size_t(w.n_tok) * size_t(rs.k);
            total_entries += entries;
            rows_agent.assign(entries * size_t(rs.n_embd), 0.f);
            rows_cpu.assign(entries * size_t(rs.n_embd), 0.f);

            double sycl_ms = 0.0, cpu_ms = 0.0;
            if (!compute_rows_agent(cache, w, rs.k, rs.n_embd, rows_agent.data(),
                                    computed_mask, sycl_ms, err)) {
                std::fprintf(stderr, "spike stop: %s\n", err.c_str());
                return 1;
            }
            size_t wi_res = 0;
            for (uint8_t m : computed_mask) wi_res += m ? 1 : 0;
            resident_entries += wi_res;
            if (wi_res == 0) continue;

            sycl_ms_list.push_back(sycl_ms);
            const bool sample = rs.windows.size() <= 100 || wi < 20 || (wi % 100 == 0);
            if (!sample) continue;

            if (!compute_rows_cpu(src,
#if defined(STRATA_FRANKEN_SYCL_SPIKE)
                                  cpu_pool,
#endif
                                  w, rs.k, rs.n_embd, computed_mask, rows_cpu.data(),
                                  cpu_ms, err)) {
                std::fprintf(stderr, "spike stop: %s\n", err.c_str());
                return 1;
            }
            cpu_ms_list.push_back(cpu_ms);

            for (size_t en = 0; en < entries; ++en) {
                if (!computed_mask[en]) continue;
                const RowStats s = compare_rows(&rows_agent[en * size_t(rs.n_embd)],
                                                &rows_cpu[en * size_t(rs.n_embd)], rs.n_embd);
                rel_l2.push_back(s.rel_l2);
                max_abs.push_back(s.max_abs_over_rms);
                cosine.push_back(s.cosine);
            }
        }

        std::printf("resident coverage: %zu / %zu entries (%.1f%%)\n",
                    resident_entries, total_entries,
                    total_entries > 0 ? 100.0 * double(resident_entries) / double(total_entries) : 0.0);
        std::printf("row divergence vs CPU reference (%zu sampled entries):\n", rel_l2.size());
        print_summary("rel_l2", summarize(rel_l2));
        print_summary("max_abs/rms", summarize(max_abs));
        print_summary("cosine", summarize(cosine));
        std::printf("window compute latency (ms):\n");
        print_summary("sycl_window_ms", summarize(sycl_ms_list));
        print_summary("cpu_window_ms", summarize(cpu_ms_list));
        return rel_l2.empty() ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "spike error: %s\n", ex.what());
        return 1;
    }
}
