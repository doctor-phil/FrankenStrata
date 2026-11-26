// franken_agent_gpu.cpp — AMD HIP / NVIDIA CUDA foreign-vendor expert tier agent.
//
// Compiled as `franken-agent-hip` under STRATA_ENABLE_HIP (via strata_hip_runtime's
// CUDA-to-HIP header shim) and as `franken-agent-cuda` under STRATA_ENABLE_CUDA.
// Mirrors franken-agent-sycl with:
//   - Native IQ packs (including Hadamard rotation) & canonical s2 packs
//   - Pre-quantized q8_1 wire activations (kCapQ8_1Activations)
//   - Compact router-order row return (kCapCompactRows)
//   - Non-blocking async PROMOTE on a dedicated cudaStreamNonBlocking refill stream
//   - Pinned host staging + coherency self-test

#include "franken/agent/agent_loop.hpp"
#include "franken/wire/pack_identity.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct GpuMeta {
    unsigned long long ptr[fw::kDefaultMaxEntries];
    int32_t start[fw::kDefaultMaxEntries + 1];
    int32_t dst[fw::kDefaultMaxEntries];
    int32_t tok[fw::kDefaultMaxEntries];
    int32_t count[4];
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("franken-agent-gpu: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool verify_pinned_coherency(cudaStream_t s) {
    constexpr size_t kN = 16384;
    float* h = nullptr;
    float* d = nullptr;
    bool ok = false;
    if (cudaHostAlloc(reinterpret_cast<void**>(&h), kN * sizeof(float), cudaHostAllocDefault) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d), kN * sizeof(float)) == cudaSuccess) {
        std::vector<float> ref(kN);
        for (size_t i = 0; i < kN; ++i) {
            ref[i] = 1.0f + float((i * 2654435761u) % 4093u);
            h[i] = ref[i];
        }
        if (cudaMemcpyAsync(d, h, kN * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess &&
            cudaStreamSynchronize(s) == cudaSuccess) {
            std::memset(h, 0, kN * sizeof(float));
            if (cudaMemcpyAsync(h, d, kN * sizeof(float), cudaMemcpyDeviceToHost, s) == cudaSuccess &&
                cudaStreamSynchronize(s) == cudaSuccess) {
                ok = (std::memcmp(h, ref.data(), kN * sizeof(float)) == 0);
            }
        }
    }
    if (d != nullptr) (void)cudaFree(d);
    if (h != nullptr) (void)cudaFreeHost(h);
    (void)cudaGetLastError();
    return ok;
}

class GpuAgentBackend final : public fa::AgentBackend {
public:
    GpuAgentBackend(int device, std::string pack_dir, std::string native_gguf,
                    uint32_t max_slots, uint32_t reserve_mib)
        : device_(device),
          pack_dir_(std::move(pack_dir)),
          native_gguf_(std::move(native_gguf)),
          max_slots_(max_slots),
          reserve_mib_(reserve_mib) {}

    ~GpuAgentBackend() override { shutdown(); }

    fw::AgentVendor vendor() const override {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
        return fw::AgentVendor::Hip;
#else
        return fw::AgentVendor::Cuda;
#endif
    }

    bool init(const fw::Geometry& g,
              const uint64_t* blob_bytes,
              fw::Caps& out_caps,
              fw::PackIdentity& out_pack,
              std::string& err) override {
        g_ = g;
        int dev_count = 0;
        if (!ck(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount", err)) return false;
        if (device_ < 0 || device_ >= dev_count) {
            err = "franken-agent-gpu: invalid --device " + std::to_string(device_) +
                  " (visible devices: " + std::to_string(dev_count) + ")";
            return false;
        }
        if (!ck(cudaSetDevice(device_), "cudaSetDevice", err)) return false;

        cudaDeviceProp prop{};
        (void)cudaGetDeviceProperties(&prop, device_);

        if (!cpu::expert_layout_load(pack_dir_, g.n_layers, g.n_expert, err)) return false;
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        for (int32_t l = 0; l < g.n_layers; ++l) {
            if (lay.blob_bytes(l) != blob_bytes[l]) {
                err = "GpuAgentBackend: blob_bytes mismatch at layer " + std::to_string(l);
                return false;
            }
        }
        if (!fi::compute_pack_identity(pack_dir_, native_gguf_, lay, out_pack, err)) return false;

        file_src_ = std::make_unique<strata::core::FileExpertSource>();
        if (!native_gguf_.empty()) file_src_->set_gguf(native_gguf_);
        if (!file_src_->open(pack_dir_, g.n_layers, g.n_expert, err)) {
            file_src_.reset();
            return false;
        }

        if (!ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream", err) ||
            !ck(cudaStreamCreateWithFlags(&refill_stream_, cudaStreamNonBlocking), "create refill stream", err)) {
            shutdown();
            return false;
        }

        pinned_ok_ = verify_pinned_coherency(stream_);
        std::fprintf(stderr, "franken-agent-gpu: pinned host staging coherency self-test: %s\n",
                     pinned_ok_ ? "PASS" : "FAIL (pageable fallback)");

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

        bool alloc_ok =
            ck(cudaMalloc(reinterpret_cast<void**>(&d_x_), size_t(cap_tok_) * size_t(g.n_embd) * sizeof(float)), "d_x", err) &&
            ck(cudaMalloc(reinterpret_cast<void**>(&d_out_), size_t(cap_ent_) * size_t(g.n_embd) * sizeof(float)), "d_out", err) &&
            ck(cudaMalloc(reinterpret_cast<void**>(&d_meta_), sizeof(GpuMeta)), "d_meta", err) &&
            ck(cudaMalloc(reinterpret_cast<void**>(&d_q8_), q8_bytes), "d_q8", err) &&
            ck(cudaMalloc(reinterpret_cast<void**>(&d_scales_), size_t(cap_tok_) * size_t(g.n_embd / 32) * sizeof(float)), "d_scales", err) &&
            ck(cudaMalloc(&d_scratch_, scratch_bytes), "d_scratch", err);
        if (alloc_ok && pinned_ok_) {
            alloc_ok =
                ck(cudaHostAlloc(reinterpret_cast<void**>(&h_x_), size_t(cap_tok_) * size_t(g.n_embd) * sizeof(float), cudaHostAllocDefault), "h_x", err) &&
                ck(cudaHostAlloc(reinterpret_cast<void**>(&h_q8_), q8_bytes, cudaHostAllocDefault), "h_q8", err) &&
                ck(cudaHostAlloc(reinterpret_cast<void**>(&h_out_), size_t(cap_ent_) * size_t(g.n_embd) * sizeof(float), cudaHostAllocDefault), "h_out", err) &&
                ck(cudaHostAlloc(reinterpret_cast<void**>(&h_meta_), sizeof(GpuMeta), cudaHostAllocDefault), "h_meta", err) &&
                ck(cudaHostAlloc(reinterpret_cast<void**>(&refill_bounce_pinned_), size_t(lay.max_blob), cudaHostAllocDefault), "refill_bounce", err);
        } else if (alloc_ok) {
            h_x_vec_.resize(size_t(cap_tok_) * size_t(g.n_embd));
            h_q8_vec_.resize(q8_bytes);
            h_out_vec_.resize(size_t(cap_ent_) * size_t(g.n_embd));
            h_meta_vec_.resize(1);
            refill_bounce_vec_.resize(size_t(lay.max_blob));
            h_x_ = h_x_vec_.data();
            h_q8_ = h_q8_vec_.data();
            h_out_ = h_out_vec_.data();
            h_meta_ = h_meta_vec_.data();
        }
        if (!alloc_ok) {
            shutdown();
            return false;
        }

        if (cpu::hadamard_spec().any()) {
            d_sx_ = strata::kernels::hadamard_device_signs(cpu::hadamard_signs(g.n_embd), g.n_embd);
            d_sh_ = strata::kernels::hadamard_device_signs(cpu::hadamard_signs(int(ff)), int(ff));
        }

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
                     "franken-agent-gpu: device %d '%s', free %.2f GiB, reserve %u MiB, "
                     "budget %.2f GiB -> advertising %u slots\n",
                     device_, prop.name, double(free_b) / 1073741824.0, reserve_mib_,
                     double(vram_budget_) / 1073741824.0, est_slots);

        out_caps.max_tokens_per_window = cap_tok_;
        out_caps.max_entries_per_window = cap_ent_;
        out_caps.slot_count = est_slots;
        out_caps.vendor = uint32_t(vendor());
        out_caps.feature_flags = fw::kCapQ8_1Activations | fw::kCapCompactRows;
        return true;
    }

    bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                   std::vector<std::pair<int32_t, int32_t>>& out_loaded,
                   std::string& err) override {
        if (!ck(cudaSetDevice(device_), "cudaSetDevice", err)) return false;
        const cpu::ExpertLayout& lay = cpu::expert_layout();

        std::lock_guard<std::mutex> lk(res_mu_);
        cache_.close();
        res_.assign(size_t(g_.n_layers) * size_t(g_.n_expert), strata::core::kNotResident);
        free_slots_.clear();
        out_loaded.clear();

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
        if (pick.empty()) return true;

        const bool use_uniform = !lay.native || (uint64_t(pick.size()) * max_b256 <= vram_budget_);
        const bool opened = use_uniform
            ? cache_.open(int64_t(pick.size()), g_.n_layers, g_.n_expert, int64_t(max_b256), err)
            : cache_.open_sized(sizes, g_.n_layers, g_.n_expert, err);
        if (!opened) return false;
        if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) return false;

        for (const auto& [l, e] : pick) {
            const int32_t slot = cache_.admit(l, e);
            const uint8_t* b = file_src_->blob(l, e);
            if (slot < 0 || b == nullptr) {
                err = "GpuAgentBackend::open_fill: blob read or admit failed";
                return false;
            }
            if (!cache_.fill_slot_blocking(slot, b, err, int64_t(lay.blob_bytes(l)))) return false;
            res_[size_t(l) * size_t(g_.n_expert) + size_t(e)] = slot;
            out_loaded.emplace_back(l, e);
        }

        const auto [l0, e0] = out_loaded.front();
        const int32_t s0 = res_[size_t(l0) * size_t(g_.n_expert) + size_t(e0)];
        if (!cache_.verify_slot(s0, file_src_->blob(l0, e0), err, int64_t(lay.blob_bytes(l0)))) {
            return false;
        }
        std::fprintf(stderr, "franken-agent-gpu: verified slot 0, resident %zu experts (%.2f GiB VRAM, %s slots)\n",
                     out_loaded.size(), cache_.gib(), use_uniform ? "uniform" : "sized");
        return true;
    }

    bool promote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "GpuAgentBackend::promote: (layer, expert) out of range";
            return false;
        }
        if (!ck(cudaSetDevice(device_), "cudaSetDevice(promote)", err)) return false;

        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        const uint64_t need = lay.blob_bytes(layer);

        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lk(res_mu_);
            if (res_[idx] >= 0) return true;
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
                err = "GpuAgentBackend::promote: no freed slot with room >= " + std::to_string(need);
                return false;
            }
            slot = free_slots_[best_pos];
            free_slots_.erase(free_slots_.begin() + ptrdiff_t(best_pos));
        }

        uint8_t* bounce = refill_bounce_pinned_ != nullptr ? refill_bounce_pinned_ : refill_bounce_vec_.data();
        if (!file_src_->copy_blob(layer, expert, bounce)) {
            std::lock_guard<std::mutex> lk(res_mu_);
            free_slots_.push_back(slot);
            err = "GpuAgentBackend::promote: copy_blob failed";
            return false;
        }
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), bounce, size_t(need),
                                cudaMemcpyHostToDevice, refill_stream_), "promote h2d", err) ||
            !ck(cudaStreamSynchronize(refill_stream_), "promote sync", err)) {
            std::lock_guard<std::mutex> lk(res_mu_);
            free_slots_.push_back(slot);
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(res_mu_);
            res_[idx] = slot;
        }
        return true;
    }

    bool demote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "GpuAgentBackend::demote: (layer, expert) out of range";
            return false;
        }
        std::lock_guard<std::mutex> lk(res_mu_);
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        const int32_t slot = res_[idx];
        if (slot < 0) return true;
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
            err = "GpuAgentBackend: window exceeds agent caps";
            return false;
        }

        const bool act_q8_1 = (desc_flags & fw::kDescFlagActQ8_1) != 0;
        const bool compact_out = (desc_flags & fw::kDescFlagCompactOut) != 0;

        int32_t compact_of[fw::kDefaultMaxEntries];
        int32_t n_compact = 0;
        for (int64_t i = 0; i < n; ++i) {
            compact_of[i] = (kind[i] == 3) ? n_compact++ : -1;
        }

        GpuMeta& m = *h_meta_;
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
                    err = "GpuAgentBackend: routed kind==3 expert (" + std::to_string(layer) +
                          "," + std::to_string(e) + ") not resident";
                    return false;
                }
                m.ptr[groups] = reinterpret_cast<unsigned long long>(cache_.device_slot(slot));
                m.start[groups] = rows;
                for (int64_t j = i; j < n; ++j) {
                    if (kind[j] == 3 && ids[j] == e) {
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

        if (!ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(GpuMeta), cudaMemcpyHostToDevice, stream_), "meta h2d", err)) {
            return false;
        }

        if (lay.native) {
            const cpu::NativeFmt& f = lay.fmt[size_t(layer)];
            if (act_q8_1 && !f.had_gu) {
                if (!ck(cudaMemcpyAsync(d_q8_, h_q8_, q8_1_bytes, cudaMemcpyHostToDevice, stream_), "q8_1 h2d", err)) {
                    return false;
                }
            } else {
                if (!ck(cudaMemcpyAsync(d_x_, h_x_, fp32_bytes, cudaMemcpyHostToDevice, stream_), "x h2d", err)) {
                    return false;
                }
                if (f.had_gu) {
                    strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, g_.n_embd, f.had_block, d_sx_, stream_);
                }
                strata::kernels::quantize_q8_1_rows(d_x_, n_tok, g_.n_embd, d_q8_, stream_);
            }
            auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            if (f.had_d) {
                L.had_block = f.had_block;
                L.d_signs = d_sh_;
            }
            strata::kernels::native_expert_grouped(
                L, d_meta_->ptr, d_meta_->start, d_meta_->count,
                d_meta_->dst, d_meta_->tok, groups, rows,
                d_q8_, d_scratch_, d_out_, stream_);
        } else {
            if (!ck(cudaMemcpyAsync(d_x_, h_x_, fp32_bytes, cudaMemcpyHostToDevice, stream_), "x h2d", err)) {
                return false;
            }
            strata::kernels::quantize_q8_0_scaled(d_x_, d_q8_, d_scales_,
                                                  int64_t(n_tok) * int64_t(g_.n_embd), stream_);
            strata::kernels::moe_grouped_s2(
                d_meta_->ptr, d_meta_->start, d_meta_->count,
                d_meta_->dst, d_meta_->tok, groups, rows,
                d_q8_, d_scales_, d_scratch_, d_out_, stream_);
        }

        if (!ck(cudaMemcpyAsync(h_out_, d_out_, size_t(rows) * size_t(g_.n_embd) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_), "out d2h", err) ||
            !ck(cudaStreamSynchronize(stream_), "stream sync", err)) {
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
        if (device_ >= 0) {
            (void)cudaSetDevice(device_);
            if (stream_) (void)cudaStreamSynchronize(stream_);
            if (refill_stream_) (void)cudaStreamSynchronize(refill_stream_);
            cache_.close();
            if (d_x_) (void)cudaFree(d_x_);
            if (d_out_) (void)cudaFree(d_out_);
            if (d_meta_) (void)cudaFree(d_meta_);
            if (d_q8_) (void)cudaFree(d_q8_);
            if (d_scales_) (void)cudaFree(d_scales_);
            if (d_scratch_) (void)cudaFree(d_scratch_);
            if (pinned_ok_) {
                if (h_x_) (void)cudaFreeHost(h_x_);
                if (h_q8_) (void)cudaFreeHost(h_q8_);
                if (h_out_) (void)cudaFreeHost(h_out_);
                if (h_meta_) (void)cudaFreeHost(h_meta_);
                if (refill_bounce_pinned_) (void)cudaFreeHost(refill_bounce_pinned_);
            }
            if (stream_) (void)cudaStreamDestroy(stream_);
            if (refill_stream_) (void)cudaStreamDestroy(refill_stream_);
        }
        d_x_ = d_out_ = nullptr;
        d_meta_ = nullptr;
        d_q8_ = nullptr;
        d_scales_ = nullptr;
        d_scratch_ = nullptr;
        h_x_ = h_out_ = nullptr;
        h_q8_ = refill_bounce_pinned_ = nullptr;
        h_meta_ = nullptr;
        stream_ = refill_stream_ = nullptr;
        d_sx_ = d_sh_ = nullptr;
        h_x_vec_.clear();
        h_q8_vec_.clear();
        h_out_vec_.clear();
        h_meta_vec_.clear();
        refill_bounce_vec_.clear();
        res_.clear();
        free_slots_.clear();
        file_src_.reset();
    }

private:
    int device_ = 0;
    std::string pack_dir_;
    std::string native_gguf_;
    uint32_t max_slots_ = 0;
    uint32_t reserve_mib_ = 1024;
    fw::Geometry g_{};
    uint32_t cap_tok_ = 0, cap_ent_ = 0;
    uint64_t vram_budget_ = 0;
    bool pinned_ok_ = false;

    std::unique_ptr<strata::core::FileExpertSource> file_src_;
    strata::core::ExpertCache cache_;
    std::mutex res_mu_;
    std::vector<int32_t> res_;
    std::vector<int32_t> free_slots_;

    cudaStream_t stream_ = nullptr;
    cudaStream_t refill_stream_ = nullptr;
    float* h_x_ = nullptr;
    uint8_t* h_q8_ = nullptr;
    float* h_out_ = nullptr;
    GpuMeta* h_meta_ = nullptr;
    uint8_t* refill_bounce_pinned_ = nullptr;
    std::vector<float> h_x_vec_;
    std::vector<uint8_t> h_q8_vec_;
    std::vector<float> h_out_vec_;
    std::vector<GpuMeta> h_meta_vec_;
    std::vector<uint8_t> refill_bounce_vec_;
    float* d_x_ = nullptr;
    float* d_out_ = nullptr;
    GpuMeta* d_meta_ = nullptr;
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
    int device = 0;
    uint32_t max_slots = 0;
    uint32_t reserve_mib = 1024;
    uint32_t wait_ms = 30000;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--shm") shm_name = argv[i + 1];
        else if (a == "--pack") pack_dir = argv[i + 1];
        else if (a == "--native") native_gguf = argv[i + 1];
        else if (a == "--device") device = std::stoi(argv[i + 1]);
        else if (a == "--slots") max_slots = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--reserve-mib") reserve_mib = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--wait-ms") wait_ms = uint32_t(std::stoul(argv[i + 1]));
        else {
            std::fprintf(stderr, "franken-agent-gpu: unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (shm_name.empty() || pack_dir.empty()) {
        std::fprintf(stderr,
                     "usage: franken-agent-gpu --shm <name> --pack <dir> "
                     "[--native <shard1.gguf>] [--device <N>] [--slots <N>] "
                     "[--reserve-mib <M>] [--wait-ms <ms>]\n");
        return 2;
    }

    fa::install_agent_signal_handlers();
    GpuAgentBackend backend(device, pack_dir, native_gguf, max_slots, reserve_mib);
    fa::AgentLoopConfig cfg{};
    cfg.shm_name = shm_name;
    cfg.attach_timeout_ms = wait_ms;
    fa::AgentLoopStats stats{};
    std::string err;
    if (!fa::run_agent_loop(cfg, backend, stats, err)) {
        std::fprintf(stderr, "franken-agent-gpu: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
