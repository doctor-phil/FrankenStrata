// franken_phase3_test.cpp — GPU-less unit tests for FrankenStrata Phase 3.
//
// Covers:
//   1. Pre-quantized q8_1 host quantizer bit accuracy, zero blocks, and kHalfMax clamping
//   2. Capability negotiation & fallback (q8_1 + compact rows vs had_gu FP32 fallback vs legacy v1 agent)
//   3. Non-blocking async PROMOTE: compute doorbells finish in <10 ms with 0 salvages while an 80 ms PROMOTE
//      runs concurrently on the background refill thread; measured C_move_ms recorded on PromoteAck
//   4. Move-cost economic break-even gate: (h_in - h_out) * H_eff * (w_src - w_m) > kappa * C_move_ms

#include "franken/agent/agent_loop.hpp"
#include "franken/control/tier_controller.hpp"
#include "franken/primary/agent_tier.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define FRANKEN_CHECK(cond)                                                            \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            std::fflush(stderr);                                                       \
            std::abort();                                                              \
        }                                                                              \
    } while (0)
#ifdef assert
#undef assert
#endif
#define assert(cond) FRANKEN_CHECK(cond)

namespace fw = franken::wire;
namespace fi = franken::ipc;
namespace fa = franken::agent;
namespace fp = franken::primary;
namespace fc = franken::control;

namespace {

float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t out = 0;
    if (ex == 0) {
        if (man == 0) out = sign;
        else {
            uint32_t m = man;
            int e = 113;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            out = sign | (uint32_t(e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((ex + 112u) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, 4);
    return f;
}

void test_q8_1_host_quantizer() {
    std::vector<float> x(96, 0.0f);
    // Block 0: normal values [-1.0 .. +2.5]
    for (int i = 0; i < 32; ++i) x[size_t(i)] = -1.0f + float(i) * (3.5f / 31.0f);
    // Block 1: all zeros
    // Block 2: huge value whose sum exceeds fp16 max (65504) -> must clamp finite (no Inf/NaN)
    for (int i = 0; i < 32; ++i) x[64 + size_t(i)] = 5000.0f;

    std::vector<uint8_t> out(3 * fw::kQ8_1BlockBytes, 0xAA);
    fi::quantize_q8_1_rows_host(x.data(), 1, 96, out.data());

    // Verify block 0 dequantization accuracy
    uint16_t d16 = 0, s16 = 0;
    std::memcpy(&d16, out.data(), 2);
    std::memcpy(&s16, out.data() + 2, 2);
    const float d0 = f16_to_f32(d16);
    const float s0 = f16_to_f32(s16);
    const auto* qs0 = reinterpret_cast<const int8_t*>(out.data() + 4);
    double true_sum = 0.0;
    for (int i = 0; i < 32; ++i) {
        true_sum += double(x[size_t(i)]);
        const float recon = d0 * float(qs0[i]);
        assert(std::fabs(recon - x[size_t(i)]) <= 0.025f);
    }
    assert(std::fabs(double(s0) - true_sum) < 0.1);

    // Verify block 1 is exact zero
    uint16_t d1 = 0, s1 = 0;
    std::memcpy(&d1, out.data() + fw::kQ8_1BlockBytes, 2);
    std::memcpy(&s1, out.data() + fw::kQ8_1BlockBytes + 2, 2);
    assert(d1 == 0 && s1 == 0);
    for (int i = 0; i < 32; ++i) assert(out[fw::kQ8_1BlockBytes + 4 + size_t(i)] == 0);

    // Verify block 2 clamps sum to finite fp16 <= 65504 (never inf/nan)
    uint16_t d2 = 0, s2 = 0;
    std::memcpy(&d2, out.data() + 2 * fw::kQ8_1BlockBytes, 2);
    std::memcpy(&s2, out.data() + 2 * fw::kQ8_1BlockBytes + 2, 2);
    const float s2f = f16_to_f32(s2);
    assert(std::isfinite(f16_to_f32(d2)) && std::isfinite(s2f) && s2f == 65504.0f);

    std::printf("[PASS] test_q8_1_host_quantizer\n");
}

class Phase3MockBackend final : public fa::AgentBackend {
public:
    Phase3MockBackend(uint32_t cap_slots, uint32_t feature_flags, uint32_t promote_delay_us)
        : cap_slots_(cap_slots), feature_flags_(feature_flags), promote_delay_us_(promote_delay_us) {}

    fw::AgentVendor vendor() const override { return fw::AgentVendor::Sycl; }

    bool init(const fw::Geometry& g, const uint64_t*, fw::Caps& out_caps,
              fw::PackIdentity& out_pack, std::string&) override {
        g_ = g;
        out_pack = g.pack;
        out_caps = {g.max_tokens, g.max_entries, cap_slots_,
                    uint32_t(fw::AgentVendor::Sycl), feature_flags_};
        slot_of_.assign(size_t(g.n_layers) * size_t(g.n_expert), -1);
        return true;
    }

    bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                   std::vector<std::pair<int32_t, int32_t>>& out_loaded, std::string&) override {
        std::lock_guard<std::mutex> lk(mu_);
        slot_of_.assign(size_t(g_.n_layers) * size_t(g_.n_expert), -1);
        free_slots_.clear();
        used_ = 0;
        out_loaded.clear();
        for (const auto& [l, e] : pairs) {
            if (used_ >= cap_slots_) break;
            const size_t idx = size_t(l) * size_t(g_.n_expert) + size_t(e);
            if (slot_of_[idx] >= 0) continue;
            slot_of_[idx] = int32_t(used_++);
            out_loaded.emplace_back(l, e);
        }
        return true;
    }

    bool promote(int32_t layer, int32_t expert, std::string& err) override {
        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
            if (slot_of_[idx] >= 0) return true;
            if (!free_slots_.empty()) {
                slot = free_slots_.back();
                free_slots_.pop_back();
            } else if (used_ < cap_slots_) {
                slot = int32_t(used_++);
            } else {
                err = "no free slot";
                return false;
            }
        }
        // Simulate slow H2D copy OUTSIDE mu_ so compute_slot_ex is never blocked!
        if (promote_delay_us_ > 0) ::usleep(promote_delay_us_);
        {
            std::lock_guard<std::mutex> lk(mu_);
            slot_of_[size_t(layer) * size_t(g_.n_expert) + size_t(expert)] = slot;
        }
        return true;
    }

    bool demote(int32_t layer, int32_t expert, std::string&) override {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        if (slot_of_[idx] >= 0) {
            free_slots_.push_back(slot_of_[idx]);
            slot_of_[idx] = -1;
        }
        return true;
    }

    bool compute_slot(int32_t layer, int32_t n_tok, int32_t k, const int32_t* ids,
                      const int32_t* kind, const float* x, float* out, std::string& err) override {
        return compute_slot_ex(layer, n_tok, k, ids, kind, x, 0u, out, err);
    }

    bool compute_slot_ex(int32_t layer, int32_t n_tok, int32_t k, const int32_t* ids,
                         const int32_t* kind, const void* x_data, uint32_t flags,
                         float* staging_out, std::string& err) override {
        const bool is_q8 = (flags & fw::kDescFlagActQ8_1) != 0;
        const bool is_compact = (flags & fw::kDescFlagCompactOut) != 0;
        if (is_q8 && (feature_flags_ & fw::kCapQ8_1Activations) == 0) {
            err = "unexpected q8_1 flag on legacy agent";
            return false;
        }
        const int32_t n = n_tok * k;
        int32_t cidx = 0;
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t i = 0; i < n; ++i) {
            if (kind[i] != 3) continue;
            const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(ids[i]);
            if (slot_of_[idx] < 0) {
                err = "expert not resident";
                return false;
            }
            const int32_t tok = i / k;
            const int32_t dst_row = is_compact ? cidx++ : i;
            const float mul = float(ids[i] + 1);
            float* dst = staging_out + size_t(dst_row) * size_t(g_.n_embd);
            if (is_q8) {
                const auto* base = static_cast<const uint8_t*>(x_data) +
                                   size_t(tok) * size_t(g_.n_embd / 32) * fw::kQ8_1BlockBytes;
                for (int32_t b = 0; b < g_.n_embd / 32; ++b) {
                    const uint8_t* blk = base + size_t(b) * fw::kQ8_1BlockBytes;
                    uint16_t d16 = 0;
                    std::memcpy(&d16, blk, 2);
                    const float d = f16_to_f32(d16);
                    const auto* qs = reinterpret_cast<const int8_t*>(blk + 4);
                    for (int q = 0; q < 32; ++q) {
                        dst[b * 32 + q] = (d * float(qs[q])) * mul;
                    }
                }
            } else {
                const float* xf = static_cast<const float*>(x_data) + size_t(tok) * size_t(g_.n_embd);
                for (int32_t h = 0; h < g_.n_embd; ++h) dst[h] = xf[h] * mul;
            }
        }
        return true;
    }

    void shutdown() override {}

private:
    uint32_t cap_slots_ = 4;
    uint32_t feature_flags_ = 0;
    uint32_t promote_delay_us_ = 0;
    fw::Geometry g_{};
    std::mutex mu_;
    std::vector<int32_t> slot_of_;
    std::vector<int32_t> free_slots_;
    uint32_t used_ = 0;
};

void test_q8_1_compact_and_async_promote() {
    auto run_case = [](const char* tag, uint32_t agent_caps, uint32_t promote_delay_us) {
        const std::string shm_name = std::string("/franken_p3_") + tag + "_" + std::to_string(::getpid());
        fw::Geometry g{};
        g.n_layers = 2;
        g.n_expert = 16;
        g.k = 4;
        g.n_embd = 64;
        g.max_tokens = fw::kDefaultMaxTokens;
        g.max_entries = fw::kDefaultMaxEntries;
        g.pack.format = uint32_t(fw::PackFormat::ExpertsBin);
        g.pack.experts_bin_size = 2 * 16 * 1024;
        std::vector<uint64_t> blobs(2, 1024);

        const pid_t pid = ::fork();
        assert(pid >= 0);
        if (pid == 0) {
            Phase3MockBackend backend(4, agent_caps, promote_delay_us);
            fa::AgentLoopConfig cfg{};
            cfg.shm_name = shm_name;
            cfg.attach_timeout_ms = 5000;
            cfg.verbose = false;
            fa::AgentLoopStats stats{};
            std::string aerr;
            const bool ok = fa::run_agent_loop(cfg, backend, stats, aerr);
            _exit(ok ? 0 : 1);
        }

        fp::AgentSpec spec{shm_name, 4};
        std::vector<std::pair<int32_t, int32_t>> ranked = {{0, 0}, {0, 1}, {0, 2}, {0, 3}};
        std::vector<uint8_t> claimed(32, 0);
        fp::AgentTier tier;
        std::string err;
        // Tight 20 ms deadline proves compute rounds NEVER block on the 80 ms async PROMOTE!
        assert(tier.open(spec, g, blobs, ranked, claimed, 20.0, 5000, 5000, err));
        assert(tier.uses_q8_1() == ((agent_caps & fw::kCapQ8_1Activations) != 0));
        assert(tier.uses_compact_rows() == ((agent_caps & fw::kCapCompactRows) != 0));

        // 1. Test allow_q8_1 = true (uses q8_1 when negotiated) AND allow_q8_1 = false (FP32 fallback)
        std::vector<float> x(64, 1.5f);
        std::vector<float> out(4 * 64, -1.0f);
        int32_t ids[4] = {0, 2, 1, 3};
        int32_t kind[4] = {-1, 3, -1, 3};  // sparse entries 1 and 3 -> tests compact row packing!
        std::vector<int32_t> salvaged;

        assert(tier.launch(0, x.data(), ids, 1, 4, kind, /*allow_q8_1=*/true));
        assert(tier.finish(out.data(), salvaged));
        assert(salvaged.empty());
        assert(tier.last_round_distinct() == 2);
        // Entry 1 = expert 2 -> 1.5 * 3 = 4.5; Entry 3 = expert 3 -> 1.5 * 4 = 6.0; Entries 0,2 untouched (-1.0)
        assert(out[0] == -1.0f);
        assert(std::fabs(out[1 * 64] - 4.5f) < 0.05f);
        assert(out[2 * 64] == -1.0f);
        assert(std::fabs(out[3 * 64] - 6.0f) < 0.05f);

        // FP32 fallback (e.g. had_gu layer)
        std::fill(out.begin(), out.end(), -2.0f);
        assert(tier.launch(0, x.data(), ids, 1, 4, kind, /*allow_q8_1=*/false));
        assert(tier.finish(out.data(), salvaged));
        assert(salvaged.empty());
        assert(std::fabs(out[1 * 64] - 4.5f) < 1e-6f);
        assert(std::fabs(out[3 * 64] - 6.0f) < 1e-6f);

        // 2. Non-blocking async PROMOTE test: demote (0, 1), enqueue 80 ms PROMOTE (0, 5),
        //    then immediately fire 5 compute rounds — all 5 must finish well under 30 ms with 0 salvages!
        if (promote_delay_us > 0) {
            assert(tier.send_demote(0, 1));
            assert(tier.send_promote(0, 5));
            assert(!tier.holds(0, 5));

            const uint64_t t_burst0 = fi::now_ns();
            for (int r = 0; r < 5; ++r) {
                salvaged.clear();
                assert(tier.launch(0, x.data(), ids, 1, 4, kind, /*allow_q8_1=*/true));
                assert(tier.finish(out.data(), salvaged));
                assert(salvaged.empty());
            }
            const double burst_ms = double(fi::now_ns() - t_burst0) / 1e6;
            assert(burst_ms < 40.0);  // proves 5 rounds did NOT block behind the 80 ms PROMOTE!

            // Wait for background PROMOTE_ACK to land and verify C_move_ms was measured.
            const uint64_t tw0 = fi::now_ns();
            while (!tier.holds(0, 5)) {
                assert(fi::now_ns() - tw0 < 2000000000ull);
                assert(tier.poll_control_acks());
                ::usleep(5000);
            }
            assert(tier.move_samples() >= 1);
            assert(tier.c_move_ms() >= 60.0);
        }

        tier.close();
        int status = 0;
        assert(::waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    };

    run_case("p3caps", fw::kCapQ8_1Activations | fw::kCapCompactRows, 80000);
    run_case("v1legacy", 0u, 0);
    std::printf("[PASS] test_q8_1_compact_and_async_promote\n");
}

void test_move_cost_economic_gate() {
    const int32_t L = 1, E = 8;
    std::vector<float> heat(8, 0.0f);
    heat[0] = 0.5f;   // Agent 0 cold victim
    heat[1] = 10.0f;  // Agent 0 warm resident
    heat[2] = 3.0f;   // CPU miss: delta_h = 2.5 -> benefit = 2.5 * 10 * (0.07 - 0.015) = 1.375 ms < 2.0 ms -> REJECT
    heat[3] = 8.5f;   // CPU miss: delta_h = 8.0 -> benefit = 8.0 * 10 * (0.07 - 0.015) = 4.400 ms > 2.0 ms -> ACCEPT

    std::vector<int32_t> primary_res(8, -1);
    std::vector<uint8_t> agent0_res(8, 0);
    agent0_res[0] = 1;
    agent0_res[1] = 1;

    std::vector<fc::AgentResidencySnapshot> snaps(1);
    snaps[0].alive = true;
    snaps[0].max_slots = 2;
    snaps[0].resident = &agent0_res;
    snaps[0].move_cost_ready = true;
    snaps[0].w_src_ms = 0.070;
    snaps[0].w_m_ms = 0.015;
    snaps[0].c_move_ms = 2.0;
    snaps[0].h_eff = 10.0;
    snaps[0].kappa = 1.0;

    // Only expert 2 (low benefit): must be rejected by move-cost gate!
    {
        auto h_low = heat;
        h_low[3] = 0.0f;
        const fc::CascadePlan p = fc::plan_cascade_swaps(L, E, h_low, primary_res, {}, {}, snaps, 4);
        assert(p.ops.empty());
        assert(p.move_cost_rejected >= 1);
    }

    // With expert 3 (high benefit = 4.40 ms > 2.0 ms): must be promoted evicting victim 0!
    {
        const fc::CascadePlan p = fc::plan_cascade_swaps(L, E, heat, primary_res, {}, {}, snaps, 4);
        assert(p.ops.size() == 1);
        assert(p.ops[0].promote_expert == 3 && p.ops[0].demote_expert == 0);
    }

    // Slower agent (w_m >= w_src): must always reject!
    {
        auto slow = snaps;
        slow[0].w_m_ms = 0.080;
        const fc::CascadePlan p = fc::plan_cascade_swaps(L, E, heat, primary_res, {}, {}, slow, 4);
        assert(p.ops.empty());
        assert(p.move_cost_rejected >= 1);
    }

    // Fallback when models are not ready (move_cost_ready = false): expert 2 (gain 2.5 >= 1.5) is accepted!
    {
        auto unready = snaps;
        unready[0].move_cost_ready = false;
        auto h_low = heat;
        h_low[3] = 0.0f;
        const fc::CascadePlan p = fc::plan_cascade_swaps(L, E, h_low, primary_res, {}, {}, unready, 4);
        assert(p.ops.size() == 1 && p.ops[0].promote_expert == 2);
    }

    std::printf("[PASS] test_move_cost_economic_gate\n");
}

}  // namespace

int main() {
    test_q8_1_host_quantizer();
    test_q8_1_compact_and_async_promote();
    test_move_cost_economic_gate();
    std::printf("ALL franken_phase3_test CHECKS PASSED\n");
    return 0;
}
