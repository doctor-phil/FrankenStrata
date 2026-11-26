// franken_phase2_test.cpp — GPU-less unit tests for FrankenStrata Phase 2.
//
// Covers:
//   1. HwProfile JSON serialization / parsing round-trip (file I/O + edge fields)
//   2. recommend_tier_plan deterministic Primary selection, Agent ordering, slot budgets, and min-entry threshold
//   3. Live cross-process PROMOTE / DEMOTE / RESYNC over POSIX shm with real AgentLoop + AgentTier:
//      - DEMOTE un-routes immediately on primary
//      - Un-acked PROMOTE stays non-routable until PromoteAck lands
//      - Acked PROMOTE becomes routable and serves exact rows
//      - RESYNC reconciles full resident bitmap
//   4. Cascade swap planner (Primary dedup DEMOTE, Primary eviction cascade, hot CPU miss swaps)
//   5. Online latency cost model gating (tiny high-IPC rounds gated to CPU, large / parallel rounds dispatched,
//      A/B cost_model=off bypass, and online latency refinement)

#include "franken/agent/agent_loop.hpp"
#include "franken/control/tier_controller.hpp"
#include "franken/primary/agent_tier.hpp"
#include "franken/profiler/hw_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
namespace pr = franken::profiler;
namespace fc = franken::control;

namespace {

void test_hw_profile_json_roundtrip() {
    pr::HwProfile p{};
    p.schema_version = 1;
    p.timestamp_utc = "2026-11-25T23:45:00Z";
    p.pack_dir = "/work/Strata-data/packs/coder-iq1_m";
    p.n_layers = 48;
    p.n_expert = 256;
    p.k = 10;
    p.n_embd = 2560;
    p.avg_blob_bytes = 2048000;

    p.cpu.cpu_model = "AMD Ryzen 9 5900XT \"Vermeer\"";
    p.cpu.logical_cores = 32;
    p.cpu.physical_cores = 16;
    p.cpu.ram_total_bytes = 64ull << 30;
    p.cpu.ram_available_bytes = 52ull << 30;
    p.cpu.expert_latency = {{1, 0.30}, {2, 0.55}, {4, 1.05}, {8, 2.05}};
    pr::fit_linear_latency(p.cpu.expert_latency, p.cpu.ms_base, p.cpu.ms_per_expert);

    pr::GpuProfile g0{};
    g0.id = "hip:0";
    g0.vendor = "HIP";
    g0.device_name = "AMD Radeon RX 7900 XTX";
    g0.vram_total_bytes = 24ull << 30;
    g0.vram_usable_bytes = 18677760000ull;
    g0.recommended_slots = 9121;
    g0.pcie_h2d_gbps = 14.2;
    g0.pcie_d2h_gbps = 13.8;
    g0.a_ipc_ms = 0.05;
    g0.b_per_expert_ms = 0.04;
    g0.expert_latency = {{1, 0.09}, {2, 0.13}, {4, 0.21}, {8, 0.37}, {16, 0.69}};

    pr::GpuProfile g1{};
    g1.id = "/franken0";
    g1.vendor = "SYCL";
    g1.device_name = "Intel Arc Pro B70";
    g1.vram_total_bytes = 32ull << 30;
    g1.vram_usable_bytes = 30ull << 30;
    g1.recommended_slots = 12288;
    g1.pcie_h2d_gbps = 7.1;
    g1.pcie_d2h_gbps = 6.9;
    g1.a_ipc_ms = 0.60;
    g1.b_per_expert_ms = 0.08;
    g1.expert_latency = {{1, 0.68}, {2, 0.76}, {4, 0.92}, {8, 1.24}, {16, 1.88}};

    p.gpus.push_back(g0);
    p.gpus.push_back(g1);

    const std::string path = "/tmp/franken_p2_test_hw_" + std::to_string(::getpid()) + ".json";
    std::string err;
    assert(pr::save_hw_profile(path, p, err));

    pr::HwProfile loaded{};
    assert(pr::load_hw_profile(path, loaded, err));
    std::filesystem::remove(path);

    assert(loaded.schema_version == 1);
    assert(loaded.timestamp_utc == p.timestamp_utc);
    assert(loaded.pack_dir == p.pack_dir);
    assert(loaded.n_layers == 48 && loaded.n_expert == 256 && loaded.k == 10 && loaded.n_embd == 2560);
    assert(loaded.avg_blob_bytes == 2048000);
    assert(loaded.cpu.cpu_model == p.cpu.cpu_model);
    assert(loaded.cpu.physical_cores == 16 && loaded.cpu.ram_total_bytes == (64ull << 30));
    assert(loaded.cpu.expert_latency.size() == 4);
    assert(std::fabs(loaded.cpu.ms_per_expert - 0.25) < 1e-3);
    assert(loaded.gpus.size() == 2);
    assert(loaded.gpus[0].vendor == "HIP" && loaded.gpus[0].recommended_slots == 9121);
    assert(loaded.gpus[1].id == "/franken0" && loaded.gpus[1].vendor == "SYCL");
    assert(std::fabs(loaded.gpus[1].a_ipc_ms - 0.60) < 1e-5);
    assert(std::fabs(loaded.gpus[1].b_per_expert_ms - 0.08) < 1e-5);
    assert(loaded.gpus[1].expert_latency.size() == 5);
    std::printf("[PASS] test_hw_profile_json_roundtrip\n");
}

void test_recommend_tier_plan_ordering() {
    pr::HwProfile p{};
    p.n_layers = 48;
    p.n_expert = 256;  // 12,288 total experts
    p.avg_blob_bytes = 2000000;
    p.cpu.ms_base = 0.05;
    p.cpu.ms_per_expert = 0.30;  // T_cpu(n) = 0.05 + 0.30*n

    // Deliberately insert in unsorted order: CPU agent, SYCL B70, HIP 7900 XTX.
    pr::GpuProfile cpu_ag{};
    cpu_ag.id = "/franken_cpu";
    cpu_ag.vendor = "CPU";
    cpu_ag.device_name = "CPU Conformance";
    cpu_ag.recommended_slots = 1000;
    cpu_ag.a_ipc_ms = 0.20;
    cpu_ag.b_per_expert_ms = 0.30;

    pr::GpuProfile b70{};
    b70.id = "/franken0";
    b70.vendor = "SYCL";
    b70.device_name = "Arc Pro B70";
    b70.vram_usable_bytes = 30ull << 30;
    b70.recommended_slots = 12288;
    b70.a_ipc_ms = 0.60;
    b70.b_per_expert_ms = 0.08;  // beats CPU at n=3: 0.84 < 0.95 (at n=2: 0.76 > 0.65)

    pr::GpuProfile xtx{};
    xtx.id = "hip:0";
    xtx.vendor = "HIP";
    xtx.device_name = "RX 7900 XTX";
    xtx.vram_usable_bytes = 18ull << 30;
    xtx.recommended_slots = 9121;
    xtx.a_ipc_ms = 0.05;
    xtx.b_per_expert_ms = 0.04;

    p.gpus = {cpu_ag, b70, xtx};

    const pr::TierPlan plan = pr::recommend_tier_plan(p);
    assert(plan.primary_id == "hip:0");
    assert(plan.primary_vendor == "HIP");
    assert(plan.primary_slot_budget == 9121);
    assert(plan.agents.size() == 2);
    assert(plan.agents[0].id == "/franken0");
    assert(plan.agents[0].vendor == "SYCL");
    assert(plan.agents[0].slot_budget == 12288 - 9121);  // 3167 remaining
    assert(plan.agents[0].min_entries_threshold == 3);
    assert(plan.agents[1].id == "/franken_cpu");
    assert(plan.agents[1].slot_budget == 0);  // all 12,288 already covered
    std::printf("[PASS] test_recommend_tier_plan_ordering (primary=%s, tier1=%s slots=%u min_ent=%d)\n",
                plan.primary_id.c_str(), plan.agents[0].id.c_str(),
                plan.agents[0].slot_budget, plan.agents[0].min_entries_threshold);
}

// Lightweight in-memory CPU backend exercising the real AgentLoop PROMOTE/DEMOTE/RESYNC path.
class MockDynamicCpuBackend final : public fa::AgentBackend {
public:
    MockDynamicCpuBackend(uint32_t cap_slots, uint32_t promote_delay_us)
        : cap_slots_(cap_slots), promote_delay_us_(promote_delay_us) {}

    fw::AgentVendor vendor() const override { return fw::AgentVendor::Cpu; }

    bool init(const fw::Geometry& g, const uint64_t* /*blob_bytes*/,
              fw::Caps& out_caps, fw::PackIdentity& out_pack, std::string& /*err*/) override {
        g_ = g;
        out_pack = g.pack;
        out_caps = {g.max_tokens, g.max_entries, cap_slots_, uint32_t(fw::AgentVendor::Cpu)};
        slot_of_.assign(size_t(g.n_layers) * size_t(g.n_expert), -1);
        return true;
    }

    bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                   std::vector<std::pair<int32_t, int32_t>>& out_loaded,
                   std::string& /*err*/) override {
        slot_of_.assign(size_t(g_.n_layers) * size_t(g_.n_expert), -1);
        free_slots_.clear();
        used_slots_ = 0;
        out_loaded.clear();
        for (const auto& [l, e] : pairs) {
            if (used_slots_ >= cap_slots_) break;
            const size_t idx = size_t(l) * size_t(g_.n_expert) + size_t(e);
            if (slot_of_[idx] >= 0) continue;
            slot_of_[idx] = int32_t(used_slots_++);
            out_loaded.emplace_back(l, e);
        }
        return true;
    }

    bool promote(int32_t layer, int32_t expert, std::string& err) override {
        if (promote_delay_us_ > 0) ::usleep(promote_delay_us_);
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        if (slot_of_[idx] >= 0) return true;
        if (!free_slots_.empty()) {
            slot_of_[idx] = free_slots_.back();
            free_slots_.pop_back();
            return true;
        }
        if (used_slots_ < cap_slots_) {
            slot_of_[idx] = int32_t(used_slots_++);
            return true;
        }
        err = "no free slots";
        return false;
    }

    bool demote(int32_t layer, int32_t expert, std::string& /*err*/) override {
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        if (slot_of_[idx] >= 0) {
            free_slots_.push_back(slot_of_[idx]);
            slot_of_[idx] = -1;
        }
        return true;
    }

    bool compute_slot(int32_t layer, int32_t n_tok, int32_t k,
                      const int32_t* ids, const int32_t* kind, const float* x,
                      float* staging_out, std::string& err) override {
        const int32_t n = n_tok * k;
        for (int32_t i = 0; i < n; ++i) {
            if (kind[i] != 3) continue;
            const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(ids[i]);
            if (slot_of_[idx] < 0) {
                err = "expert not resident during compute_slot";
                return false;
            }
            const int32_t tok = i / k;
            const float mul = float(layer * 100 + ids[i] + 1);
            for (int32_t h = 0; h < g_.n_embd; ++h) {
                staging_out[size_t(i) * size_t(g_.n_embd) + size_t(h)] =
                    x[size_t(tok) * size_t(g_.n_embd) + size_t(h)] * mul;
            }
        }
        return true;
    }

    void shutdown() override {}

private:
    uint32_t cap_slots_ = 4;
    uint32_t promote_delay_us_ = 0;
    fw::Geometry g_{};
    std::vector<int32_t> slot_of_;
    std::vector<int32_t> free_slots_;
    uint32_t used_slots_ = 0;
};

void test_promote_demote_resync_over_shm() {
    const std::string shm_name = "/franken_p2_dyn_" + std::to_string(::getpid());
    fw::Geometry g{};
    g.n_layers = 4;
    g.n_expert = 16;
    g.k = 4;
    g.n_embd = 64;
    g.max_tokens = fw::kDefaultMaxTokens;
    g.max_entries = fw::kDefaultMaxEntries;
    g.pack.format = uint32_t(fw::PackFormat::ExpertsBin);
    g.pack.experts_bin_size = 4 * 16 * 1024;
    std::vector<uint64_t> blobs(size_t(g.n_layers), 1024);

    const pid_t pid = ::fork();
    assert(pid >= 0);
    if (pid == 0) {
        // 4-slot agent with 25 ms promote delay so we can assert un-acked PROMOTE stays non-routable.
        MockDynamicCpuBackend backend(/*cap_slots=*/4, /*promote_delay_us=*/25000);
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
    std::vector<uint8_t> claimed(size_t(g.n_layers * g.n_expert), 0);
    fp::AgentTier tier;
    std::string err;
    assert(tier.open(spec, g, blobs, ranked, claimed, 100.0, 5000, 5000, err));
    assert(tier.resident() == 4);
    assert(tier.allocated_slots() == 4);
    assert(tier.holds(0, 1) && !tier.holds(0, 4));

    // 1. DEMOTE must un-route IMMEDIATELY on primary (before any ack is polled).
    assert(tier.send_demote(0, 1));
    assert(!tier.holds(0, 1));
    assert(tier.resident() == 3);

    // 2. PROMOTE must stay NON-ROUTABLE until matching PromoteAck arrives.
    assert(tier.send_promote(0, 4));
    assert(!tier.holds(0, 4));
    assert(tier.has_pending_promote(0, 4));

    // Wait for agent to finish the 25 ms promote and poll acks.
    const uint64_t t0 = fi::now_ns();
    while (!tier.holds(0, 4)) {
        assert(fi::now_ns() - t0 < 2000000000ull);
        assert(tier.poll_control_acks());
        ::usleep(5000);
    }
    assert(tier.holds(0, 4));
    assert(!tier.has_pending_promote(0, 4));
    assert(tier.promotes_acked() == 1);
    assert(tier.resident() == 4);

    // 3. Verify real compute over shm uses newly promoted (0, 4).
    std::vector<float> x(size_t(g.n_embd), 2.0f);
    std::vector<float> out(size_t(g.k * g.n_embd), 0.0f);
    int32_t ids[4] = {0, 4, 2, 3};
    int32_t kind[4] = {3, 3, -1, -1};
    std::vector<int32_t> salvaged;
    assert(tier.launch(0, x.data(), ids, 1, 4, kind));
    assert(tier.finish(out.data(), salvaged));
    assert(salvaged.empty());
    // Entry 0 = expert 0 -> 2.0 * (0 + 1) = 2.0; Entry 1 = expert 4 -> 2.0 * (4 + 1) = 10.0.
    assert(std::fabs(out[0] - 2.0f) < 1e-6f);
    assert(std::fabs(out[size_t(g.n_embd)] - 10.0f) < 1e-6f);

    // 4. Verify RESYNC reconciles exact resident bitmap: {0, 2, 3, 4} in layer 0.
    assert(tier.resync(2000));
    assert(tier.resyncs_done() == 1);
    assert(tier.resident() == 4);
    assert(tier.holds(0, 0) && !tier.holds(0, 1) && tier.holds(0, 2) && tier.holds(0, 3) && tier.holds(0, 4));

    tier.close();
    int status = 0;
    assert(::waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    std::printf("[PASS] test_promote_demote_resync_over_shm\n");
}

void test_cascade_swap_planner() {
    const int32_t L = 1, E = 10;
    std::vector<float> heat(size_t(L * E), 0.0f);
    heat[0] = 15.0f;  // Primary resident
    heat[1] = 0.2f;   // Agent 0 coldest resident
    heat[2] = 11.0f;  // Agent 0 resident -> Primary just adopted this! (dedup)
    heat[3] = 0.5f;   // Agent 0 second-coldest resident
    heat[4] = 5.0f;   // Agent 0 warm resident
    heat[5] = 8.0f;   // Primary just evicted this! (cascade into Agent 0)
    heat[6] = 1.0f;   // Lukewarm CPU miss (< min_heat 2.0, ignored)
    heat[7] = 12.0f;  // Hot CPU miss -> should swap into Agent 0 evicting coldest

    std::vector<int32_t> primary_res(size_t(L * E), -1);
    primary_res[0] = 0;
    primary_res[5] = 1;  // being evicted

    std::vector<uint8_t> agent0_res(size_t(L * E), 0);
    agent0_res[1] = 1;
    agent0_res[2] = 1;
    agent0_res[3] = 1;
    agent0_res[4] = 1;

    std::vector<fc::AgentResidencySnapshot> snaps(1);
    snaps[0].alive = true;
    snaps[0].max_slots = 4;
    snaps[0].pending_promotes = 0;
    snaps[0].resident = &agent0_res;

    const fc::CascadePlan plan = fc::plan_cascade_swaps(
        L, E, heat, primary_res,
        /*primary_adopted=*/{{0, 2}},
        /*primary_evicted=*/{{0, 5}},
        snaps, /*max_swaps_per_agent=*/4, /*min_heat=*/2.0f, /*min_gain=*/1.5f);

    // Expect:
    //   Op 0: dedup demote (0, 2) -> frees 1 slot on Agent 0
    //   Op 1: cascade promote (0, 5) into freed slot (demote=-1)
    //   Op 2: hot CPU miss (0, 7) swaps in, demoting coldest victim (0, 1)
    assert(plan.ops.size() == 3);
    assert(plan.ops[0].is_dedup && plan.ops[0].demote_expert == 2 && plan.ops[0].promote_expert == -1);
    assert(plan.ops[1].is_cascade_from_primary && plan.ops[1].promote_expert == 5 && plan.ops[1].demote_expert == -1);
    assert(!plan.ops[2].is_cascade_from_primary && plan.ops[2].promote_expert == 7 && plan.ops[2].demote_expert == 1);
    std::printf("[PASS] test_cascade_swap_planner\n");
}

void test_online_cost_model_gating() {
    fc::TierController ctrl;
    // We don't need live shm pointers just to test the cost model math (1 dummy slot).
    std::vector<fp::AgentTier*> dummy(1, nullptr);
    ctrl.init(48, 256, dummy, /*cost_model_enabled=*/true, /*max_swaps=*/4);

    pr::HwProfile prof{};
    prof.cpu.ms_base = 0.02;
    prof.cpu.ms_per_expert = 0.25;  // T_cpu(n) = 0.02 + 0.25*n
    pr::GpuProfile g{};
    g.id = "/franken0";
    g.vendor = "SYCL";
    g.a_ipc_ms = 0.60;
    g.b_per_expert_ms = 0.05;  // T_agent(n) = 0.60 + 0.05*n
    prof.gpus.push_back(g);

    // Seed CPU + agent directly via profile and observe baseline
    ctrl.seed_from_profile(prof);
    // Because dummy[0] is nullptr, seed agent 0 via observe() with 12 samples along 0.60 + 0.05*n
    for (int k = 0; k < 3; ++k) {
        ctrl.observe_agent(0, 1, 0.65);
        ctrl.observe_agent(0, 2, 0.70);
        ctrl.observe_agent(0, 4, 0.80);
        ctrl.observe_agent(0, 8, 1.00);
    }

    // 1. Tiny batch (1 agent entry, 0 CPU misses): T_agent(1)=0.65 > T_cpu(1)=0.27 -> GATED OFF
    assert(!ctrl.should_dispatch_agent(0, 1, 0));
    assert(ctrl.gated_rounds() == 1 && ctrl.gated_entries() == 1);

    // 2. Large batch (6 agent entries, 0 CPU misses): T_agent(6)=0.90 < T_cpu(6)=1.52 -> DISPATCHED
    assert(ctrl.should_dispatch_agent(0, 6, 0));

    // 3. Parallel overlap (2 agent entries, 4 CPU misses):
    //    max(T_agent(2)=0.70, T_cpu(4)=1.02) = 1.02 < T_cpu(6)=1.52 -> DISPATCHED
    assert(ctrl.should_dispatch_agent(0, 2, 4));

    // 4. Cost model OFF bypasses gating even for tiny 1-entry batch
    ctrl.set_cost_model_enabled(false);
    assert(ctrl.should_dispatch_agent(0, 1, 0));
    ctrl.set_cost_model_enabled(true);

    // 5. Online refinement: if agent suddenly becomes ultra-fast (0.05 + 0.01*n),
    //    exponentially weighted fit adapts and admits n=1 again.
    for (int i = 0; i < 400; ++i) {
        ctrl.observe_agent(0, 1, 0.06);
        ctrl.observe_agent(0, 4, 0.09);
    }
    assert(ctrl.should_dispatch_agent(0, 1, 0));
    std::printf("[PASS] test_online_cost_model_gating\n");
}

}  // namespace

int main() {
    test_hw_profile_json_roundtrip();
    test_recommend_tier_plan_ordering();
    test_promote_demote_resync_over_shm();
    test_cascade_swap_planner();
    test_online_cost_model_gating();
    std::printf("ALL franken_phase2_test CHECKS PASSED\n");
    return 0;
}
