// tier_controller.hpp — FrankenStrata Phase 2 Dynamic Multi-Tier Control Module.
//
// Coordinates expert movement and per-layer dispatch decisions across:
//   Tier 0 (Primary GPU VRAM) <-> Tier 1..N (Foreign-Vendor Agents) <-> CPU/RAM
//
// Components:
//   1. Online latency cost model: exponentially weighted least squares fitting
//      T_agent_i(n) = a_i + b_i * n and T_cpu(n) = a_cpu + b_cpu * n (optionally
//      seeded from franken-hw-profile.json), gating tiny high-IPC-overhead rounds
//      back to CPU when max(T_primary, T_agent, T_cpu) would increase.
//   2. Cascade swap planner (`plan_cascade_swaps`): pure deterministic planner
//      handling Primary-adopted deduplication (DEMOTE on Agent), Primary-evicted
//      cascade into Agent VRAM, and hot CPU misses -> Agent VRAM swaps.

#pragma once

#include "franken/primary/agent_tier.hpp"
#include "franken/profiler/hw_profile.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace franken {
namespace control {

// ---------------------------------------------------------------------------
// Exponentially weighted linear latency model: T(n) = a + b * n (ms)
// ---------------------------------------------------------------------------

struct LinearLatencyModel {
    bool seeded = false;
    double a = 0.0;       // fixed overhead (ms), e.g. IPC doorbell
    double b = 0.0;       // marginal latency per expert (ms)
    double A[2][2] = {};  // normal equations for [1, n]
    double B[2] = {};
    int64_t samples = 0;

    void seed(double prior_a_ms, double prior_b_ms, double weight = 4.0);
    void observe(int32_t n_experts, double elapsed_ms);
    void solve();

    bool ready() const { return seeded || samples >= 6; }
    double predict(int32_t n_experts) const {
        if (n_experts <= 0) return 0.0;
        return std::max(0.0, a + b * double(n_experts));
    }
};

// ---------------------------------------------------------------------------
// Pure cascade swap planner (GPU-less & unit-testable)
// ---------------------------------------------------------------------------

struct AgentResidencySnapshot {
    bool alive = false;
    uint32_t max_slots = 0;
    size_t pending_promotes = 0;
    const std::vector<uint8_t>* resident = nullptr;  // size n_layers * n_expert
    // Phase 3 move-cost economic break-even parameters (enabled when enabled == true
    // and c_move_ms > 0): promote only if (h_in - h_out) * h_eff * (w_src_ms - w_m_ms)
    // > kappa * c_move_ms. When disabled (models not ready yet), falls back to min_gain.
    bool move_cost_ready = false;
    double w_src_ms = 0.0;    // marginal per-expert latency on source tier (CPU)
    double w_m_ms = 0.0;      // marginal per-expert latency on this agent tier
    double c_move_ms = 0.0;   // measured/estimated PROMOTE transfer cost (ms)
    double h_eff = 10.0;      // effective horizon multiplier (~1/(1-decay))
    double kappa = 1.0;       // break-even safety margin
};

struct CascadeOp {
    size_t agent_idx = 0;
    int32_t demote_layer = -1;
    int32_t demote_expert = -1;
    int32_t promote_layer = -1;
    int32_t promote_expert = -1;
    float gain = 0.0f;
    bool is_dedup = false;
    bool is_cascade_from_primary = false;
};

struct CascadePlan {
    std::vector<CascadeOp> ops;
    uint64_t move_cost_rejected = 0;
};

// Deterministically plans deduplication demotions, Primary-evicted cascades,
// and hot CPU-miss promotions across Tier 1..N agents.
CascadePlan plan_cascade_swaps(int32_t n_layers,
                               int32_t n_expert,
                               const std::vector<float>& heat,
                               const std::vector<int32_t>& primary_res,
                               const std::vector<std::pair<int32_t, int32_t>>& primary_adopted,
                               const std::vector<std::pair<int32_t, int32_t>>& primary_evicted,
                               const std::vector<AgentResidencySnapshot>& agents,
                               int32_t max_swaps_per_agent,
                               float min_heat = 2.0f,
                               float min_gain = 1.5f);

// ---------------------------------------------------------------------------
// Runtime TierController owned by the primary engine
// ---------------------------------------------------------------------------

class TierController {
public:
    TierController() = default;

    void init(int32_t n_layers,
              int32_t n_expert,
              const std::vector<primary::AgentTier*>& agents,
              bool cost_model_enabled = true,
              int32_t max_swaps_per_agent = 8);

    // Seeds latency priors from a saved franken-hw-profile.json.
    void seed_from_profile(const profiler::HwProfile& prof);

    void set_cost_model_enabled(bool on) { cost_model_enabled_ = on; }
    bool cost_model_enabled() const { return cost_model_enabled_; }

    // Online latency gating consulted by expert_pool_dispatch_multi before launching
    // agent `agent_idx` for `n_agent_distinct` distinct experts alongside `n_cpu_distinct`
    // distinct CPU experts (passing `n_agent_entries` for telemetry).
    bool should_dispatch_agent(size_t agent_idx, int32_t n_agent_distinct, int32_t n_cpu_distinct,
                               int32_t n_agent_entries = -1);

    // Online timing updates called after each layer completes (keyed on distinct experts).
    void observe_agent(size_t agent_idx, int32_t n_distinct, double elapsed_ms);
    void observe_cpu(int32_t n_distinct, double elapsed_ms);
    void observe_primary(double elapsed_ms);

    // Non-blocking poll of control acks across all agents (triggers resync if needed).
    void poll_acks();

    // Invoked at `--adapt-every` intervals alongside Primary's adaptive tier.
    // `usage` is the current (pre-decay) routing usage array [n_layers * n_expert].
    void on_adapt_round(const std::vector<float>& usage,
                        const std::vector<int32_t>& primary_res,
                        const std::vector<std::pair<int32_t, int32_t>>& primary_adopted,
                        const std::vector<std::pair<int32_t, int32_t>>& primary_evicted);

    // Diagnostics & counters
    const LinearLatencyModel& cpu_model() const { return cpu_model_; }
    const LinearLatencyModel& agent_model(size_t idx) const { return agent_models_[idx]; }
    double agent_c_move_ms(size_t idx) const;
    double primary_ms() const { return primary_ms_; }
    uint64_t gated_rounds() const { return gated_rounds_; }
    uint64_t gated_entries() const { return gated_entries_; }
    uint64_t adapt_rounds() const { return adapt_rounds_; }
    uint64_t dedup_demotes() const { return dedup_demotes_; }
    uint64_t cascade_promotes() const { return cascade_promotes_; }
    uint64_t miss_promotes() const { return miss_promotes_; }
    uint64_t move_cost_rejected() const { return move_cost_rejected_; }

private:
    int32_t n_layers_ = 0;
    int32_t n_expert_ = 0;
    std::vector<primary::AgentTier*> agents_;
    bool cost_model_enabled_ = true;
    int32_t max_swaps_per_agent_ = 8;

    LinearLatencyModel cpu_model_{};
    std::vector<LinearLatencyModel> agent_models_;
    std::vector<double> seeded_c_move_ms_;
    double primary_ms_ = 0.0;
    int64_t primary_samples_ = 0;
    uint32_t explore_counter_ = 0;

    uint64_t gated_rounds_ = 0;
    uint64_t gated_entries_ = 0;
    uint64_t adapt_rounds_ = 0;
    uint64_t dedup_demotes_ = 0;
    uint64_t cascade_promotes_ = 0;
    uint64_t miss_promotes_ = 0;
    uint64_t move_cost_rejected_ = 0;
};

}  // namespace control
}  // namespace franken
