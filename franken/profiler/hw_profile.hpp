// hw_profile.hpp — FrankenStrata Phase 2 hardware profile schema, JSON I/O, and tier planner.
//
// Captures host CPU/RAM and per-GPU (Primary + Foreign Agents) latency curves,
// usable VRAM, slot budgets, and IPC doorbell overhead in franken-hw-profile.json,
// and deterministically computes a ranked TierPlan (Primary selection, Agent order,
// slot budgets, and per-agent minimum-entry dispatch thresholds).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace franken {
namespace profiler {

struct BatchLatencyPoint {
    int32_t n_experts = 0;
    double latency_ms = 0.0;
};

struct CpuProfile {
    std::string cpu_model;
    int32_t logical_cores = 0;
    int32_t physical_cores = 0;
    uint64_t ram_total_bytes = 0;
    uint64_t ram_available_bytes = 0;
    std::vector<BatchLatencyPoint> expert_latency;  // n = 1, 2, 4, 8
    double ms_base = 0.0;                           // fitted intercept a_cpu (ms)
    double ms_per_expert = 0.0;                     // fitted slope b_cpu (ms/expert)
};

struct GpuProfile {
    std::string id;                   // e.g. "hip:0" or "/franken0"
    std::string vendor;               // "HIP", "SYCL", "CUDA", "CPU"
    std::string device_name;
    uint64_t vram_total_bytes = 0;
    uint64_t vram_usable_bytes = 0;
    uint32_t recommended_slots = 0;
    double pcie_h2d_gbps = 0.0;
    double pcie_d2h_gbps = 0.0;
    double a_ipc_ms = 0.0;            // /dev/shm doorbell + fixed launch overhead (ms)
    double b_per_expert_ms = 0.0;     // marginal compute latency per expert (ms/expert)
    std::vector<BatchLatencyPoint> expert_latency;  // n = 1, 2, 4, 8, 16
};

struct HwProfile {
    int32_t schema_version = 1;
    std::string timestamp_utc;
    std::string pack_dir;
    int32_t n_layers = 0;
    int32_t n_expert = 0;
    int32_t k = 0;
    int32_t n_embd = 0;
    uint64_t avg_blob_bytes = 0;
    CpuProfile cpu;
    std::vector<GpuProfile> gpus;
};

struct AgentTierPlan {
    std::string id;
    std::string vendor;
    std::string device_name;
    uint32_t slot_budget = 0;
    int32_t min_entries_threshold = 1;
    double a_ipc_ms = 0.0;
    double b_per_expert_ms = 0.0;
    double effective_experts_per_ms = 0.0;
};

struct TierPlan {
    int32_t primary_gpu_index = -1;
    std::string primary_id;
    std::string primary_vendor;
    std::string primary_device_name;
    uint32_t primary_slot_budget = 0;
    std::vector<AgentTierPlan> agents;
    std::string recommended_cli_flags;
};

// Fits T(n) = a + b * n (with a >= 0, b >= 0) by ordinary least squares.
void fit_linear_latency(const std::vector<BatchLatencyPoint>& pts,
                        double& out_a_ms, double& out_b_ms);

// Self-contained JSON serialization & parsing (zero external dependencies).
std::string serialize_hw_profile_json(const HwProfile& prof);
bool parse_hw_profile_json(const std::string& json, HwProfile& out, std::string& err);

bool save_hw_profile(const std::string& path, const HwProfile& prof, std::string& err);
bool load_hw_profile(const std::string& path, HwProfile& out, std::string& err);

// Deterministically ranks hardware into Primary + ordered Agent tiers with
// slot budgets and per-layer minimum-entry dispatch thresholds.
// `total_model_experts`: if 0, uses prof.n_layers * prof.n_expert.
// `preferred_primary_id`: optional override to pin a specific GPU id as Primary.
TierPlan recommend_tier_plan(const HwProfile& prof,
                             uint32_t total_model_experts = 0,
                             const std::string& preferred_primary_id = {});

// Finds the best matching GpuProfile for an agent (by shm id first, then vendor).
const GpuProfile* find_gpu_profile(const HwProfile& prof,
                                   const std::string& shm_name,
                                   const std::string& vendor_str);

}  // namespace profiler
}  // namespace franken
