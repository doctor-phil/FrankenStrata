// tier_controller.cpp — FrankenStrata Phase 2 Dynamic Multi-Tier Control Module.

#include "franken/control/tier_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace franken {
namespace control {

void LinearLatencyModel::seed(double prior_a_ms, double prior_b_ms, double weight) {
    seeded = true;
    a = std::max(0.0, prior_a_ms);
    b = std::max(0.0, prior_b_ms);
    A[0][0] = A[0][1] = A[1][0] = A[1][1] = 0.0;
    B[0] = B[1] = 0.0;
    // Seed normal equations across synthetic points n = 1, 2, 4, 8 with small prior weight
    // so online measurements smoothly refine the initial hardware profile.
    const double ns[4] = {1.0, 2.0, 4.0, 8.0};
    const double w = std::max(0.25, weight / 4.0);
    for (double n : ns) {
        const double y = a + b * n;
        A[0][0] += w;
        A[0][1] += w * n;
        A[1][0] += w * n;
        A[1][1] += w * n * n;
        B[0] += w * y;
        B[1] += w * n * y;
    }
}

void LinearLatencyModel::observe(int32_t n_experts, double elapsed_ms) {
    if (n_experts <= 0 || !std::isfinite(elapsed_ms) || elapsed_ms < 0.0) return;
    constexpr double kLambda = 0.99;
    const double x = double(n_experts);
    A[0][0] = kLambda * A[0][0] + 1.0;
    A[0][1] = kLambda * A[0][1] + x;
    A[1][0] = kLambda * A[1][0] + x;
    A[1][1] = kLambda * A[1][1] + x * x;
    B[0] = kLambda * B[0] + elapsed_ms;
    B[1] = kLambda * B[1] + x * elapsed_ms;
    ++samples;
    if ((samples & 7) == 0 || samples <= 8) solve();
}

void LinearLatencyModel::solve() {
    const double ridge = 1e-3 * std::max(1.0, A[0][0]);
    const double a00 = A[0][0];
    const double a01 = A[0][1];
    const double a10 = A[1][0];
    const double a11 = A[1][1] + ridge;
    const double det = a00 * a11 - a01 * a10;
    if (std::fabs(det) < 1e-12 || a00 < 1e-9) return;

    double na = (B[0] * a11 - a01 * B[1]) / det;
    double nb = (a00 * B[1] - a10 * B[0]) / det;
    if (nb < 0.0) {
        nb = 0.0;
        na = std::max(0.0, B[0] / a00);
    }
    if (na < 0.0) {
        na = 0.0;
        nb = a11 > 1e-12 ? std::max(0.0, B[1] / a11) : 0.0;
    }
    a = na;
    b = nb;
}

CascadePlan plan_cascade_swaps(int32_t n_layers,
                               int32_t n_expert,
                               const std::vector<float>& heat,
                               const std::vector<int32_t>& primary_res,
                               const std::vector<std::pair<int32_t, int32_t>>& primary_adopted,
                               const std::vector<std::pair<int32_t, int32_t>>& primary_evicted,
                               const std::vector<AgentResidencySnapshot>& agents,
                               int32_t max_swaps_per_agent,
                               float min_heat,
                               float min_gain) {
    CascadePlan plan{};
    const size_t total = size_t(n_layers) * size_t(n_expert);
    if (total == 0 || heat.size() < total || primary_res.size() < total || agents.empty()) {
        return plan;
    }

    // Track mutable residency view across planning steps so two agents never pick
    // the same expert and freed deduplication slots can be reused immediately.
    const size_t na = agents.size();
    std::vector<std::vector<uint8_t>> res(na);
    std::vector<uint32_t> free_slots(na, 0);
    std::vector<int32_t> swaps_used(na, 0);

    for (size_t a = 0; a < na; ++a) {
        if (!agents[a].alive || agents[a].resident == nullptr || agents[a].resident->size() < total) {
            continue;
        }
        res[a] = *agents[a].resident;
        uint32_t count = 0;
        for (uint8_t v : res[a]) count += v ? 1u : 0u;
        const uint32_t used = count + uint32_t(agents[a].pending_promotes);
        free_slots[a] = agents[a].max_slots > used ? (agents[a].max_slots - used) : 0u;
    }

    auto any_agent_holds = [&](size_t idx) -> bool {
        for (size_t a = 0; a < na; ++a) {
            if (!res[a].empty() && res[a][idx] != 0) return true;
        }
        return false;
    };

    // Build effective primary residency (including newly adopted, excluding evicted).
    std::vector<uint8_t> in_primary(total, 0);
    for (size_t i = 0; i < total; ++i) in_primary[i] = primary_res[i] >= 0 ? 1 : 0;
    for (const auto& [l, e] : primary_adopted) {
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            in_primary[size_t(l) * size_t(n_expert) + size_t(e)] = 1;
        }
    }
    for (const auto& [l, e] : primary_evicted) {
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            in_primary[size_t(l) * size_t(n_expert) + size_t(e)] = 0;
        }
    }

    // Rule 1: Deduplicate — when Primary adopts an expert an Agent already holds,
    // immediately DEMOTE it on the Agent to reclaim VRAM with zero duplication.
    for (const auto& [l, e] : primary_adopted) {
        if (l < 0 || l >= n_layers || e < 0 || e < 0 || e >= n_expert) continue;
        const size_t idx = size_t(l) * size_t(n_expert) + size_t(e);
        for (size_t a = 0; a < na; ++a) {
            if (res[a].empty() || res[a][idx] == 0) continue;
            res[a][idx] = 0;
            ++free_slots[a];
            CascadeOp op{};
            op.agent_idx = a;
            op.demote_layer = l;
            op.demote_expert = e;
            op.is_dedup = true;
            plan.ops.push_back(op);
        }
    }

    if (max_swaps_per_agent <= 0) return plan;

    // Build per-agent sorted victim lists (coldest resident experts first).
    struct Victim { float h; int32_t l; int32_t e; };
    std::vector<std::vector<Victim>> victims(na);
    std::vector<size_t> victim_cursor(na, 0);
    for (size_t a = 0; a < na; ++a) {
        if (res[a].empty()) continue;
        victims[a].reserve(256);
        for (int32_t l = 0; l < n_layers; ++l) {
            for (int32_t e = 0; e < n_expert; ++e) {
                const size_t idx = size_t(l) * size_t(n_expert) + size_t(e);
                if (res[a][idx] != 0) victims[a].push_back({heat[idx], l, e});
            }
        }
        std::sort(victims[a].begin(), victims[a].end(), [](const Victim& x, const Victim& y) {
            if (x.h != y.h) return x.h < y.h;
            if (x.l != y.l) return x.l < y.l;
            return x.e < y.e;
        });
    }

    // Phase 3 economic break-even gate:
    //   (h_in - h_out) * H_eff * (w_src - w_m) > kappa * C_move_ms
    // Falls back to plain hysteresis (min_gain) when move_cost_ready is false.
    auto passes_move_cost = [&](size_t a, float delta_h) -> bool {
        const AgentResidencySnapshot& s = agents[a];
        if (!s.move_cost_ready || s.c_move_ms <= 0.0) return true;
        const double delta_w = s.w_src_ms - s.w_m_ms;
        if (delta_w <= 0.0) {
            ++plan.move_cost_rejected;
            return false;
        }
        const double benefit_ms = double(delta_h) * s.h_eff * delta_w;
        const double threshold_ms = s.kappa * s.c_move_ms;
        if (benefit_ms > threshold_ms) return true;
        ++plan.move_cost_rejected;
        return false;
    };

    // Helper: attempt to place candidate (l, e, h) onto agent `a`, preferring
    // a free slot first, else evicting the coldest victim if gain >= min_gain
    // and the economic move-cost break-even gate passes.
    auto try_place_on_agent = [&](size_t a, int32_t l, int32_t e, float h, bool from_primary) -> bool {
        if (res[a].empty() || swaps_used[a] >= max_swaps_per_agent) return false;
        const size_t idx = size_t(l) * size_t(n_expert) + size_t(e);
        if (in_primary[idx] || any_agent_holds(idx)) return false;

        if (free_slots[a] > 0) {
            if (!passes_move_cost(a, h)) return false;
            --free_slots[a];
            res[a][idx] = 1;
            ++swaps_used[a];
            CascadeOp op{};
            op.agent_idx = a;
            op.promote_layer = l;
            op.promote_expert = e;
            op.gain = h;
            op.is_cascade_from_primary = from_primary;
            plan.ops.push_back(op);
            return true;
        }

        while (victim_cursor[a] < victims[a].size()) {
            const Victim& v = victims[a][victim_cursor[a]];
            const size_t vidx = size_t(v.l) * size_t(n_expert) + size_t(v.e);
            if (res[a][vidx] == 0) {
                ++victim_cursor[a];
                continue;
            }
            if (h < v.h + min_gain) return false;
            if (!passes_move_cost(a, h - v.h)) return false;
            ++victim_cursor[a];
            res[a][vidx] = 0;
            res[a][idx] = 1;
            ++swaps_used[a];
            CascadeOp op{};
            op.agent_idx = a;
            op.demote_layer = v.l;
            op.demote_expert = v.e;
            op.promote_layer = l;
            op.promote_expert = e;
            op.gain = h - v.h;
            op.is_cascade_from_primary = from_primary;
            plan.ops.push_back(op);
            return true;
        }
        return false;
    };

    // Rule 2: Cascade Eviction — offer Primary-evicted experts to the hottest eligible
    // Agent tier (Tier 1 first, then Tier 2..N) instead of dropping straight to CPU.
    struct Cand { float h; int32_t l; int32_t e; };
    std::vector<Cand> evicted_cands;
    evicted_cands.reserve(primary_evicted.size());
    for (const auto& [l, e] : primary_evicted) {
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) continue;
        const size_t idx = size_t(l) * size_t(n_expert) + size_t(e);
        if (!in_primary[idx] && !any_agent_holds(idx) && heat[idx] > 0.0f) {
            evicted_cands.push_back({heat[idx], l, e});
        }
    }
    std::sort(evicted_cands.begin(), evicted_cands.end(), [](const Cand& x, const Cand& y) {
        if (x.h != y.h) return x.h > y.h;
        if (x.l != y.l) return x.l < y.l;
        return x.e < y.e;
    });
    for (const Cand& c : evicted_cands) {
        for (size_t a = 0; a < na; ++a) {
            if (try_place_on_agent(a, c.l, c.e, c.h, /*from_primary=*/true)) break;
        }
    }

    // Rule 3: Promote hot CPU misses into Agent VRAM (ordered Tier 1 -> Tier N).
    std::vector<Cand> cpu_cands;
    for (int32_t l = 0; l < n_layers; ++l) {
        for (int32_t e = 0; e < n_expert; ++e) {
            const size_t idx = size_t(l) * size_t(n_expert) + size_t(e);
            if (!in_primary[idx] && !any_agent_holds(idx) && heat[idx] >= min_heat) {
                cpu_cands.push_back({heat[idx], l, e});
            }
        }
    }
    std::sort(cpu_cands.begin(), cpu_cands.end(), [](const Cand& x, const Cand& y) {
        if (x.h != y.h) return x.h > y.h;
        if (x.l != y.l) return x.l < y.l;
        return x.e < y.e;
    });

    for (const Cand& c : cpu_cands) {
        bool all_full = true;
        for (size_t a = 0; a < na; ++a) {
            if (!res[a].empty() && swaps_used[a] < max_swaps_per_agent) {
                all_full = false;
                if (try_place_on_agent(a, c.l, c.e, c.h, /*from_primary=*/false)) break;
            }
        }
        if (all_full) break;
    }

    return plan;
}

void TierController::init(int32_t n_layers,
                          int32_t n_expert,
                          const std::vector<primary::AgentTier*>& agents,
                          bool cost_model_enabled,
                          int32_t max_swaps_per_agent) {
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    agents_ = agents;
    cost_model_enabled_ = cost_model_enabled;
    max_swaps_per_agent_ = max_swaps_per_agent;
    cpu_model_ = LinearLatencyModel{};
    agent_models_.assign(agents.size(), LinearLatencyModel{});
    seeded_c_move_ms_.assign(agents.size(), 0.0);
    for (size_t i = 0; i < agents.size(); ++i) {
        if (agents[i] != nullptr && agents[i]->open_fill_ms() > 0.0 && agents[i]->resident() > 0) {
            // Seed initial per-blob move cost from observed OPEN fill duration.
            seeded_c_move_ms_[i] = agents[i]->open_fill_ms() / double(agents[i]->resident());
        }
    }
    primary_ms_ = 0.0;
    primary_samples_ = 0;
    explore_counter_ = 0;
    gated_rounds_ = 0;
    gated_entries_ = 0;
    adapt_rounds_ = 0;
    dedup_demotes_ = 0;
    cascade_promotes_ = 0;
    miss_promotes_ = 0;
    move_cost_rejected_ = 0;
}

void TierController::seed_from_profile(const profiler::HwProfile& prof) {
    if (prof.cpu.ms_base > 0.0 || prof.cpu.ms_per_expert > 0.0) {
        cpu_model_.seed(prof.cpu.ms_base, prof.cpu.ms_per_expert, /*weight=*/4.0);
    }
    const double avg_blob_b = prof.avg_blob_bytes > 0 ? double(prof.avg_blob_bytes) : 2048000.0;
    for (size_t i = 0; i < agents_.size(); ++i) {
        if (agents_[i] == nullptr) continue;
        const profiler::GpuProfile* gp =
            profiler::find_gpu_profile(prof, agents_[i]->shm_name(), agents_[i]->vendor_str());
        if (gp != nullptr) {
            if (gp->a_ipc_ms > 0.0 || gp->b_per_expert_ms > 0.0) {
                agent_models_[i].seed(gp->a_ipc_ms, gp->b_per_expert_ms, /*weight=*/4.0);
            }
            if (gp->pcie_h2d_gbps > 0.0) {
                seeded_c_move_ms_[i] = avg_blob_b / (gp->pcie_h2d_gbps * 1e6);
            }
            std::fprintf(stderr,
                         "strata franken-control: seeded agent[%zu] (%s/%s) prior T(n)=%.3f+%.3f*n ms, C_move=%.3f ms\n",
                         i, agents_[i]->shm_name().c_str(), agents_[i]->vendor_str(),
                         agent_models_[i].a, agent_models_[i].b, seeded_c_move_ms_[i]);
        }
    }
    if (cpu_model_.seeded) {
        std::fprintf(stderr, "strata franken-control: seeded CPU prior T_cpu(n) = %.3f + %.3f * n ms\n",
                     cpu_model_.a, cpu_model_.b);
    }
}

double TierController::agent_c_move_ms(size_t idx) const {
    if (idx < agents_.size() && agents_[idx] != nullptr && agents_[idx]->move_samples() > 0) {
        return agents_[idx]->c_move_ms();
    }
    return idx < seeded_c_move_ms_.size() ? seeded_c_move_ms_[idx] : 0.0;
}

bool TierController::should_dispatch_agent(size_t agent_idx,
                                           int32_t n_agent_distinct,
                                           int32_t n_cpu_distinct,
                                           int32_t n_agent_entries) {
    if (n_agent_distinct <= 0) return false;
    if (!cost_model_enabled_ || agent_idx >= agent_models_.size()) return true;

    const LinearLatencyModel& am = agent_models_[agent_idx];
    if (!am.ready() || !cpu_model_.ready()) return true;

    // Periodic exploration so online fits keep observing agent latency even if usually gated.
    if ((++explore_counter_ & 63u) == 0) return true;

    const double t_agent = am.predict(n_agent_distinct);
    const double t_cpu_with = cpu_model_.predict(n_cpu_distinct);
    const double t_cpu_without = cpu_model_.predict(n_cpu_distinct + n_agent_distinct);

    const double makespan_with = std::max({primary_ms_, t_agent, t_cpu_with});
    const double makespan_without = std::max(primary_ms_, t_cpu_without);

    if (makespan_with <= makespan_without) {
        return true;
    }
    ++gated_rounds_;
    gated_entries_ += uint64_t(n_agent_entries > 0 ? n_agent_entries : n_agent_distinct);
    return false;
}

void TierController::observe_agent(size_t agent_idx, int32_t n_entries, double elapsed_ms) {
    if (agent_idx < agent_models_.size()) {
        agent_models_[agent_idx].observe(n_entries, elapsed_ms);
    }
}

void TierController::observe_cpu(int32_t n_entries, double elapsed_ms) {
    cpu_model_.observe(n_entries, elapsed_ms);
}

void TierController::observe_primary(double elapsed_ms) {
    if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0) return;
    if (primary_samples_ < 16) {
        primary_ms_ += (elapsed_ms - primary_ms_) / double(++primary_samples_);
    } else {
        primary_ms_ += 0.05 * (elapsed_ms - primary_ms_);
    }
}

void TierController::poll_acks() {
    for (auto* ag : agents_) {
        if (ag == nullptr) continue;
        if (!ag->poll_control_acks() || ag->needs_resync()) {
            (void)ag->resync(2000);
        }
    }
}

void TierController::on_adapt_round(const std::vector<float>& usage,
                                    const std::vector<int32_t>& primary_res,
                                    const std::vector<std::pair<int32_t, int32_t>>& primary_adopted,
                                    const std::vector<std::pair<int32_t, int32_t>>& primary_evicted) {
    ++adapt_rounds_;
    poll_acks();

    std::vector<AgentResidencySnapshot> snaps(agents_.size());
    for (size_t i = 0; i < agents_.size(); ++i) {
        if (agents_[i] == nullptr) continue;
        snaps[i].alive = agents_[i]->alive();
        snaps[i].max_slots = agents_[i]->allocated_slots();
        snaps[i].pending_promotes = agents_[i]->pending_promote_count();
        snaps[i].resident = &agents_[i]->resident_table();
        const double cmove = agent_c_move_ms(i);
        if (cpu_model_.ready() && agent_models_[i].ready() && cmove > 0.0) {
            snaps[i].move_cost_ready = true;
            snaps[i].w_src_ms = cpu_model_.b;
            snaps[i].w_m_ms = agent_models_[i].b;
            snaps[i].c_move_ms = cmove;
            snaps[i].h_eff = 10.0;
            snaps[i].kappa = 1.0;
        }
    }

    const CascadePlan plan = plan_cascade_swaps(
        n_layers_, n_expert_, usage, primary_res,
        primary_adopted, primary_evicted, snaps, max_swaps_per_agent_);
    move_cost_rejected_ += plan.move_cost_rejected;

    for (const CascadeOp& op : plan.ops) {
        if (op.agent_idx >= agents_.size() || agents_[op.agent_idx] == nullptr) continue;
        auto* ag = agents_[op.agent_idx];
        if (!ag->alive()) continue;

        // Always DEMOTE before PROMOTE so the agent frees the slot before filling.
        if (op.demote_layer >= 0 && op.demote_expert >= 0) {
            if (ag->send_demote(op.demote_layer, op.demote_expert) && op.is_dedup) {
                ++dedup_demotes_;
            }
        }
        if (op.promote_layer >= 0 && op.promote_expert >= 0) {
            if (ag->send_promote(op.promote_layer, op.promote_expert)) {
                if (op.is_cascade_from_primary) ++cascade_promotes_;
                else ++miss_promotes_;
            }
        }
    }
}

}  // namespace control
}  // namespace franken
