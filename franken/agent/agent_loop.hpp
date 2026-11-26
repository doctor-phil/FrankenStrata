// agent_loop.hpp — Vendor-neutral FrankenStrata agent runtime loop (DESIGN §4.3, §4.4, §6.1).
//
// Maps the primary's POSIX shared-memory segment, verifies Geometry & PackIdentity,
// posts HELLO + Caps, runs a 10 ms heartbeat thread, handles control messages
// (OPEN, BYE, plus PROMOTE/DEMOTE/RESYNC hooks), and drives the doorbell spin/park
// compute loop with strict SlotState::Claimed salvage checking before writing rows.

#pragma once

#include "franken/wire/franken_ipc.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace franken {
namespace agent {

class AgentBackend {
public:
    virtual ~AgentBackend() = default;

    virtual wire::AgentVendor vendor() const = 0;

    // Opens the pack, verifies compatibility with `g` and `blob_bytes[0..n_layers)`,
    // probes hardware capacity, and returns `out_caps` and `out_pack`.
    virtual bool init(const wire::Geometry& g,
                      const uint64_t* blob_bytes,
                      wire::Caps& out_caps,
                      wire::PackIdentity& out_pack,
                      std::string& err) = 0;

    // Allocates cache slots and loads as many (layer, expert) pairs from `pairs`
    // (hottest-first) as fit in hardware memory. Short loads truncate cleanly into
    // `out_loaded` (never corrupt).
    virtual bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                           std::vector<std::pair<int32_t, int32_t>>& out_loaded,
                           std::string& err) = 0;

    // Computes all entries `i` in [0, n_tok * k) with `kind[i] == 3` into
    // `staging_out`. Must synchronize the device queue BEFORE returning.
    // Phase 1 entry point (FP32 x, sparse staging_out[i * H]).
    virtual bool compute_slot(int32_t layer,
                              int32_t n_tok,
                              int32_t k,
                              const int32_t* ids,
                              const int32_t* kind,
                              const float* x,
                              float* staging_out,
                              std::string& err) = 0;

    // Phase 3 extended compute entry point supporting DescriptorHeader::flags:
    //   - kDescFlagActQ8_1: x_data holds n_tok * (H/32) * 36 B pre-quantized q8_1 blocks
    //   - kDescFlagCompactOut: staging_out holds n_agent_entries contiguous rows [0..n_agent_entries)
    //     in router order instead of sparse [entry * H] rows.
    // Default implementation delegates to compute_slot() for backward compatibility.
    virtual bool compute_slot_ex(int32_t layer,
                                 int32_t n_tok,
                                 int32_t k,
                                 const int32_t* ids,
                                 const int32_t* kind,
                                 const void* x_data,
                                 uint32_t desc_flags,
                                 float* staging_out,
                                 std::string& err) {
        if ((desc_flags & wire::kDescFlagActQ8_1) != 0) {
            err = "AgentBackend: pre-quantized q8_1 activations not supported by this backend";
            return false;
        }
        return compute_slot(layer, n_tok, k, ids, kind,
                            static_cast<const float*>(x_data), staging_out, err);
    }

    // Phase 2/3 dynamic residency hooks.
    // NOTE: AgentLoop runs promote() on a dedicated background refill thread so
    // compute doorbells never stall on blob reads / H2D copies. Backends that
    // implement promote() must synchronize internal slot tables with demote()
    // and compute_slot_ex().
    virtual bool promote(int32_t /*layer*/, int32_t /*expert*/, std::string& err) {
        err = "PROMOTE is not enabled in this backend";
        return false;
    }
    virtual bool demote(int32_t /*layer*/, int32_t /*expert*/, std::string& err) {
        err = "DEMOTE is not enabled in this backend";
        return false;
    }

    virtual void shutdown() = 0;
};

struct AgentLoopConfig {
    std::string shm_name;
    uint32_t attach_timeout_ms = 30000;
    uint32_t spin_us = wire::kSpinAgentUs;
    uint32_t park_slice_us = 2000;
    bool verbose = true;
};

struct AgentLoopStats {
    uint64_t rounds_seen = 0;
    uint64_t rounds_computed = 0;
    uint64_t rounds_salvaged_before_claim = 0;
    uint64_t rounds_salvaged_before_commit = 0;
    uint64_t entries_computed = 0;
    uint64_t async_promotes_completed = 0;
    uint64_t q8_1_rounds = 0;
    uint64_t compact_rounds = 0;
    double compute_ms = 0.0;
    double commit_ms = 0.0;
};

// Runs the full agent lifecycle until BYE, fatal error, or SIGINT/SIGTERM.
bool run_agent_loop(const AgentLoopConfig& cfg,
                    AgentBackend& backend,
                    AgentLoopStats& stats,
                    std::string& err);

// Installs SIGINT / SIGTERM handlers that gracefully stop run_agent_loop.
void install_agent_signal_handlers();

}  // namespace agent
}  // namespace franken
