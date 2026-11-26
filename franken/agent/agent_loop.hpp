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
    // `staging_out[i * H .. (i+1) * H)`. Must synchronize the device queue BEFORE
    // returning so the caller can immediately verify SlotState::Claimed and copy
    // rows into the shared-memory row arena.
    virtual bool compute_slot(int32_t layer,
                              int32_t n_tok,
                              int32_t k,
                              const int32_t* ids,
                              const int32_t* kind,
                              const float* x,
                              float* staging_out,
                              std::string& err) = 0;

    // Phase 2 dynamic residency hooks (cleanly stubbed in Phase 1).
    virtual bool promote(int32_t /*layer*/, int32_t /*expert*/, std::string& err) {
        err = "PROMOTE is not enabled in Phase 1 static residency";
        return false;
    }
    virtual bool demote(int32_t /*layer*/, int32_t /*expert*/, std::string& err) {
        err = "DEMOTE is not enabled in Phase 1 static residency";
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
