// agent_tier.hpp — Primary-side interface to one foreign-vendor FrankenStrata agent.
//
// Each AgentTier owns one POSIX shared-memory segment and speaks the wire protocol
// (franken_wire.hpp / franken_ipc.hpp). The primary engine holds a vector of 1..N
// AgentTiers, assigns each the next-hottest unclaimed experts at OPEN, dispatches
// kind == 3 entries right after GpuPlanSink publish (overlapping Primary VRAM +
// Agent(s) + CPU), and handles deadline salvage and 250 ms heartbeat staleness.

#pragma once

#include "franken/wire/franken_ipc.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace franken {
namespace primary {

struct AgentSpec {
    std::string shm_name;
    uint32_t max_slots = 0;  // 0 = use full slot_count advertised by agent HELLO
};

// Parses "<shm_name>[,slots=N]" CLI argument into AgentSpec.
bool parse_agent_spec(const std::string& arg, AgentSpec& out, std::string& err);

class AgentTier {
public:
    AgentTier() = default;
    ~AgentTier();
    AgentTier(const AgentTier&) = delete;
    AgentTier& operator=(const AgentTier&) = delete;

    // Creates the shm segment, waits for HELLO, selects the next-hottest unclaimed
    // experts from `ranked` (falling back to layer-early if `ranked` is empty),
    // sends OPEN, waits for OPEN_ACK, populates the residency table, and marks
    // loaded experts in `claimed` (size n_layers * n_expert).
    bool open(const AgentSpec& spec,
              const wire::Geometry& geom_offer,
              const std::vector<uint64_t>& blob_bytes,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              std::vector<uint8_t>& claimed,
              double fixed_deadline_ms,
              uint32_t hello_timeout_ms,
              uint32_t open_timeout_ms,
              std::string& err);

    // Sends BYE, waits briefly for clean exit, and unmaps/unlinks the shm segment.
    void close();

    // True iff the agent is Ready and not yet marked Dead.
    bool alive() const { return alive_; }

    // True iff this agent can handle a window of (n_tok, k) within negotiated caps.
    bool can_dispatch(int64_t n_tok, int64_t k) const {
        return alive_ &&
               n_tok > 0 && k > 0 &&
               uint64_t(n_tok) <= max_tokens_ &&
               uint64_t(n_tok * k) <= max_entries_;
    }

    // True iff this agent is alive and holds (layer, expert) in its resident cache.
    bool holds(int64_t layer, int32_t expert) const {
        if (!alive_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
            return false;
        }
        return resident_[size_t(layer) * size_t(n_expert_) + size_t(expert)] != 0;
    }

    // Publishes a round descriptor for entries with `kind[i] == 3` and rings the
    // agent doorbell. If `allow_q8_1` is true and the agent negotiated
    // kCapQ8_1Activations, pre-quantizes x to q8_1 on the host (~3.5x smaller wire
    // payload, skips agent-side quantization).
    bool launch(int64_t layer,
                const float* x,
                const int32_t* ids,
                int64_t n_tok,
                int64_t k,
                const int32_t* kind,
                bool allow_q8_1 = false);

    bool launched() const { return launched_; }

    // Waits on the launched slot up to its deadline (or heartbeat staleness).
    // On Done: copies every launched entry's row from shm into `out` and returns true.
    // On Salvaged (deadline expiry / heartbeat stale / agent Dead): appends every
    // launched entry index into `salvaged_entries` for immediate CPU recompute and
    // returns false.
    bool finish(float* out, std::vector<int32_t>& salvaged_entries);

    // Phase 2 dynamic residency control (DESIGN §6.3):
    //   - send_demote: primary-authoritative and immediate for routing; bumps residency_gen
    //     and pushes CtrlType::Demote so the agent frees its slot.
    //   - send_promote: pushes CtrlType::Promote; expert becomes routable ONLY when
    //     poll_control_acks() receives a matching AckType::PromoteAck (safe by construction).
    //   - poll_control_acks: non-blocking drain at round/adapt boundaries. Returns false if
    //     a generation mismatch was observed (caller should call resync()).
    //   - resync: sends CtrlType::Resync and reinstalls the agent's exact resident bitmap.
    bool send_demote(int32_t layer, int32_t expert);
    bool send_promote(int32_t layer, int32_t expert);
    bool poll_control_acks();
    bool resync(uint32_t timeout_ms = 5000);

    // Diagnostics & counters
    const std::string& shm_name() const { return seg_.name(); }
    wire::AgentVendor vendor() const { return static_cast<wire::AgentVendor>(caps_.vendor); }
    const char* vendor_str() const;
    uint32_t advertised_slots() const { return caps_.slot_count; }
    uint32_t allocated_slots() const { return allocated_slots_; }
    uint32_t feature_flags() const { return caps_.feature_flags; }
    bool uses_q8_1() const { return use_q8_1_; }
    bool uses_compact_rows() const { return use_compact_rows_; }
    uint32_t residency_gen() const { return residency_gen_; }
    bool needs_resync() const { return needs_resync_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t resident() const { return resident_count_; }
    const std::vector<uint8_t>& resident_table() const { return resident_; }
    size_t pending_promote_count() const { return pending_promotes_.size(); }
    bool has_pending_promote(int64_t layer, int32_t expert) const;
    uint64_t rounds_launched() const { return rounds_launched_; }
    uint64_t rounds_done() const { return rounds_done_; }
    uint64_t rounds_salvaged() const { return rounds_salvaged_; }
    uint64_t entries_served() const { return entries_served_; }
    uint64_t promotes_sent() const { return promotes_sent_; }
    uint64_t promotes_acked() const { return promotes_acked_; }
    uint64_t demotes_sent() const { return demotes_sent_; }
    uint64_t resyncs_done() const { return resyncs_done_; }
    double c_move_ms() const { return c_move_ms_; }
    uint64_t move_samples() const { return move_samples_; }
    double last_round_ms() const { return last_round_ms_; }
    int32_t last_round_entries() const { return last_round_entries_; }
    int32_t last_round_distinct() const { return last_round_distinct_; }
    double open_fill_ms() const { return open_fill_ms_; }
    uint64_t open_fill_bytes() const { return open_fill_bytes_; }
    double ms_launch() const { return ms_launch_; }
    double ms_wait() const { return ms_wait_; }
    double ms_copy() const { return ms_copy_; }
    double current_deadline_ms() const { return double(compute_deadline_ns()) / 1e6; }

private:
    struct PendingPromote {
        int32_t layer = 0;
        int32_t expert = 0;
        uint32_t gen = 0;
        uint64_t ctrl_seq = 0;
        uint64_t send_time_ns = 0;
    };

    bool check_heartbeat_alive();
    uint64_t compute_deadline_ns() const;
    void record_latency_ns(uint64_t lat_ns);
    void mark_dead(const char* reason);
    void drain_error_acks();
    bool install_bitmap(uint64_t bitmap_offset, uint32_t gen, std::vector<uint8_t>* claimed_out, std::string& err);

    ipc::ShmSegment seg_;
    wire::Geometry geom_{};
    wire::Caps caps_{};
    uint32_t max_tokens_ = 0;
    uint32_t max_entries_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t n_embd_ = 0;

    bool alive_ = false;
    bool launched_ = false;
    bool needs_resync_ = false;
    bool use_q8_1_ = false;
    bool use_compact_rows_ = false;
    bool active_compact_out_ = false;
    uint32_t residency_gen_ = 0;
    uint64_t next_ctrl_seq_ = 1;
    uint64_t next_doorbell_seq_ = 0;
    uint64_t active_seq_ = 0;
    uint64_t launch_start_ns_ = 0;
    uint64_t active_deadline_ns_ = 0;

    std::vector<uint8_t> resident_;  // [n_layers * n_expert]
    int64_t resident_count_ = 0;
    uint32_t allocated_slots_ = 0;
    std::vector<int32_t> launched_entries_;
    std::unordered_map<uint64_t, PendingPromote> pending_promotes_;  // key = layer * n_expert + expert

    ipc::HeartbeatMonitor hb_{};
    double fixed_deadline_ms_ = 0.0;  // <= 0: adaptive max(2 ms, 8 * p95)
    std::deque<uint64_t> recent_lat_ns_;
    uint32_t consecutive_salvages_ = 0;

    uint64_t rounds_launched_ = 0;
    uint64_t rounds_done_ = 0;
    uint64_t rounds_salvaged_ = 0;
    uint64_t entries_served_ = 0;
    uint64_t promotes_sent_ = 0;
    uint64_t promotes_acked_ = 0;
    uint64_t demotes_sent_ = 0;
    uint64_t resyncs_done_ = 0;
    double c_move_ms_ = 0.0;
    uint64_t move_samples_ = 0;
    uint64_t last_promote_ack_ns_ = 0;
    double open_fill_ms_ = 0.0;
    uint64_t open_fill_bytes_ = 0;
    double last_round_ms_ = 0.0;
    int32_t last_round_entries_ = 0;
    int32_t last_round_distinct_ = 0;
    int32_t active_distinct_ = 0;
    double ms_launch_ = 0.0;
    double ms_wait_ = 0.0;
    double ms_copy_ = 0.0;
};

}  // namespace primary
}  // namespace franken
