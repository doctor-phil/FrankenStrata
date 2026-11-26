// franken_ipc.hpp — FrankenStrata OS IPC layer over POSIX shared memory & Linux futexes.
//
// Implements DESIGN §4.4.1 / §4.5 / §4.6 on top of the pure layout definitions
// in franken/wire/franken_wire.hpp:
//   - POSIX shm_open / ftruncate / mmap / shm_unlink segment lifecycle
//   - Non-private cross-process Linux futex wait/wake (FUTEX_WAIT / FUTEX_WAKE)
//     so host processes and Docker containers sharing /dev/shm synchronize cleanly
//   - CLOCK_MONOTONIC timestamps and bounded spin-before-park wait loops
//   - Lock-free SPSC control/ack ring helpers
//   - Deterministic variable-area partitioning (blob_bytes, fill lists, bitmap, error text)

#pragma once

#include "franken/wire/franken_wire.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace franken {
namespace ipc {

// ---------------------------------------------------------------------------
// Deterministic layout for the variable area beyond SegmentLayout::var_offset.
// Every sub-region has a single writer so primary and agent never race on
// variable-area allocation.
// ---------------------------------------------------------------------------

inline constexpr uint64_t kErrorTextCapacity = 4096;

struct SegmentPlan {
    wire::SegmentLayout layout{};
    uint64_t blob_bytes_offset = 0;      // uint64_t[n_layers], primary-written at creation
    uint64_t primary_pairs_offset = 0;   // int32_t[n_layers * n_expert][2], primary-written (OPEN / RESYNC)
    uint64_t primary_pairs_bytes = 0;
    uint64_t agent_bitmap_offset = 0;    // uint64_t[ceil(L * E / 64)], agent-written (OPEN_ACK)
    uint64_t agent_bitmap_bytes = 0;
    uint64_t agent_error_offset = 0;     // char[kErrorTextCapacity], agent-written (ERROR)
    uint64_t agent_error_bytes = kErrorTextCapacity;
    uint64_t total_bytes = 0;
};

// Computes the complete segment layout and variable-area plan for `g`.
SegmentPlan compute_segment_plan(const wire::Geometry& g);

// Computes the number of 64-bit bitmap words needed for L * E experts.
inline constexpr uint64_t bitmap_words(int32_t n_layers, int32_t n_expert) {
    const uint64_t total = uint64_t(n_layers) * uint64_t(n_expert);
    return (total + 63ull) / 64ull;
}

// ---------------------------------------------------------------------------
// Monotonic clock & CPU pause
// ---------------------------------------------------------------------------

uint64_t now_ns();
void cpu_relax();

// ---------------------------------------------------------------------------
// Cross-process Linux futex primitives (NON-PRIVATE: FUTEX_WAIT / FUTEX_WAKE)
// ---------------------------------------------------------------------------

// Blocks on `*word` while it equals `expected`, up to `timeout_ns` (0 = wait indefinitely).
// Returns true if woken normally or value changed (EAGAIN), false on timeout (ETIMEDOUT).
bool futex_wait_word(std::atomic<uint32_t>* word, uint32_t expected, uint64_t timeout_ns);

// Wakes one or all waiters blocked on `*word` across processes.
void futex_wake_one(std::atomic<uint32_t>* word);
void futex_wake_all(std::atomic<uint32_t>* word);

// Cross-process futex helpers operating on the low 32-bit half of a monotonic
// uint64_t counter (doorbell_seq). Safe on little-endian targets; every +1
// changes the low 32 bits.
bool futex_wait_u64_low(std::atomic<uint64_t>* counter, uint64_t expected, uint64_t timeout_ns);
void futex_wake_u64_low(std::atomic<uint64_t>* counter);

// ---------------------------------------------------------------------------
// Normalized POSIX shared memory name helper ("/name", no nested slashes)
// ---------------------------------------------------------------------------

bool normalize_shm_name(const std::string& raw, std::string& out, std::string& err);

// ---------------------------------------------------------------------------
// POSIX shared-memory segment owner / attacher
// ---------------------------------------------------------------------------

class ShmSegment {
public:
    ShmSegment() = default;
    ~ShmSegment();
    ShmSegment(const ShmSegment&) = delete;
    ShmSegment& operator=(const ShmSegment&) = delete;
    ShmSegment(ShmSegment&& other) noexcept;
    ShmSegment& operator=(ShmSegment&& other) noexcept;

    // Primary side: creates a fresh POSIX shm segment, zero-initializes it,
    // writes Geometry + blob_bytes[] + descriptor slot offsets, and sets
    // agent_state = Absent.
    bool create(const std::string& name, const wire::Geometry& geom_offer,
                const std::vector<uint64_t>& blob_bytes, std::string& err);

    // Agent side: opens an existing POSIX shm segment and validates magic,
    // protocol version, header size, and total size.
    bool attach(const std::string& name, std::string& err);

    void close();

    bool valid() const { return base_ != nullptr; }
    bool is_owner() const { return owner_; }
    const std::string& name() const { return name_; }
    uint64_t size() const { return size_; }
    uint8_t* base() { return base_; }
    const uint8_t* base() const { return base_; }

    const SegmentPlan& plan() const { return plan_; }
    wire::ControlBlock* control() { return reinterpret_cast<wire::ControlBlock*>(base_); }
    const wire::ControlBlock* control() const {
        return reinterpret_cast<const wire::ControlBlock*>(base_);
    }

    wire::DescriptorHeader* descriptor(uint64_t seq) {
        const uint64_t idx = seq % wire::kRingSlots;
        return reinterpret_cast<wire::DescriptorHeader*>(
            base_ + plan_.layout.desc_offset + idx * plan_.layout.desc_slot_bytes);
    }
    const wire::DescriptorHeader* descriptor(uint64_t seq) const {
        const uint64_t idx = seq % wire::kRingSlots;
        return reinterpret_cast<const wire::DescriptorHeader*>(
            base_ + plan_.layout.desc_offset + idx * plan_.layout.desc_slot_bytes);
    }

    float* row_slot(uint64_t seq) {
        const uint64_t idx = seq % wire::kRingSlots;
        return reinterpret_cast<float*>(
            base_ + plan_.layout.row_offset + idx * plan_.layout.row_slot_bytes);
    }
    const float* row_slot(uint64_t seq) const {
        const uint64_t idx = seq % wire::kRingSlots;
        return reinterpret_cast<const float*>(
            base_ + plan_.layout.row_offset + idx * plan_.layout.row_slot_bytes);
    }

    wire::MsgRing<wire::CtrlMsg, wire::kControlRingSlots>* ctrl_ring() {
        return reinterpret_cast<wire::MsgRing<wire::CtrlMsg, wire::kControlRingSlots>*>(
            base_ + plan_.layout.ctrl_offset);
    }
    wire::MsgRing<wire::AckMsg, wire::kAckRingSlots>* ack_ring() {
        return reinterpret_cast<wire::MsgRing<wire::AckMsg, wire::kAckRingSlots>*>(
            base_ + plan_.layout.ack_offset);
    }

    const uint64_t* blob_bytes_table() const {
        return reinterpret_cast<const uint64_t*>(base_ + plan_.blob_bytes_offset);
    }
    uint64_t* blob_bytes_table() {
        return reinterpret_cast<uint64_t*>(base_ + plan_.blob_bytes_offset);
    }

    bool var_range_ok(uint64_t offset, uint64_t bytes) const {
        return offset >= plan_.layout.var_offset && bytes <= size_ && offset <= size_ - bytes;
    }
    uint8_t* var_ptr(uint64_t offset) { return base_ + offset; }
    const uint8_t* var_ptr(uint64_t offset) const { return base_ + offset; }

    // Convenience writers for the variable-area error buffer (agent -> primary).
    void write_error_text(const std::string& msg, uint64_t& out_offset, uint32_t& out_len);
    std::string read_error_text(uint64_t offset, uint32_t len) const;

private:
    std::string name_;
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    bool owner_ = false;
    SegmentPlan plan_{};
};

// ---------------------------------------------------------------------------
// Lock-free SPSC ring operations (safe across uint32_t wraparound for power-of-2 N)
// ---------------------------------------------------------------------------

template <typename Msg, uint32_t N>
inline uint32_t ring_size(const wire::MsgRing<Msg, N>& ring) {
    static_assert((N & (N - 1)) == 0, "MsgRing capacity must be a power of 2");
    const uint32_t h = ring.head.value.load(std::memory_order_acquire);
    const uint32_t t = ring.tail.value.load(std::memory_order_acquire);
    return uint32_t(h - t);
}

template <typename Msg, uint32_t N>
inline bool ring_empty(const wire::MsgRing<Msg, N>& ring) {
    return ring_size(ring) == 0;
}

template <typename Msg, uint32_t N>
inline bool ring_push(wire::MsgRing<Msg, N>& ring, const Msg& msg) {
    static_assert((N & (N - 1)) == 0, "MsgRing capacity must be a power of 2");
    const uint32_t h = ring.head.value.load(std::memory_order_relaxed);
    const uint32_t t = ring.tail.value.load(std::memory_order_acquire);
    if (uint32_t(h - t) >= N) return false;
    ring.slots[h & (N - 1)] = msg;
    ring.head.value.store(h + 1, std::memory_order_release);
    return true;
}

template <typename Msg, uint32_t N>
inline bool ring_pop(wire::MsgRing<Msg, N>& ring, Msg& out) {
    static_assert((N & (N - 1)) == 0, "MsgRing capacity must be a power of 2");
    const uint32_t t = ring.tail.value.load(std::memory_order_relaxed);
    const uint32_t h = ring.head.value.load(std::memory_order_acquire);
    if (h == t) return false;
    out = ring.slots[t & (N - 1)];
    ring.tail.value.store(t + 1, std::memory_order_release);
    return true;
}

// ---------------------------------------------------------------------------
// Bounded spin-before-park synchronization & salvage race guard
// ---------------------------------------------------------------------------

enum class SlotWaitResult {
    Done,      // Agent finished and release-stored SlotState::Done
    Salvaged,  // Deadline expired or agent declared Dead; primary CAS-stored Salvaged
};

// Primary waits on `slot->state` for Done up to `deadline_ns` (CLOCK_MONOTONIC).
// Spins up to `spin_us` before parking on FUTEX_WAIT. If `is_dead` returns true
// or `now_ns() >= deadline_ns`, atomically transitions Published/Claimed -> Salvaged
// via CAS so a racing agent finishing right at the deadline still cleanly wins Done.
SlotWaitResult wait_slot_or_salvage(wire::DescriptorHeader* slot,
                                    uint64_t deadline_ns,
                                    uint32_t spin_us = wire::kSpinPrimaryUs,
                                    const std::function<bool()>& is_dead = {});

// Agent waits until `doorbell_seq != last_seq`, a control message arrives, or
// `should_stop()` becomes true. Spins up to `spin_us` before parking on the
// cross-process doorbell futex in short slices (`park_slice_us`, default 2000 us).
void wait_agent_wakeup(ShmSegment& seg,
                       uint64_t last_doorbell_seq,
                       uint32_t spin_us = wire::kSpinAgentUs,
                       uint32_t park_slice_us = 2000,
                       const std::function<bool()>& should_stop = {});

// Agent side: attempts Published -> Claimed transition. Returns false if slot
// was already Salvaged or not in Published state.
bool try_claim_slot(wire::DescriptorHeader* slot);

// Agent side: verifies slot is still Claimed before copying rows, performs
// `write_rows()` if so, then atomically transitions Claimed -> Done via CAS and
// wakes the primary with FUTEX_WAKE. Returns true iff Done was published.
bool commit_slot_if_claimed(wire::DescriptorHeader* slot,
                            const std::function<void()>& write_rows);

// ---------------------------------------------------------------------------
// Heartbeat staleness tracker (primary side)
// ---------------------------------------------------------------------------

struct HeartbeatMonitor {
    uint64_t last_seen_beat = 0;
    uint64_t last_change_ns = 0;
    uint64_t max_age_ns = uint64_t(wire::kHeartbeatMaxAgeMs) * 1000000ull;

    void reset(uint64_t current_beat, uint64_t t_ns = now_ns()) {
        last_seen_beat = current_beat;
        last_change_ns = t_ns;
    }

    // Updates with the latest counter read from shared memory; returns true if stale.
    bool check_stale(uint64_t current_beat, uint64_t t_ns = now_ns()) {
        if (current_beat != last_seen_beat) {
            last_seen_beat = current_beat;
            last_change_ns = t_ns;
            return false;
        }
        return t_ns >= last_change_ns && (t_ns - last_change_ns) >= max_age_ns;
    }
};

// ---------------------------------------------------------------------------
// PackIdentity helper — deterministic FNV-1a hash shared by primary and agents
// ---------------------------------------------------------------------------

// Computes a deterministic 64-bit FNV-1a hash for PackIdentity::gguf_shard_list_hash
// from pack geometry, per-layer blob sizes, per-layer (gu_type, d_type), per-layer
// GGUF role offsets, and sorted unique shard file sizes.
uint64_t fnv1a_init();
void fnv1a_update(uint64_t& h, const void* data, size_t len);
void fnv1a_u64(uint64_t& h, uint64_t v);

bool pack_identity_matches(const wire::PackIdentity& a, const wire::PackIdentity& b);

// ---------------------------------------------------------------------------
// Phase 3 canonical q8_1 wire activation quantizer (matches q8_1_store bit-for-bit)
// ---------------------------------------------------------------------------

// Quantizes `n_rows` rows of `n_cols` floats (n_cols a multiple of 32) into
// `n_rows * (n_cols / 32) * 36` bytes of ggml block_q8_1 ({fp16 d, fp16 s, int8 qs[32]}).
void quantize_q8_1_rows_host(const float* x, int32_t n_rows, int32_t n_cols, uint8_t* dst);

}  // namespace ipc
}  // namespace franken
