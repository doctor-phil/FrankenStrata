// franken_wire.hpp — FrankenStrata shared-memory wire protocol (layout spec).
//
// Implements the segment layout, messages, and synchronization contract of
// DESIGN.md §4 ("Shared-memory protocol") as a self-contained C++17 header.
// This header defines BYTES AND MEANINGS ONLY: no OS calls, no runtime types.
// Segment creation/mapping, the futex wait/wake, and the monotonic clock are
// the three OS primitives of DESIGN §4.6 and live behind the platform layer;
// nothing OS-specific is allowed to leak in here (§4.6, §13 Q4).
//
// Vendor-neutrality rules for the wire (DESIGN §4.1):
//   - On the wire: int32 expert ids and entry indices; fp32 rows; geometry in
//     engine-common terms (n_layers, n_expert, per-layer blob_bytes); quant
//     types identified by their ggml type ids (third_party/ggml/ggml-common.h
//     is the one vocabulary both engine trees already share).
//   - NEVER on the wire: device pointers, stream/event handles, USM / CUDA /
//     HIP / SYCL types, or the primary's plan layout. Experts are referred to
//     only as (layer, expert) into the shared pack; each side resolves them
//     through its own cache.
//   - Phase 1 ships fp32 activations. The q8_1 activation image + fp32 scale
//     sidecar (§3.2) gets its canonical byte definition in Phase 3 (§4.1).
//
// Non-preclusion (DESIGN §13 Q6): the descriptor carries a payload_kind field
// so the wire format does not hardcode "expert row" as the only payload class
// forever. v1 implements expert rows only.
//
// The upstream caps the protocol inherits via the HELLO handshake instead of
// hardcoding (§3.1): verify windows are capped at MAXT = 8 tokens and
// kMaxWindowEntries = 128 entries (kernels/cpu/expert.hpp:143,
// src/core/expert_source.cpp:4791). The constants below are the v1 defaults
// the primary offers in its geometry block; the agent's HELLO caps are the
// values actually in force for the session.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace franken {
namespace wire {

// ---------------------------------------------------------------------------
// Identity and versioning (DESIGN §4.2: control block opens with magic +
// protocol version, so a mismatched peer is refused instead of misread).
// ---------------------------------------------------------------------------

inline constexpr uint64_t kMagic = 0x4652414E4B454E31ull;  // "FRANKEN1"
inline constexpr uint32_t kProtocolVersion = 1;

// ---------------------------------------------------------------------------
// Session limits (DESIGN §3.1, §4.2 "caps").
// ---------------------------------------------------------------------------

inline constexpr uint32_t kRingSlots = 8;            // R, descriptor/row rings (§4.2)
inline constexpr uint32_t kDefaultMaxTokens = 8;     // MAXT (§3.1)
inline constexpr uint32_t kDefaultMaxEntries = 128;  // kMaxWindowEntries (§3.1)

// Control/ack ring depths are an implementation choice (the design fixes the
// descriptor ring at R = 8 and leaves control rings "small messages", §4.2).
inline constexpr uint32_t kControlRingSlots = 256;
inline constexpr uint32_t kAckRingSlots = 256;

// ---------------------------------------------------------------------------
// Timing / degradation constants (DESIGN §4.4.1, §4.5).
// ---------------------------------------------------------------------------

inline constexpr uint32_t kHeartbeatPeriodMs = 10;    // agent beats >= every 10 ms (§4.2)
inline constexpr uint32_t kHeartbeatMaxAgeMs = 250;   // staleness => dead for routing (§4.5)
inline constexpr uint32_t kSpinPrimaryUs = 50;        // primary spin before futex park (§4.4.1)
inline constexpr uint32_t kSpinAgentUs = 200;         // agent doorbell spin fuse, >= 200 us (§4.4.1)
inline constexpr uint32_t kMinDeadlineMs = 2;         // deadline = max(2 ms, 8 x rolling p95) (§4.5)
inline constexpr uint32_t kDeadlineP95Multiplier = 8; // (§4.5)
inline constexpr uint32_t kMaxSalvagesBeforeDead = 3; // 3 salvages in a window => DEAD until RESYNC (§4.5)

// ---------------------------------------------------------------------------
// Agent lifecycle state (DESIGN §4.2: agent_state, written by the agent).
// Only in state Ready may dispatch assign kind 3 (§4.3 step 5).
// ---------------------------------------------------------------------------

enum class AgentState : uint32_t {
    Absent = 0,   // segment created, no agent has mapped it
    Loading = 1,  // agent mapped, pack open, HELLO posted; fill in progress
    Ready = 2,    // OPEN_ACK installed; kind-3 assignment permitted
    Dead = 3,     // agent declares shutdown (or primary marks it after BYE)
};

// ---------------------------------------------------------------------------
// Descriptor slot state word (DESIGN §4.2, §4.4). One word per slot carries
// both the state machine and the salvage guard, and doubles as the slot's
// futex word (§4.4.1: "the futex-on-the-state-word fuses the wakeup with the
// state check itself"; §4.5 step 3: "DONE/salvage live in one word: the
// agent's write-guard and the primary's wait are the same atomic").
//
//   Empty -> Published -> Claimed -> Done | Salvaged
//
// Salvaged is written by the PRIMARY (deadline expiry, §4.5): it is the epoch
// bump. An agent that finishes late re-reads this word before writing rows;
// state != Claimed means discard results and write nothing.
// ---------------------------------------------------------------------------

enum class SlotState : uint32_t {
    Empty = 0,
    Published = 1,
    Claimed = 2,
    Done = 3,
    Salvaged = 4,
};

// ---------------------------------------------------------------------------
// Payload kinds (DESIGN §13 Q6 non-preclusion). v1 defines exactly one.
// ---------------------------------------------------------------------------

enum class PayloadKind : uint32_t {
    ExpertRows = 0,  // ids/kind/x describe routed expert entries; rows are fp32[n][H]
    // Future payload classes (other stateless overflow) extend this enum
    // without a protocol redesign; v1 agents must refuse unknown kinds.
};

// ---------------------------------------------------------------------------
// Agent vendor self-identification (DESIGN §5.4: "--franken-agent-vendor is
// not needed — the agent self-identifies in HELLO; the primary neither knows
// nor cares which runtime is behind the segment"). Carried for logging and
// the compatibility matrix (§14), never for dispatch decisions.
// ---------------------------------------------------------------------------

enum class AgentVendor : uint32_t {
    Unknown = 0,
    Sycl = 1,
    Hip = 2,
    Cuda = 3,
    Cpu = 4,  // the reference CPU agent (§14 conformance floor)
};

// ---------------------------------------------------------------------------
// Pack identity (DESIGN §4.3 step 1: "the pack identity: pack dir +
// experts.bin size / GGUF shard list hash, so the agent can refuse a
// mismatched pack instead of computing garbage").
// ---------------------------------------------------------------------------

enum class PackFormat : uint32_t {
    ExpertsBin = 0,  // experts.bin — blob index layer * n_expert + expert, no padding (§4.1)
    GgufRoles = 1,   // native GGUF role slices (gate/up/down) (§4.1)
};

struct PackIdentity {
    uint32_t format;              // PackFormat
    uint32_t reserved0;
    uint64_t experts_bin_size;    // bytes; 0 when format == GgufRoles
    uint64_t gguf_shard_list_hash;  // hash of the shard list; 0 when ExpertsBin
};

// ---------------------------------------------------------------------------
// Geometry block (DESIGN §4.2, §4.3 step 1). Written by the primary at
// segment creation, before any agent maps the segment. blob_bytes[] itself
// lives in the segment's variable area (offset below) — one uint64 per layer.
// ---------------------------------------------------------------------------

struct Geometry {
    uint64_t magic;             // kMagic
    uint32_t protocol_version;  // kProtocolVersion
    uint32_t header_bytes;      // sizeof(Geometry), for forward checks
    int32_t n_layers;           // L
    int32_t n_expert;           // E
    int32_t k;                  // routed experts per token
    int32_t n_embd;             // H — fp32 row width
    uint32_t max_tokens;        // offered window cap (<= kDefaultMaxTokens in v1)
    uint32_t max_entries;       // offered entry cap (<= kDefaultMaxEntries in v1)
    PackIdentity pack;          // what both sides must open (§4.3)
    uint64_t blob_bytes_offset; // segment offset of uint64 blob_bytes[n_layers]
    uint64_t segment_bytes;     // total segment size the primary created
};

// ---------------------------------------------------------------------------
// Caps (DESIGN §4.2: "caps: max tokens/window, max entries/window (from
// HELLO)"; §4.3 step 2: the agent "answers with its caps and slot count").
// Written by the agent into the control block as part of HELLO, before
// agent_state leaves Absent. The session runs at min(geometry offer, caps).
// ---------------------------------------------------------------------------

struct Caps {
    uint32_t max_tokens_per_window;
    uint32_t max_entries_per_window;
    uint32_t slot_count;  // ExpertCache slots the agent allocated (§4.3 step 2)
    uint32_t vendor;      // AgentVendor (informational only, §5.4)
};

// ---------------------------------------------------------------------------
// Cache-line-isolated control fields (DESIGN §4.2: "one cache-line-isolated
// field per writer"). Each field below is written by exactly one side; the
// alignas(64) padding keeps the other side's reads from false-sharing.
// ---------------------------------------------------------------------------

template <typename T>
struct alignas(64) WriterField {
    std::atomic<T> value;
    // The futex word for slot states is the descriptor's state word, not a
    // control field; control fields are polled or read at window boundaries.
};
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "wire atomics must be lock-free to work cross-process");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "wire atomics must be lock-free to work cross-process");

// ---------------------------------------------------------------------------
// Control block (DESIGN §4.2). One per segment; the segment has exactly one
// agent, so there is exactly one control block at offset 0.
// ---------------------------------------------------------------------------

struct ControlBlock {
    Geometry geometry;  // primary-written, immutable after creation

    // Agent-written fields.
    WriterField<uint32_t> agent_state;  // AgentState (§4.2)
    WriterField<uint64_t> heartbeat;    // monotonic counter, +1 >= every 10 ms (§4.2)

    // Primary-written fields.
    WriterField<uint32_t> residency_gen;  // generation of the residency table in force (§4.2)
    WriterField<uint64_t> doorbell_seq;   // last published round sequence (§4.2)

    // Agent-written once at HELLO (§4.2 caps, §4.3 step 2). Plain (not
    // atomic): readers observe it only after agent_state != Absent, and the
    // state store that publishes it is release-ordered (§4.3).
    Caps caps;
};

// ---------------------------------------------------------------------------
// Descriptor slot (DESIGN §4.2). R = kRingSlots slots. The slot's variable
// arenas (ids, kind, x) follow the header in segment memory; their sizes are
// functions of the session caps and H — see the layout helpers below.
//
//   header: seq, layer, n_tok, k, n_agent_entries, deadline_hint  (§4.2)
//   ids[]   int32[n_tok * k] — the window's routed ids (full, unfiltered)
//   kind[]  int32[n_tok * k] — primary's partition; agent reads kind == 3
//   x arena fp32[n_tok * H]  — this round's activations (Phase 1 format)
// ---------------------------------------------------------------------------

struct DescriptorHeader {
    // The state word is the slot's first field so its address is stable and
    // known to both sides: it is the futex word (§4.4.1) and the salvage
    // guard (§4.5). Written per the §4.4 sequence: primary release-stores
    // Published after writing ids/kind/x; agent claims, then release-stores
    // Done after rows are complete and visible; primary may store Salvaged
    // at deadline expiry.
    std::atomic<uint32_t> state;  // SlotState
    uint32_t payload_kind;        // PayloadKind (§13 Q6)
    uint64_t seq;                 // round sequence; must match doorbell/ack (§4.5 sanity)
    int32_t layer;
    int32_t n_tok;
    int32_t k;
    int32_t n_agent_entries;      // count of kind == 3 entries in kind[]
    uint64_t deadline_hint_ns;    // this round's deadline budget, ns (§4.5)
    // Arena offsets, relative to this slot's base, in bytes.
    uint32_t ids_offset;
    uint32_t kind_offset;
    uint32_t x_offset;
    uint32_t reserved0;
};

// Row arena (DESIGN §4.2): R slots of fp32[max_entries * H]. This region IS
// the primary's `out` buffer (replicated across the process boundary, as the
// in-process peer does with portable mapped host memory); the pool consumes
// rows from it exactly as it consumes CPU rows. Slot i of the row arena pairs
// with descriptor slot i. Rows land at out[entry] — entry index =
// token * k + j, router order (§4.4 step 3, the R4.2c rule).

// ---------------------------------------------------------------------------
// Control messages (DESIGN §4.2 control ring, primary -> agent):
//   OPEN(ranked fill list) · PROMOTE(layer, expert) · DEMOTE(layer, expert)
//   RESYNC · BYE
// Variable-length payloads (fill lists, residency tables) live in the
// segment's variable area and are referenced by offset, never copied through
// the ring. Blob bytes never cross the segment at all: loads read the pack
// through the page cache (§4.3 step 4), and PROMOTE moves zero blob bytes —
// the agent reads the blob from the shared pack itself (§5.3).
// ---------------------------------------------------------------------------

enum class CtrlType : uint32_t {
    Open = 1,    // ranked fill list follows in the variable area (§4.3 step 3)
    Promote = 2, // (layer, expert) -> agent loads from pack (§6.3)
    Demote = 3,  // (layer, expert) -> primary already un-routed it (§6.3)
    Resync = 4,  // re-send of the current residency table (§4.5.1, §6.3)
    Bye = 5,     // primary is done; agent may exit
};

struct OpenPayload {
    uint32_t gen;           // residency generation this fill installs (1 at first OPEN, §4.3)
    uint32_t n_fill;        // number of (layer, expert) pairs
    uint64_t fill_offset;   // segment offset of int32 pairs[n_fill][2], ranked hottest-first
};

struct PromotePayload {
    int32_t layer;
    int32_t expert;
    uint32_t gen;  // generation the primary will install on PROMOTE_ACK (§6.3)
};

struct DemotePayload {
    int32_t layer;
    int32_t expert;
    uint32_t gen;  // generation in force after this demote (§6.3)
};

struct ResyncPayload {
    uint32_t gen;            // generation of the table that follows
    uint32_t n_resident;     // number of (layer, expert) pairs
    uint64_t table_offset;   // segment offset of int32 pairs[n_resident][2]
};

struct CtrlMsg {
    uint32_t type;  // CtrlType
    uint32_t reserved0;
    uint64_t seq;   // control-ring sequence, for ack matching and logging
    union {
        OpenPayload open;
        PromotePayload promote;
        DemotePayload demote;
        ResyncPayload resync;
        // Bye carries no payload.
    };
};

// ---------------------------------------------------------------------------
// Control acknowledgements (DESIGN §4.2 control ack ring, agent -> primary):
//   OPEN_ACK(resident bitmap) · PROMOTE_ACK(gen) · DEMOTE_ACK(gen) ·
//   ERROR(text)
// plus HELLO, the agent's first message (§4.3 step 2).
//
// Generation discipline (DESIGN §6.3): a PROMOTE becomes routable only when
// its PROMOTE_ACK(gen) lands — "an expert is routable to the agent only when
// the agent has said it holds the bytes; a lost ACK = the expert stays
// CPU-served: safe by construction." A DEMOTE is primary-authoritative and
// immediate for routing; DEMOTE_ACK only confirms the slot was freed.
// A generation mismatch at a round boundary triggers RESYNC before further
// kind-3 assignments (§6.3).
// ---------------------------------------------------------------------------

enum class AckType : uint32_t {
    Hello = 1,      // agent mapped, pack verified, caps posted (§4.3 step 2)
    OpenAck = 2,    // resident bitmap actually achieved (§4.3 step 4)
    PromoteAck = 3, // blob loaded; carries the installed generation (§6.3)
    DemoteAck = 4,  // slot freed; carries the generation (§6.3)
    Error = 5,      // human-readable text in the variable area (§4.2)
};

struct HelloPayload {
    Caps caps;           // mirrors ControlBlock.caps (§4.2)
    uint64_t pack_size;  // pack size the agent opened; must match geometry.pack
};

struct OpenAckPayload {
    uint32_t gen;            // generation installed (echoes OPEN)
    uint32_t n_resident;     // experts actually loaded (short loads truncate, never corrupt, §4.3)
    uint64_t bitmap_offset;  // segment offset of the resident bitmap, ceil(L * E / 64) uint64 words
};

struct PromoteAckPayload {
    int32_t layer;
    int32_t expert;
    uint32_t gen;  // generation now in force on the agent (§6.3)
};

struct DemoteAckPayload {
    int32_t layer;
    int32_t expert;
    uint32_t gen;  // generation now in force on the agent (§6.3)
};

struct ErrorPayload {
    uint64_t text_offset;  // segment offset of UTF-8 text (not NUL-terminated)
    uint32_t text_len;
    uint32_t fatal;  // nonzero: agent is going Dead; zero: diagnostic only
};

struct AckMsg {
    uint32_t type;  // AckType
    uint32_t reserved0;
    uint64_t seq;   // echoes the CtrlMsg seq it answers (0 for Hello)
    union {
        HelloPayload hello;
        OpenAckPayload open_ack;
        PromoteAckPayload promote_ack;
        DemoteAckPayload demote_ack;
        ErrorPayload error;
    };
};

// ---------------------------------------------------------------------------
// Message rings (DESIGN §4.2). Single-producer / single-consumer: the
// primary alone writes the control ring, the agent alone writes the ack
// ring; each side advances only its own head. head/tail are writer-isolated
// per the control block's rule (§4.2).
// ---------------------------------------------------------------------------

template <typename Msg, uint32_t N>
struct MsgRing {
    WriterField<uint32_t> head;  // producer: next slot to write (monotonic)
    WriterField<uint32_t> tail;  // consumer: next slot to read (monotonic)
    Msg slots[N];                // slot index = sequence % N
};

// ---------------------------------------------------------------------------
// Segment layout (DESIGN §4.2). One segment per agent, created by the
// primary, mapped by both processes:
//
//   [0]                ControlBlock
//   [desc_offset]      DescriptorHeader[kRingSlots], each followed by its
//                      ids/kind/x arenas
//   [row_offset]       row arena: fp32[kRingSlots][max_entries * H]
//   [ctrl_offset]      MsgRing<CtrlMsg, kControlRingSlots>
//   [ack_offset]       MsgRing<AckMsg, kAckRingSlots>
//   [var_offset]       variable area: blob_bytes[], fill lists, residency
//                      tables, resident bitmaps, error text
//
// All offsets are from the segment base; no raw pointers are stored in the
// segment, so either side may map it at any address (§4.1).
// ---------------------------------------------------------------------------

struct SegmentLayout {
    uint64_t control_offset;
    uint64_t desc_offset;
    uint64_t desc_slot_bytes;  // header + ids + kind + x for one slot
    uint64_t row_offset;
    uint64_t row_slot_bytes;   // max_entries * H * sizeof(float)
    uint64_t ctrl_offset;
    uint64_t ack_offset;
    uint64_t var_offset;
    uint64_t total_bytes;
};

inline constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

// Computes the layout for a session. Pure arithmetic on the geometry offer;
// the primary runs this at creation and publishes Geometry.segment_bytes.
inline SegmentLayout compute_layout(const Geometry& g) {
    const uint64_t ids_bytes = uint64_t(g.max_entries) * sizeof(int32_t);
    const uint64_t kind_bytes = ids_bytes;
    const uint64_t x_bytes =
        uint64_t(g.max_tokens) * uint64_t(g.n_embd) * sizeof(float);

    SegmentLayout s{};
    s.control_offset = 0;
    s.desc_offset = align_up(sizeof(ControlBlock), 64);
    s.desc_slot_bytes =
        align_up(sizeof(DescriptorHeader) + ids_bytes + kind_bytes + x_bytes, 64);
    s.row_slot_bytes = uint64_t(g.max_entries) * uint64_t(g.n_embd) * sizeof(float);
    s.row_offset = s.desc_offset + s.desc_slot_bytes * kRingSlots;
    s.ctrl_offset = align_up(s.row_offset + s.row_slot_bytes * kRingSlots, 64);
    s.ack_offset = align_up(s.ctrl_offset + sizeof(MsgRing<CtrlMsg, kControlRingSlots>), 64);
    s.var_offset = align_up(s.ack_offset + sizeof(MsgRing<AckMsg, kAckRingSlots>), 64);
    s.total_bytes = s.var_offset;  // variable area sized by the creator beyond this
    return s;
}

// Arena accessors: base pointers for one descriptor slot's arrays, given the
// slot's header address. (Layout math only; lifetime/mapping is §4.6's job.)
inline const int32_t* slot_ids(const DescriptorHeader* h) {
    return reinterpret_cast<const int32_t*>(
        reinterpret_cast<const uint8_t*>(h) + h->ids_offset);
}
inline int32_t* slot_ids(DescriptorHeader* h) {
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(h) + h->ids_offset);
}
inline const int32_t* slot_kind(const DescriptorHeader* h) {
    return reinterpret_cast<const int32_t*>(
        reinterpret_cast<const uint8_t*>(h) + h->kind_offset);
}
inline int32_t* slot_kind(DescriptorHeader* h) {
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(h) + h->kind_offset);
}
inline const float* slot_x(const DescriptorHeader* h) {
    return reinterpret_cast<const float*>(
        reinterpret_cast<const uint8_t*>(h) + h->x_offset);
}
inline float* slot_x(DescriptorHeader* h) {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(h) + h->x_offset);
}

}  // namespace wire
}  // namespace franken
