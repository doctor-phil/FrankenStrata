# franken/ — FrankenStrata additions

Everything FrankenStrata adds lives in this directory as **new files**; no
upstream file is modified (fork hygiene: DESIGN §12 — a new feature is opt-in
and the default path stays byte-identical to upstream). Primary-side patches
(`--franken-agent`, kind 3, `AgentTier`, `AgentExpertSource`) are Phase 1 work
and will land as their own new files plus flagged touch-points listed in
DESIGN §5; they are not here yet.

Design references are to `DESIGN.md` in Philip's research notes
(`~/workspace/research_notes/strata-multi-gpu-recon/DESIGN.md`).

## File map

| File | What it is | Design |
|---|---|---|
| `wire/franken_wire.hpp` | The shared-memory wire protocol as a self-contained C++17 header: segment layout, control block, descriptor/row rings, control + ack messages, handshake and generation discipline. Layout and semantics only — no OS calls. | §4 (all of it; see per-part map below) |
| `agent/agent_spike.cpp` | The Phase 0 agent spike: opens the validation pack with the SYCL ports, fills the SYCL `ExpertCache`, replays recorded (activation, ids) windows, and compares agent rows against the CPU-row reference with the §8.2 statistics. GPU/SYCL bodies are TODO stubs that fail loudly; host-side loader, fill builder, and statistics are real code. Compiles standalone (host C++17). | §7 Phase 0 (spike), §6.2 (reuse inventory), §8.2 (statistics) |

## `wire/franken_wire.hpp` — part-by-part design map

| Header part | Design |
|---|---|
| Magic / protocol version, geometry block, pack identity | §4.2, §4.3 step 1 |
| Vendor-neutrality rules (comment block at top) | §4.1 |
| Caps + HELLO | §4.2 (caps), §4.3 step 2, §5.4 (vendor self-identifies) |
| Control block, cache-line-isolated writer fields | §4.2 |
| `agent_state` (Absent/Loading/Ready/Dead), heartbeat | §4.2, §4.5 |
| `residency_gen`, generation discipline | §4.2, §6.3 |
| `doorbell_seq`, descriptor ring (R = 8), slot arenas | §4.2, §4.4 |
| Slot state word: Empty → Published → Claimed → Done \| Salvaged | §4.2, §4.4, §4.5 |
| Futex-on-the-state-word signaling, spin budgets | §4.4.1 |
| Deadline/salvage constants, heartbeat staleness (250 ms) | §4.5 |
| Row arena = the primary's `out` buffer | §4.2, §4.4 step 3 |
| Control ring: OPEN / PROMOTE / DEMOTE / RESYNC / BYE | §4.2, §4.3 step 3, §6.3, §4.5.1 |
| Ack ring: HELLO / OPEN_ACK / PROMOTE_ACK / DEMOTE_ACK / ERROR | §4.2, §4.3 steps 2+4, §6.3 |
| `PayloadKind` (expert rows only in v1, space kept open) | §13 Q6 (non-preclusion) |
| Session caps inherited, not hardcoded (MAXT = 8, 128 entries) | §3.1 |
| OS primitives named, never used (segment, doorbell word, clock) | §4.6, §13 Q4 |

## Phase 0 status

- Wire header: compiles standalone (`g++ -std=c++17`); layout math verified
  against the §3.1 s2 geometry (48 layers, H = 2560, 128 entries →
  1,310,720 B row slots).
- Spike: compiles standalone; with no arguments it prints usage and exits 2;
  any run stops at the first unlinked SYCL TODO with a named error — it
  cannot fabricate numbers.
- Next: link the spike against the SYCL tree on the B70 (the runbook's job),
  grow the primary-side `--franken-agent` scaffolding (Phase 1, DESIGN §5).
