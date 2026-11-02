# FrankenStrata

Cross-vendor expert tiers for Strata: one engine process (the **primary**) gains
expert rows from *other vendors'* GPUs, by dispatching to one small **agent
process** per foreign vendor over POSIX shared memory. Upstream Strata already
implements expert tiers (`--peer-device`, `--expert-cache-device1..3`) — but only
inside one engine binary, and an engine binary drives exactly one vendor's
runtime. FrankenStrata builds the missing cross-vendor edges, reusing upstream's
own row-dispatch contract (the `ExpertSource` seam, verify.cpp's CPU-row path),
so the primary's correctness never depends on any agent being alive.

## Architecture

- **One primary** — a stock engine build (CUDA, HIP, or SYCL) owning the session,
  the dense spine, KV, and all routing. Reference rig: **HIP engine on the
  7900 XT**, because its engine is the complete one.
- **One agent per additional vendor** — a standalone executable
  (`franken-agent-sycl`, `-hip`, `-cuda`) containing an `ExpertCache` and the
  grouped expert kernels from that vendor's engine tree, and nothing else.
- **New row kind 3** — the dispatch partition gains one answer to "who computes
  this row": primary VRAM (0), PCIe share (1), in-process peer (2), CPU rows
  (−1), or **agent rows (3)**. Every agent row is recomputable on the CPU path:
  a slow, absent, or dead agent is a demotion event, never a session failure.
- **Wire protocol** — one shared-memory segment per agent: control block,
  descriptor ring (R=8), a row arena that *is* the primary's `out` buffer, and
  control/ack rings (OPEN fill, PROMOTE/DEMOTE, RESYNC, heartbeats, residency
  generations). No device pointers or runtime types ever cross the wire —
  experts are named only as (layer, expert) into the shared pack.
- **Degradation floor** — with `--franken-agent` absent, the primary is
  byte-identical to stock Strata. Worst case on any rig: stock behavior on the
  primary alone.

## Phases

- **Phase 0** — baselines and de-risking spikes: both stock engines standalone
  (HIP on the XT, SYCL on the B70), tok/s + cache-hit + dispatch-timing tables,
  the cross-engine divergence control, and a minimal agent spike that replays
  recorded inputs against the CPU-row reference.
- **Phase 1** — static shard behind `--franken-agent`, kind 3 in both dispatch
  paths, deadline/salvage/heartbeat machinery, kill drills, §8 correctness gate.
- **Phase 2** — dynamic residency: profile-frequency fill, PROMOTE/DEMOTE over
  the control ring under residency generations, RESYNC drills.
- **Phase 3** — q8_1 wire activations, direct device writes into the row arena,
  cost-model balancing from measured round times.

Validation model (decided): a fits-on-one-card MoE first — the engine's s2
35B-class pack (Qwen3.8-Flash-Next family, 48 × 512 experts) — with both cards
in the loop from the first run; the 120B-class archetype is Phase 1's second
model.

## This repo

Fork of `Niko1221/Strata` at v0.1.53 (`46c152a`), branch **`franken`**. Upstream
files are untouched; all FrankenStrata work lands in new files:

- `franken/` — the wire protocol header and the Phase 0 agent spike
  (see `franken/README.md` for the file → design-section map).
- `BUILD_PROBE.md` — what a CPU-only build of the stock tree does in a
  GPU-less environment, with the exact commands and errors.

## Design

The full design (protocol spec, primary/agent integration, phased plan,
correctness basis, and the closed design questions) lives in Philip's research
notes: `~/workspace/research_notes/strata-multi-gpu-recon/DESIGN.md`, with the
upstream recon it rests on in `REPORT.md` alongside it.
