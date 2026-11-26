# FrankenStrata

Cross-vendor expert tiers for Strata: one primary engine process gains expert rows from other vendors' GPUs by dispatching to one standalone agent process per foreign vendor over POSIX shared memory (`/dev/shm`) and non-private Linux futexes.

Built and verified on a Linux rig combining an **AMD Radeon RX 7900 XTX (24 GB, `gfx1100`, HIP)** and an **Intel Arc Pro B70 (32 GB, `bmg-g31`, SYCL)** with an **AMD Ryzen 9 5900XT (16C/32T, 64 GB DDR4)**. When `--franken-agent` is omitted, Strata runs its stock single-vendor path unchanged.

## Architecture

- **One primary engine (`strata`, HIP or CUDA)** — owns the session, dense weights, KV cache, MTP drafter, routing, and Tier-0 hot experts in VRAM.
- **One agent process per additional vendor** — standalone executables (`franken-agent-sycl`, `franken-agent-hip`, `franken-agent-cuda`, and vendor-free `franken-agent-cpu` for conformance/CI) containing only that vendor's `ExpertCache` and grouped expert kernels. Because each vendor runtime lives in its own OS process, HIP, SYCL, and CUDA libraries never share an address space.
- **Row kind `3` concurrent overlap** — per-layer dispatch partitions routed experts into primary VRAM (`0`), PCIe share (`1`), in-process peer (`2`), foreign-vendor agent (`3`), and host CPU (`-1`). Agent-resident experts are excluded from `nmiss` so the PCIe share never steals them. Immediately after publishing `GpuPlanSink`, the primary rings each agent's `/dev/shm` doorbell (`FUTEX_WAKE`) so primary VRAM, foreign GPU agent(s), and host CPU overlap in parallel.
- **Wire protocol (`franken/wire/franken_wire.hpp`, `franken_ipc.{hpp,cpp}`)** — one POSIX `/dev/shm` segment per agent holding the control block, descriptor ring (`R = 8`), row arena, SPSC control/ack rings (`OPEN`, `PROMOTE`, `DEMOTE`, `RESYNC`, `BYE`), `PackIdentity` verification, and a 10 ms heartbeat counter. Experts are named only as `(layer, expert)` into the shared pack; no device pointers or vendor types ever cross the wire.
- **Phase 3 wire optimizations** — backward-compatible capability bits (`kCapQ8_1Activations`, `kCapCompactRows`) let the primary pre-quantize activations on the host with AVX2+F16C into canonical `block_q8_1` (36 B per 32 floats, 3.56x smaller than FP32) and upload straight to device `d_q8_` on native IQ layers without gate/up Hadamard rotation, while agents pack only `kind == 3` rows contiguously at the front of the row arena. Legacy v1 peers and CPU agents fall back automatically to FP32 and sparse rows.
- **`Salvaged` CPU safety floor** — every agent row is recomputable on the host CPU pool. If an agent misses a round deadline or its heartbeat goes stale for 250 ms, the primary atomically transitions the slot word to `SlotState::Salvaged`, recomputes those rows immediately on CPU, and marks the agent `Dead` after repeated salvages or death. A crashed or `SIGKILL`ed agent never hangs or corrupts generation.

## Optional Hardware Profiler (`franken-profile`)

`franken-profile` measures your rig once and writes `franken-hw-profile.json`:

- **Host CPU/RAM:** Benchmarks `strata_kernels_cpu` across a cold-DRAM working set larger than L3 cache (`>= 128 MiB`, 70+ distinct expert blobs) for batch sizes `1, 2, 4, 8`.
- **Foreign agents:** Connects to each agent over `/dev/shm`, times a 64-blob fill for H2D transfer bandwidth (`pcie_h2d_gbps`), and measures batch sizes `1, 2, 4, 8, 16` to fit `T_agent(n) = a_ipc + b * n`.
- **Tier plan (`recommend_tier_plan`):** Picks the fastest primary GPU, ranks agent tiers, computes slot budgets and minimum-entry thresholds, and prints recommended CLI flags.

```sh
./build-hip-franken/franken-profile \
  --pack /work/Strata-data/packs/coder-iq1_m \
  --native /work/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --agent /franken_prof0 \
  --out franken-hw-profile.json
```

## Dynamic Multi-Tier Controller (`TierController`)

Active whenever `--franken-agent` is set:

- **Fast path (per layer, `--franken-cost-model on|off`):** Continuously fits per-agent `T_agent(n)` and `T_cpu(n)` by exponentially weighted least squares keyed on distinct experts (seeded from `--franken-profile` when supplied). If routing a small miss count to an agent would increase `max(T_primary, T_agent, T_cpu)` relative to absorbing it on CPU, those entries stay on CPU for that layer.
- **Slow path (every `--adapt-every` rounds, `--franken-adapt-swaps N`):**
  - Deduplicates immediately with `DEMOTE` whenever the primary adopts an expert held by an agent.
  - Cascades primary-evicted experts into Tier 1..N agents when they beat the agent's coldest resident experts.
  - Promotes hot CPU misses into agent VRAM when both hysteresis and the economic move-cost break-even gate hold: `(h_in - h_out) * H_eff * (w_src - w_m) > kappa * C_move_ms`.
  - Runs `PROMOTE` reads and H2D copies asynchronously on a background refill worker and separate copy queue (Level Zero copy engine queue group 1 on Intel Battlemage; non-blocking refill stream on HIP/CUDA), publishing `PROMOTE_ACK` only after the copy completes so compute rounds never stall.

## Quick Start

### Unified launcher (`franken-strata` / `strata franken`)

Inspect discovered GPUs and the planned tiers:

```sh
python3 franken/launcher/franken_launch.py --plan
# or through the Strata terminal CLI:
python3 strata.py franken --plan
```

Run mixed-hardware generation (auto-discovers AMD + Intel GPUs, spawns and supervises agents, and guarantees process-group and `/dev/shm` cleanup on exit):

```sh
python3 franken/launcher/franken_launch.py --profile -- \
  --pack /work/Strata-data/packs/coder-iq1_m \
  --native /work/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --mtp /work/Strata-data/mtp/rt \
  --expert-profile data/expert-profile-coder-2026-11-04.bin \
  --expert-cache auto --resident-experts \
  --spec 4 --spec-min-p 0.5 --max-context 8192 --prefill auto \
  --tokens-file sycl/tools/data/prompt_deadlock_2744.tok --max-new 64 --greedy
```

Pass `--docker-sycl` if your host does not have a native Intel Level-Zero ICD and you want the launcher to run `franken-agent-sycl` inside `strata-sycl-dev:latest`.

### Boxed Self-Contained Linux Bundle

Build and stage a standalone directory containing `hip/` (TheRock ROCm runtime libs), `sycl/` (oneAPI SYCL + Level-Zero UR runtime libs), `bin/` (`franken-agent-cpu`, `franken-profile`), `launcher/`, `franken-strata`, and `BUILD.json`:

```sh
LLAMA_DIR=/path/to/llama.cpp packaging/build_franken_linux.sh
# or package existing build-hip-franken/ and build-sycl-franken/:
SKIP_BUILD=1 OUT=dist/frankenstrata-linux-x86_64 packaging/build_franken_linux.sh
```

Every ELF binary and bundled `.so` carries `RUNPATH $ORIGIN` and passes `packaging/binary_deps.py`. End users only need standard Linux kernel GPU drivers (`amdgpu` + Intel `xe`/`i915` + `libze1`); no ROCm install and no Intel oneAPI SDK install are required.

```sh
env -i PATH=/usr/bin:/bin ./dist/frankenstrata-linux-x86_64/franken-strata --plan
```

### Opt-in `serve` Integration

Add a `"franken"` block to a Strata run config JSON to supervise mixed-vendor agents under `serve/server.py`:

```json
{
  "franken": {
    "enabled": true,
    "hw_profile": "franken-hw-profile.json",
    "cost_model": "on",
    "adapt_swaps": 16,
    "agents": [
      { "vendor": "sycl", "device": 0, "shm": "/franken_serve0" }
    ]
  }
}
```

## Measured Results

Measured on one rig: **AMD Ryzen 9 5900XT (16C/32T, 64 GB DDR4-3200)**, **AMD RX 7900 XTX 24 GB (`gfx1100`, PCIe Gen4 x8)**, and **Intel Arc Pro B70 32 GB (`bmg-g31`, PCIe Gen3 x8)** on Ubuntu 26.04. Model: **Qwen3.8-Flash-Next Coder IQ1_M** (`48` layers, `256` experts/layer = `12,288` total experts, `k = 10`, `H = 2560`, MTP drafter). Protocol: greedy, `--spec 4 --spec-min-p 0.5 --max-context 8192 --prefill auto --resident-experts`, `prompt_deadlock_2744.tok` (2,743 tokens) and `p128.tok` (128 tokens), 64 generated tokens, built from this branch's engine `0.1.55` source (`build-hip-franken` + `build-sycl-franken`).

### Phase 0 SYCL Spike on B70 (1,200 real captured Coder IQ1_M windows)

- Cache fill: 12,288 / 12,288 slots in **17.37 s** (`38,880 / 38,880` entries resident).
- Row divergence vs CPU reference (560 sampled entries): `rel_l2` median **`1.411e-02`** (p99 `1.802e-02`), `max_abs/rms` median **`5.336e-02`**, cosine median **`0.9999`** (p99 `1.0000`).
- Per-window compute latency: B70 median **`0.315 ms`** (p99 `0.678 ms`) vs CPU median **`11.61 ms`** (p99 `43.36 ms`) — about **37x faster per window**.

### Hardware Profiler (`franken-profile`)

- **Host CPU (145.3 MiB cold-DRAM working set, 70 distinct blobs):** `T_cpu(n) = 0.021 + 0.058 * n ms` (`n=1: 0.09 ms`, `n=2: 0.14 ms`, `n=4: 0.23 ms`, `n=8: 0.50 ms`).
- **Arc Pro B70 (`franken-agent-sycl` over `/dev/shm`, `q8_1=1`, `compact=1`):** `T_b70(n) = 0.033 + 0.012 * n ms` (USM staging coherency: **PASS**; async refill queue: **Level Zero copy engine**).
- **RX 7900 XTX (`franken-agent-hip` over `/dev/shm`, `q8_1=1`, `compact=1`):** `T_hip(n) = 0.086 + 0.006 * n ms` (`100/100` rounds, `620` entries, pinned coherency: **PASS**).

### End-to-End Decode & Parity (`prompt_deadlock_2744.tok`, 64 generated tokens)

| Configuration | Decode tok/s | Prefill tok/s | Salvages | Agent `launch / wait / copy` (ms/rd) | Token parity vs solo HIP |
| --- | ---: | ---: | ---: | --- | --- |
| Solo HIP 7900 XTX (`auto` = 9,159 slots) | 59.29 – 60.79 | 680.4 – 682.3 | — | — | baseline (64/64) |
| 7900 XTX + B70 static (`CM off`, Phase 1–2) | 62.80 | 677.0 | 0 | `0.121 / 1.708 / 0.028` | **64/64 exact** |
| 7900 XTX + B70 (`CM on` + profile, Phase 2) | **64.65 (+9.0%)** | 677.9 | 0 | `0.011 / 0.537 / 0.010` | **64/64 exact** |
| 7900 XTX + B70 (`CM on` + cold profile, Phase 3 AVX2 `q8_1`) | 63.58 – 64.23 | 673.3 – 674.5 | 0 | `0.002 / 0.305 / 0.002` | **64/64 exact** |
| 7900 XTX + B70 **mid-generation `SIGKILL` drill** | 61.70 | — | 3 → `Dead` | `0.062 / 1.251 / 0.013` | **64/64 exact** (0 hangs, 0 NaNs) |
| Solo HIP 7900 XTX (capped `5,000` slots, ~41% VRAM) | 48.86 | 674.9 | — | — | baseline (64/64) |
| Capped `5,000` + B70 `3,000` static (`CM off`, Phase 1–2) | **56.81 (+16.3%)** | 674.8 | 0 | `0.165 / 1.691 / 0.067` | **64/64 exact** |
| Capped `5,000` + B70 `3,000` 3-tier dynamic (Phase 2 sync) | 53.37 (+9.2%) | 673.7 | 2 | `0.165 / 1.458 / 0.071` | **64/64 exact** (`103/103` promotes) |
| **Capped `5,000` + B70 `3,000` 3-tier dynamic (Phase 3 async)** | **56.09 (+14.8%)** | 669.7 | **0** | `0.180 / 1.161 / 0.062` | **64/64 exact** (`80` async promotes, `c_move=0.649 ms`, `68` move-rejected) |

On `p128.tok`, solo HIP ran at **68.34 tok/s**, HIP + `franken-agent-cpu` matched **64/64** tokens at **66.67 tok/s**, HIP + B70 SYCL matched **64/64** tokens at **71.82 tok/s (+5.1%)**, and the bare-host boxed bundle (`env -i ./franken-strata`, 781 MiB total, no Docker, no system ROCm or oneAPI) auto-discovered both cards, matched **64/64** tokens, and left zero `/dev/shm` files or orphan processes on exit.

*(This branch builds upstream engine `0.1.55`. The Phase 0 README's `79.05` / `1655` solo HIP row came from the newer released `0.1.69` `rx7900` package; every mixed-hardware delta above is measured against the same `0.1.55` HIP source build.)*

## What Was Tested vs Not Tested

- **Tested:**
  - C++ Release unit tests: `franken_ipc_test` (5/5), `franken_phase2_test` (5/5), `franken_phase3_test` (3/3).
  - Python launcher unit tests: `pytest tests/terminal/test_franken_launcher.py` (4/4).
  - Boxed Linux bundle (`packaging/package_franken.py`): `binary_deps.py` ELF closure on `hip/`, `sycl/`, `bin/`, plus bare-host `env -i` execution on 7900 XTX + B70.
  - Real hardware runs on RX 7900 XTX (HIP) + Arc Pro B70 (SYCL) + Ryzen 9 5900XT (CPU) with `Qwen3.8-Flash-Next Coder IQ1_M`: solo baseline, SYCL spike, CPU conformance, HIP+SYCL parity, mid-generation `SIGKILL` drill, `franken-agent-hip` smoke test, `franken-profile`, cost-model gating, and 3-tier async `PROMOTE`/`DEMOTE`.
- **Not tested:**
  - `franken-agent-cuda` on NVIDIA hardware (compiles under `STRATA_ENABLE_CUDA`; this rig has no NVIDIA GPU).
  - Canonical `s2` packs and `Qwen3.6-35B-A3B` end-to-end mixed-GPU runs.
  - Windows and macOS (the POSIX `/dev/shm` + Linux futex IPC layer is Linux-only).

## Files & Design

- `franken/` — wire protocol, IPC, CPU/SYCL/HIP/CUDA agents, `AgentTier`, hardware profiler, `TierController`, launcher, and unit tests (see [franken/README.md](franken/README.md)).
- `packaging/build_franken_linux.sh` and `packaging/package_franken.py` — self-contained Linux bundle builder and ELF dependency verifier.
- Full design notes: `~/workspace/research_notes/strata-multi-gpu-recon/DESIGN.md` and `REPORT.md`.
