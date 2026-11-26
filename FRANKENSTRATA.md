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
python3 terminal/strata.py franken --plan
```

Run mixed-hardware generation (auto-discovers AMD + Intel GPUs, spawns and supervises agents, and guarantees process-group and `/dev/shm` cleanup on exit):

```sh
python3 franken/launcher/franken_launch.py --profile -- \
  --pack /work/Strata-data/packs/coder-iq1_m \
  --native /work/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --mtp /work/Strata-data/mtp/rt \
  --expert-profile terminal/data/expert-profile-coder-2026-11-04.bin \
  --expert-cache auto --prefill 512 \
  --spec 4 --spec-min-p 0.5 --max-context 8192 \
  --tokens-file engine/sycl/tools/data/prompt_deadlock_2744.tok --max-new 64
```

Pass `--docker-sycl` if your host does not have a native Intel Level-Zero ICD and you want the launcher to run `franken-agent-sycl` inside `strata-sycl-dev:latest`.

### Boxed Self-Contained Linux Bundle

Build and stage a standalone directory containing `hip/` (TheRock ROCm runtime libs), `sycl/` (oneAPI SYCL + Level-Zero UR runtime libs), `bin/` (`franken-agent-cpu`, `franken-profile`), `launcher/`, `franken-strata`, and `BUILD.json`:

```sh
tools/packaging/build_franken_linux.sh
# or package existing build-hip-franken/ and build-sycl-franken/:
SKIP_BUILD=1 OUT=dist/frankenstrata-linux-x86_64 tools/packaging/build_franken_linux.sh
```

Every ELF binary and bundled `.so` carries `RUNPATH $ORIGIN` and passes `tools/packaging/binary_deps.py`. End users only need standard Linux kernel GPU drivers (`amdgpu` + Intel `xe`/`i915` + `libze1`); no ROCm install and no Intel oneAPI SDK install are required.

```sh
env -i PATH=/usr/bin:/bin ./dist/frankenstrata-linux-x86_64/franken-strata --plan
```

### Opt-in `serve` Integration

Add a `"franken"` block to a Strata run config JSON to supervise mixed-vendor agents under `terminal/serve/server.py`:

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

Measured on one rig: **AMD Ryzen 9 5900XT (16C/32T, 64 GB DDR4-3200)**, **AMD RX 7900 XTX 24 GB (`gfx1100`, PCIe Gen4 x8)**, and **Intel Arc Pro B70 32 GB (`bmg-g31`, PCIe Gen3 x8)** on Ubuntu 26.04. Model: **Qwen3.8-Flash-Next Coder IQ1_M** (`48` layers, `256` experts/layer = `12,288` total experts, `k = 10`, `H = 2560`, MTP drafter). Protocol: greedy, `--spec 4 --spec-min-p 0.5 --max-context 8192 --prefill 512`, `engine/sycl/tools/data/prompt_deadlock_2744.tok` (2,743 tokens), 64 generated tokens, built on **Strata `v0.2.22` (engine `0.1.69`)** (`build-hip-franken` + `build-sycl-franken` / boxed bare-host `franken-agent-sycl`).

<p align="center">
  <img src="franken/docs/decode-benchmark.svg" width="900" alt="FrankenStrata decode throughput on Qwen3.8-Flash-Next Coder IQ1_M (Strata v0.2.22 / engine 0.1.69), sorted descending by decode tok/s">
</p>

### End-to-End Decode & Parity on Strata `v0.2.22` / Engine `0.1.69` (`prompt_deadlock_2744.tok`, 64 generated tokens)

| Configuration (`v0.2.22` / engine `0.1.69`) | Decode tok/s | Prefill tok/s | Salvages | Agent `launch / wait / copy` (ms/rd) | Token parity vs solo |
| --- | ---: | ---: | ---: | --- | --- |
| Solo Intel Arc Pro B70 32 GB (SYCL AOT, `12,288/12,288` resident) | **77.03** | 702.46 | — | — | **64/64 exact** |
| Mixed 7900 XTX + B70 (`auto` + `3,200` slots, cost-model on + `franken-profile`) | **76.57 (+2.6%)** | 938.96 | **0** | `0.003 / 0.283 / 0.002` | **64/64 exact*** |
| Mixed 7900 XTX + B70 (`auto` + `3,200` slots, static / cost-model off) | **76.21 (+2.2%)** | 964.57 | **0** | `0.132 / 1.834 / 0.021` | **64/64 exact*** |
| Solo AMD RX 7900 XTX 24 GB (HIP `gfx1100`, `auto` = `9,121/12,288` slots) | **74.60** | 965.05 | — | — | XTX auto baseline (64/64) |
| **Mixed 7900 XTX + B70 (cap `5,000` + `3,000`, 3-tier dynamic + cost-model on)** | **72.20 (+9.5%)** | 729.23 | **0** | `0.079 / 0.698 / 0.039` | **64/64 exact** (`27/27` async promotes, `146` demotes) |
| Mixed 7900 XTX + B70 (cap `5,000` + `3,000`, static / cost-model off) | **70.86 (+7.5%)** | 738.17 | **0** | `0.221 / 1.494 / 0.058` | **64/64 exact** (`1,130/1,130` rounds) |
| Solo AMD RX 7900 XTX 24 GB (HIP `gfx1100`, `--expert-cache 5000`) | **65.94** | 715.50 | — | — | XTX 5k baseline (64/64) |
| **Reverse Mixed B70 + 7900 XTX HIP (cap `5,000` + `3,000`, 3-tier dynamic + CM on)** | **59.09 (+7.4%)** | 273.49 | 1 | `0.005 / 2.257 / 0.002` | **64/64 exact** vs B70 5k† |
| Solo Intel Arc Pro B70 32 GB (SYCL AOT, `--expert-cache 5000` → 6,519 slots) | **55.00** | 273.08 | — | — | B70 5k baseline (64/64) |
| Reverse Mixed B70 + 7900 XTX HIP (cap `5,000` + `3,000`, static / CM off) | **52.71 (-4.2%)** | 273.61 | 2 | `0.232 / 9.197 / 0.074` | **64/64 exact** vs B70 5k (`1,066/1,068` rounds)† |

*\*On `v0.2.22`, `gfx11` enables dynamic `STRATA_PREFILL_CPU_SHARE` by default, which drifted between share `0.82` and `0.84` on the two `auto` mixed runs (59/64, first diff at token 59). With `STRATA_PREFILL_CPU_SHARE=0`, both `auto` static and `auto` cost-model-on match solo **64/64** identically.*
*†Reverse direction runs SYCL AOT `strata` inside `strata-sycl-dev:latest` paired with host `franken-agent-hip` over `/dev/shm`. On round 0, cold HIP code-object loading (~60 ms) tripped 1–2 deadline salvages (warming `franken-agent-hip` on `OPEN` is a planned follow-up); the dynamic run used the same `--adapt-every 4 --adapt-swaps 32 --franken-adapt-swaps 16 --franken-cost-model on --franken-profile` flags as the forward capped dynamic run (with default `--adapt-swaps` and `--franken-deadline-ms 120`, reverse dynamic ran `54.44 tok/s`, 0 salvages, 64/64 exact).*

Key takeaways on `v0.2.22` (`0.1.69`):

- **Full VRAM (`auto`):** Because `Coder IQ1_M` fits 100% inside the 32 GB B70 (`12,288/12,288` slots), Solo B70 reaches `77.03 tok/s` decode (`702.46 tok/s` prefill). Pairing the 24 GB 7900 XTX (`9,121/12,288` slots) with the B70 agent brings decode from `74.60 -> 76.57 tok/s` (+2.6%) while keeping the 7900 XTX's `939–965 tok/s` prefill (+34% faster prefill than Solo B70).
- **Under VRAM pressure (models larger than any single GPU, simulated via 5,000-slot primary cap + 3,000-slot B70 tier):** 3-tier dynamic swapping with async copy-engine `PROMOTE` and move-cost gating beats both Solo 7900 XTX (`65.94 -> 72.20 tok/s`, **+9.5%**) and static placement (`70.86 -> 72.20 tok/s`) with **0 salvages** and **64/64 exact token parity**.

### Hardware Profiler (`franken-profile`, `v0.2.22`)

- **Host CPU (145.3 MiB cold-DRAM working set, 70 distinct blobs):** `T_cpu(n) = 0.054 + 0.047 * n ms` (`n=1: 0.10 ms`, `n=2: 0.15 ms`, `n=4: 0.24 ms`, `n=8: 0.43 ms`).
- **Arc Pro B70 (boxed bare-host `franken-agent-sycl` over `/dev/shm`, `q8_1=1`, `compact=1`):** `T_b70(n) = 0.029 + 0.010 * n ms` (USM staging coherency: **PASS**; async refill queue: **Level Zero copy engine**).
- **RX 7900 XTX (`franken-agent-hip` over `/dev/shm`, `q8_1=1`, `compact=1`, Phase 3 smoke test):** `T_hip(n) = 0.086 + 0.006 * n ms` (`100/100` rounds, `620` entries, pinned coherency: **PASS**).

### Phase 0–3 Conformance, Kill Drill & Accuracy Verification

- **Phase 0 SYCL Spike on B70 (1,200 real captured Coder IQ1_M windows):** 12,288/12,288 slots filled in **17.37 s**; row divergence vs CPU reference (560 sampled entries): `rel_l2` median **`1.411e-02`** (p99 `1.802e-02`), `max_abs/rms` median **`5.336e-02`**, cosine median **`0.9999`**; B70 median **`0.315 ms`** vs CPU **`11.61 ms`** (~37x faster per window).
- **Mid-generation `SIGKILL` drill (Phase 1–2):** killing `franken-agent-sycl` mid-generation triggered `3` salvages → `Dead` transition, seamless host CPU fallback, **0 hangs**, **0 NaNs**, and **64/64 exact token parity**.
- **CPU conformance & bare-host boxed bundle (`p128.tok`):** `HIP + franken-agent-cpu` matched **64/64** tokens; the `v0.2.22` boxed bundle (**784 MiB** total, `strata 0.1.69`, `env -i`, no Docker, no system ROCm or oneAPI) passed `tools/packaging/binary_deps.py` on `hip/`, `sycl/`, and `bin/` and drove all four `v0.2.22` mixed benchmarks directly on bare host with zero `/dev/shm` leaks.

## What Was Tested vs Not Tested

- **Tested:**
  - C++ Release unit tests on `v0.2.22` (`0.1.69`): `franken_ipc_test` (5/5), `franken_phase2_test` (5/5), `franken_phase3_test` (3/3).
  - Python launcher unit tests: `pytest terminal/tests/terminal/test_franken_launcher.py` (4/4) and `terminal/strata.py franken --plan`.
  - Boxed Linux bundle (`tools/packaging/package_franken.py`): `tools/packaging/binary_deps.py` ELF closure on `hip/`, `sycl/`, `bin/`, `env -i` launch check (`strata 0.1.69`), and bare-host SYCL agent execution across all four `v0.2.22` mixed runs.
  - Real hardware runs on RX 7900 XTX (HIP) + Arc Pro B70 (SYCL) + Ryzen 9 5900XT (CPU) with `Qwen3.8-Flash-Next Coder IQ1_M`: Solo B70, Solo 7900 XTX (auto + cap 5,000), Mixed 7900 XTX + B70 (auto static, auto CM-on, cap 5,000+3,000 static, cap 5,000+3,000 3-tier dynamic), `franken-profile`, Phase 0 SYCL spike, CPU conformance, `franken-agent-hip` smoke test, and mid-generation `SIGKILL` drill.
- **Not tested:**
  - `franken-agent-cuda` on NVIDIA hardware (compiles under `STRATA_ENABLE_CUDA`; this rig has no NVIDIA GPU).
  - Canonical `s2` packs and `Qwen3.6-35B-A3B` end-to-end mixed-GPU runs.
  - Windows and macOS (the POSIX `/dev/shm` + Linux futex IPC layer is Linux-only).

## Files & Design

- `franken/` — wire protocol, IPC, CPU/SYCL/HIP/CUDA agents, `AgentTier`, hardware profiler, `TierController`, launcher, decode chart SVG, and unit tests (see [franken/README.md](franken/README.md)).
- `tools/packaging/build_franken_linux.sh` and `tools/packaging/package_franken.py` — self-contained Linux bundle builder and ELF dependency verifier.
- Full design notes: `~/workspace/research_notes/strata-multi-gpu-recon/DESIGN.md` and `REPORT.md`.
