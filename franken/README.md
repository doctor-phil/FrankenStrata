# franken/ — FrankenStrata additions

FrankenStrata adds cross-vendor multi-GPU expert offloading over POSIX shared
memory (`/dev/shm`) and non-private Linux futexes. New code lives in `franken/`;
upstream touch-points are opt-in behind `--franken-agent` (when omitted, default
Strata behavior is unchanged).

Design references are to `DESIGN.md` in Philip's research notes
(`~/workspace/research_notes/strata-multi-gpu-recon/DESIGN.md`).

## File map

| File | What it is |
|---|---|
| `wire/franken_wire.hpp` | Pure C++17 wire layout spec: control block, descriptor/row rings, control/ack SPSC rings, slot state machine (`Empty -> Published -> Claimed -> Done \| Salvaged`), constants. |
| `wire/franken_ipc.{hpp,cpp}` | POSIX `shm_open`/`ftruncate`/`mmap`/`shm_unlink` lifecycle, non-private cross-process `FUTEX_WAIT`/`FUTEX_WAKE`, `CLOCK_MONOTONIC` helpers, bounded spin-before-park waits, SPSC ring helpers, deterministic variable-area plan, heartbeat monitor. |
| `wire/pack_identity.hpp` | Deterministic FNV-1a `PackIdentity` computation shared by primary and agents across host/container mounts. |
| `tests/franken_ipc_test.cpp` | GPU-less unit tests: layout math, SPSC ring wraparound, `Salvaged` race guard, 250 ms heartbeat staleness + `SIGKILL`, cross-process handshake & 36-round descriptor ring wraparound. |
| `agent/agent_spike.cpp` | Phase 0 SYCL extraction spike (`franken-agent-spike` in `sycl/CMakeLists.txt`): loads pack/GGUF + SYCL `ExpertCache`, replays `FRKW` windows (or `--random-windows`), reports SYCL-vs-CPU `rel_l2`, `max_abs/rms`, `cosine`, and compute latency. |
| `agent/agent_loop.{hpp,cpp}` | Vendor-neutral agent runtime: attaches shm, verifies `PackIdentity` + `Geometry`, posts `HELLO` + `Caps`, runs 10 ms heartbeat thread, handles `OPEN`/`BYE` (+ Phase 2 `PROMOTE`/`DEMOTE`/`RESYNC` hooks), spins/parks on doorbell, verifies `SlotState::Claimed` before writing rows. |
| `agent/franken_agent_cpu.cpp` | Vendor-free CPU conformance/CI agent (`franken-agent-cpu`, zero GPU dependencies; supports `OPEN`, `PROMOTE`, `DEMOTE`, `RESYNC`, `BYE`). |
| `agent/franken_agent_sycl.cpp` | Intel Arc SYCL agent (`franken-agent-sycl`): native IQ (with Hadamard) + canonical s2 packs, USM host staging + coherency self-test, pre-quantized `q8_1` wire activations, compact router-order row return, and non-blocking async `PROMOTE` on Level Zero copy engine. |
| `agent/franken_agent_gpu.cpp` | AMD HIP (`franken-agent-hip`) / NVIDIA CUDA (`franken-agent-cuda`) foreign-vendor agent mirroring `franken-agent-sycl` (pinned staging + coherency self-test, `q8_1` wire activations, compact rows, async `PROMOTE` on a dedicated non-blocking refill stream). |
| `primary/agent_tier.{hpp,cpp}` | Primary-side `AgentTier` managing 1..N foreign-vendor agents (`--franken-agent <shm>[,slots=N]`, `--franken-deadline-ms`), capability negotiation (`q8_1` AVX2+F16C host quantizer + compact rows), disjoint profile/layer-early assignment at `OPEN`, kind-3 dispatch overlap, `PROMOTE`/`DEMOTE`/`RESYNC` control + `C_move_ms` tracking, deadline salvage -> CPU recompute fallback, and 250 ms heartbeat staleness -> `Dead`. |
| `profiler/hw_profile.{hpp,cpp}` | Hardware profile JSON schema (`franken-hw-profile.json`), serializer/parser, linear latency fit, and deterministic `recommend_tier_plan`. |
| `profiler/franken_profile.cpp` | Optional hardware profiler CLI (`franken-profile`): benchmarks host CPU/RAM over a `>= 128 MiB` cold-DRAM working set + attached `/dev/shm` agents (`n = 1, 2, 4, 8, 16` + H2D fill bandwidth), writes `franken-hw-profile.json`, and prints recommended launch flags. |
| `control/tier_controller.{hpp,cpp}` | Dynamic multi-tier controller: distinct-expert online latency cost model (`--franken-cost-model on\|off`, seeded via `--franken-profile`), deduplication `DEMOTE`, Primary-evicted cascade into Agent VRAM, and move-cost-gated (`(h_in-h_out)*H_eff*(w_src-w_m) > kappa*C_move_ms`) hot CPU-miss swaps (`--franken-adapt-swaps N`). |
| `tests/franken_phase2_test.cpp` | GPU-less Phase 2 unit tests: `HwProfile` JSON round-trip, `recommend_tier_plan` ordering, live cross-process `PROMOTE`/`DEMOTE`/`RESYNC` over shm, cascade swap planner, and online cost-model gating. |
| `tests/franken_phase3_test.cpp` | GPU-less Phase 3 unit tests: `q8_1` host quantizer (accuracy, zero block, `kHalfMax` clamp), `q8_1` + compact row negotiation & FP32/v1 fallback, non-blocking async `PROMOTE` concurrency + `C_move_ms`, and economic move-cost break-even gate. |
| `launcher/franken_launch.py` | Phase 4 unified mixed-hardware launcher & `AgentSupervisor`: Linux AMD/Intel/NVIDIA GPU discovery, `franken-hw-profile.json` / `--profile` / `--reprofile`, native bundled or Docker SYCL agent supervision, and guaranteed process-group + `/dev/shm` cleanup (`strata franken ...` / `franken-strata`). |
| `../packaging/package_franken.py`, `../packaging/build_franken_linux.sh` | Phase 4 boxed Linux packager staging self-contained `hip/` (TheRock), `sycl/` (oneAPI/Level-Zero UR), `bin/` (`franken-agent-cpu`, `franken-profile`), `launcher/`, `franken-strata`, and `BUILD.json` with `RUNPATH $ORIGIN` and `binary_deps.py` verification. |
| `../tests/terminal/test_franken_launcher.py` | Phase 4 `pytest` unit tests covering GPU discovery, tier plan selection, native/Docker command assembly, and supervisor teardown + `/dev/shm` cleanup. |

See **[FRANKENSTRATA.md](../FRANKENSTRATA.md)** for full architecture, quick start, and measured benchmark tables.

## Build & verification commands (run on hardware rig)

1. **GPU-less IPC unit tests:**
   ```sh
   g++ -std=c++20 -O2 -Wall -Wextra -I. franken/wire/franken_ipc.cpp franken/tests/franken_ipc_test.cpp -pthread -lrt -o build-ipc-test
   ./build-ipc-test
   ```
2. **SYCL spike + SYCL agent (Docker `strata-sycl-dev:latest`, Arc Pro B70):**
   ```sh
   docker run --rm -it --device /dev/dri \
     -v "$PWD:/src" \
     -v /home/phil/github/frankenstrata/build-sycl-aot/_deps/strata_llamacpp-src:/ggml:ro \
     -v /work/Strata-data:/work/Strata-data:ro \
     -v /dev/shm:/dev/shm \
     strata-sycl-dev:latest bash -lc '
       cmake -S /src/sycl -B /src/build-sycl-franken -G Ninja \
         -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT=bmg-g31 -DSTRATA_GGML_DIR=/ggml
       cmake --build /src/build-sycl-franken --target franken-agent-spike franken-agent-sycl
     '
   ```
3. **HIP primary + CPU agent (host, RX 7900 XTX `gfx1100` using TheRock ROCm wheels):**
   ```sh
   cmake -S . -B build-hip-franken -G Ninja \
     -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
     -DSTRATA_GGML_DIR=/home/phil/github/frankenstrata/build-sycl-aot/_deps/strata_llamacpp-src \
     -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" -DCMAKE_HIP_COMPILER="$ROCM_PREFIX/lib/llvm/bin/clang++"
   cmake --build build-hip-franken --target strata franken_ipc_test franken-agent-cpu
   ./build-hip-franken/franken_ipc_test
   ```
4. **7900 XTX (HIP primary) + B70 (SYCL agent) on Coder IQ1_M + SIGKILL drill:**
   - Terminal 1 (Docker B70):
     ```sh
     docker run --rm --device /dev/dri -v "$PWD:/src" -v /work/Strata-data:/work/Strata-data:ro -v /dev/shm:/dev/shm \
       strata-sycl-dev:latest /src/build-sycl-franken/franken-agent-sycl \
       --shm /franken0 --pack /work/Strata-data/packs/coder-iq1_m \
       --native /work/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
     ```
   - Terminal 2 (Host 7900 XTX):
     ```sh
     ./build-hip-franken/strata \
       --pack /work/Strata-data/packs/coder-iq1_m \
       --native /work/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
       --mtp /work/Strata-data/mtp/rt \
       --expert-profile data/expert-profile-coder-2026-11-04.bin \
       --expert-cache auto --spec 4 --spec-min-p 0.5 --max-context 8192 \
       --tokens-file sycl/tools/data/prompt_deadlock_2744.tok --max-tokens 64 --temp 0 \
       --franken-agent /franken0
     ```
   - Kill drill: `kill -9 <franken-agent-sycl PID>` mid-generation; primary must salvage within deadline / 250 ms heartbeat staleness and finish cleanly on CPU with zero hangs or NaNs.
