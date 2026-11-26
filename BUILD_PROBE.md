# Build probe — stock Strata tree, CPU-only, GPU-less VM

**What this is:** a time-capped attempt to configure and build the stock engine
from this fork (branch `franken`, HEAD `46c152a`, Strata v0.1.53) in the
agent's Linux VM, which has no GPU and no GPU toolchain. Purpose: learn what a
CPU-only build does before anyone tries it on real hardware. Every claim below
is from commands actually run in this session; where a step did not complete,
it says so and says why. No benchmarks were run and no numbers were produced.

## Environment (measured)

| | |
|---|---|
| OS | Ubuntu 24.04 (VM), kernel-space x86-64 |
| CMake | 3.28.3 (tree requires ≥ 3.24 — OK) |
| Compiler | GCC / G++ 13.3.0 |
| Ninja | not installed → CMake default (Unix Makefiles) generator |
| Python | 3.12.3 |
| CPU / RAM / disk | 4 cores / 7 GB / ~148 GB free |
| GPU | none |

## Commands run

```sh
# 1. Configure (CPU-only: no STRATA_ENABLE_CUDA / _HIP / _SYCL)
cmake -S ~/workspace/frankenstrata -B /tmp/franken-probe/build -DCMAKE_BUILD_TYPE=Release

# 2. Build (attempted once — see "What happened", step 2)
timeout 1500 cmake --build /tmp/franken-probe/build -j2
```

## What happened

**Step 1 — configure: reached the ggml dependency fetch and did not finish
within the probe's effort cap (~16 minutes), so it was stopped.**

The configure log (kept at `/tmp/franken-probe/configure.log` during the
session) shows everything up to the fetch passing cleanly:

- CXX and C compiler identification (GNU 13.3.0), ABI info, compile-feature
  detection — all done.
- `STRATA_HAVE_AVXVNNI_INTRINSICS` — Success; `CMAKE_HAVE_LIBC_PTHREAD` —
  Success; `Found Threads: TRUE`.

It then went silent at the ggml step. With no `STRATA_GGML_DIR` given, the
tree FetchContent-clones llama.cpp (CMakeLists.txt:1896–1903):

```cmake
FetchContent_Declare(strata_llamacpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
  GIT_SHALLOW FALSE)
```

`GIT_SHALLOW FALSE` means a **full-history clone**. In this VM the clone was
still running when the probe was capped: the `_deps/strata_llamacpp-src`
directory grew 0 → 118 MB → 264 MB → 414 MB → **418 MB** over ~16 minutes
(~26 MB/min) without completing. Network reachability is not the problem —
`git ls-remote https://github.com/ggml-org/llama.cpp.git HEAD` returned
successfully (HEAD `785322d1…`) — the fetch is simply large and slow here.
The configure was then terminated (SIGTERM to the cmake/git process group);
no `CMakeCache.txt` was ever produced and "Configuring done" was never
reached. This is a **dependency-fetch duration blocker, not a compile or
toolchain failure** — nothing in the tree's own code was reached, so this
probe says nothing (good or bad) about whether the CPU-only targets compile.

**Step 2 — build: never legitimately ran.** One build attempt was fired
against the still-unconfigured tree and failed immediately with
`Error: could not load cache` (exit 1) — the expected error when no
`CMakeCache.txt` exists. It is recorded here only so nobody mistakes it for
a build result.

## How to finish this probe (next attempt)

- **Skip the fetch entirely:** `-DSTRATA_GGML_DIR=<llama.cpp checkout at
  3cf03257…>` — the tree then uses that checkout's `ggml/` directly
  (CMakeLists.txt:1839, 1893–1894). Pre-clone llama.cpp once, anywhere with
  a faster link, and every subsequent configure is fetch-free.
- Or re-run the same configure and let the clone finish (budget: at this
  VM's observed rate, plan on the clone alone taking well over 16 minutes;
  the full llama.cpp history is larger than the 418 MB observed).
- After a completed configure, build the CPU-only targets first
  (`strata_artifact`, `tests` per the top-level CMakeLists header) before
  the full default target set, and keep `-j2` on a 7 GB machine.

## Relevance to the reference rig

The same FetchContent applies to from-source builds on Philip's Ubuntu
unless `STRATA_GGML_DIR` is preset — worth knowing before a build night, but
it does not affect the documented routes in the Phase 0 runbook
(`./setup.sh --backend sycl` / the `strata-sycl-dev` image), which were not
exercised by this probe.
