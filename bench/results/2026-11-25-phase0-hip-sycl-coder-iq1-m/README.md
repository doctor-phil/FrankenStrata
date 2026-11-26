# Phase 0 baselines — Flash-Next Coder IQ1_M (2026-11-25)

Rig: Ryzen 9 5900XT, 64 GB DDR4-3200, ASUS B550-E, Ubuntu 26.04.
B70 32 GB in slot 1 (Gen3); 7900 XTX 24 GB (gfx1100) in slot 2 (Gen4 x8).
Protocol: runbook S4.0 — greedy, spec 4, spec-min-p 0.5, max-context 8192,
prompt sycl/tools/data/prompt_deadlock_2744.tok (2,743 tokens), 64 generated.

## Throughput rows

| date | engine | card / link | prefill tok/s | decode tok/s | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-04 | SYCL 0.1.55 | B70, slot 1, x16 Gen3 | 784.34 | 31.11 | speculation collapsed (1/179 drafts); PCIe probe 13.6 GB/s; 12,288/12,288 experts resident |
| 2026-11-25 | SYCL 0.1.55 (build-sycl-aot) | B70, slot 1, x8 Gen3 | 819.94 | 32.75 | same protocol; PCIe probe 7.1 GB/s; link halved, throughput unchanged on the all-resident workload |
| 2026-11-25 | HIP 0.1.69 (released rx7900 package) | 7900 XTX, slot 2, x8 Gen4 | 1655.27 | 79.05 | speculation worked (35/60 drafts, 2.17 tok/round); PCIe probe 14.2 GB/s; expert cache auto = 9,121/12,288 slots (17.37 GiB VRAM); dispatch plan/actq/jobs/run 0.095/0.072/0.010/2.471 ms/round |

## Kernel parity (SYCL, B70, 2026-11-25)

ctest (build-sycl-parity, on-card): 49/52 passed. Failures classified:
s2_expert_grouped_parity (upstream known gap on the s2 path);
ple_parity (missing Q2_0 fixture, not a defect);
gr_parity (QFUSE sub-checks only — STRATA_QFUSE is unsupported by design
on SYCL and env-gated off; the fallback quantize path is clean at ~1e-7).

## HIP vs SYCL divergence (this folder)

- 128-token prompt (p128.tok, first 128 IDs of the fixed prompt), 64 generated:
  outputs token-for-token identical (64/64). Window hashes in this folder.
- Full 2,743-token prompt, 64 generated: first divergence at token 39,
  40/64 strict position agreement; slip-and-resync pattern consistent with
  near-tie argmax flips under different rounding.
- Window hashes match 0 times across engines (bit-exactness probe over raw
  residual bits; a same-engine regression instrument, not cross-engine).
- --dump-logits writes nothing on native packs (documented in
  sycl/src/program/generate.cpp); --dump-layers conflicts with --spec.
  Rel-L2 / max-abs / cosine rows remain open; tools/par* is the candidate.

## Still owed

s2 pack and Qwen3.6-35B-A3B rows (downloads), CPU reference rows,
the commit of the franken/ agent work itself (Phase 1).
