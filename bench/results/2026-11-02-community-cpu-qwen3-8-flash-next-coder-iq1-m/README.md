# Community benchmark on CPU only

Measured on 2026-11-02 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 9 5900XT 16-Core Processor (16 cores, 32 threads); installed RAM: 45.9 GiB; storage: sda     Samsung SSD 860 EVO 1TB               sata; sdb     Samsung SSD 860 EVO 1TB               sata; nvme0n1 Samsung SSD 980 PRO with Heatsink 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit 84b63202; engine 0.1.55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (sycl backend); thinking off for the suite; temperature 0

```text
strata-sycl.sh --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --ple-io ram --vision --kv-resident 32768 --stream-experts --vram-reserve-mib 2048 --prefill 8192
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 63.0 (57.5-84.2) | 0.40 (0.37-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1920.2 (1909.1-1921.2) | not measured | 2.12 (2.12-2.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2165.4 (2151.2-2165.9) | not measured | 7.45 (7.45-7.51) |
| agent session, follow-up turns | 12799 (12766-12912) at the last turn | see below | up to 64 | 3 | see below | 63.6 (23.6-80.9) | 1.74 (1.20-2.13) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.3); earlier-turn tokens read again per session: 4 (3-49) (earlier turns that the cache did not hold; the 12061 (12060-12061) new tokens of the files are read once); first turn (nothing cached) first token: 0.70 s (0.68-0.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 983.4 | 0.701 | 71.2 |
| 1 | 3356 | 627 | 2729 | 1587.5 | 1.857 | 68.6 |
| 2 | 4866 | 3349 | 1517 | 1245.0 | 1.289 | 73.4 |
| 3 | 8108 | 4929 | 3179 | 1547.5 | 2.116 | 65.1 |
| 4 | 10176 | 8173 | 2003 | 1337.6 | 1.546 | 64.5 |
| 5 | 12912 | 10231 | 2681 | 1432.8 | 1.921 | 70.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 90 s.
