#!/usr/bin/env python3
"""franken_launch.py — Unified mixed-hardware launcher and agent supervisor for FrankenStrata.

Discovers AMD (HIP), Intel (SYCL), and NVIDIA (CUDA) GPUs on Linux, loads or
generates franken-hw-profile.json, selects Primary and Agent tiers, supervises
franken-agent-<vendor> subprocesses (native bundled binaries or strata-sycl-dev
Docker fallback), forwards --franken-* flags to the primary engine, and guarantees
clean process-group teardown and /dev/shm cleanup on exit or error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PCI_VENDOR_AMD = "0x1002"
PCI_VENDOR_INTEL = "0x8086"
PCI_VENDOR_NVIDIA = "0x10de"

# Known Intel discrete Arc PCI device ID prefixes (Alchemist DG2 0x56xx, Battlemage BMG 0xe2xx).
INTEL_DISCRETE_PREFIXES = ("0x56", "0xe2")


@dataclass
class DiscoveredGpu:
    vendor: str           # "hip", "sycl", "cuda"
    index: int
    name: str
    vram_bytes: int
    driver: str = ""
    pci_id: str = ""
    drm_card: str = ""
    discrete: bool = True


@dataclass
class AgentLaunchSpec:
    vendor: str           # "sycl", "hip", "cuda", "cpu"
    device_index: int
    shm_name: str
    slots: int = 0
    device_name: str = ""
    use_docker: bool = False
    docker_image: str = "strata-sycl-dev:latest"
    binary: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchPlan:
    primary: DiscoveredGpu | None
    agents: list[AgentLaunchSpec]
    profile_path: Path | None = None
    cost_model: str = "on"
    adapt_swaps: int | None = None
    deadline_ms: float | None = None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _read_int(path: Path) -> int:
    txt = _read_text(path)
    if not txt:
        return 0
    try:
        return int(txt, 0)
    except ValueError:
        return 0


def _read_kfd_gpu_vram(kfd_nodes: Path) -> list[tuple[str, int]]:
    """Returns (name, vram_bytes) entries for discrete KFD GPU nodes."""
    out: list[tuple[str, int]] = []
    if not kfd_nodes.is_dir():
        return out
    for node in sorted(kfd_nodes.iterdir()):
        props = _read_text(node / "properties")
        if not props:
            continue
        kv: dict[str, str] = {}
        for line in props.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                kv[parts[0]] = parts[1]
        if int(kv.get("simd_count", "0")) <= 0:
            continue
        vram = 0
        mem_dir = node / "mem_banks"
        if mem_dir.is_dir():
            for bank in sorted(mem_dir.iterdir()):
                bp = _read_text(bank / "properties")
                bkv = {p.split()[0]: p.split()[1] for p in bp.splitlines() if len(p.split()) >= 2}
                # heap_type 1 = FB_PUBLIC, 2 = FB_PRIVATE (VRAM)
                if bkv.get("heap_type") in ("1", "2"):
                    vram += int(bkv.get("size_in_bytes", "0"))
        raw_name = _read_text(node / "name")
        gfx_ver = kv.get("gfx_target_version", "")
        if not raw_name or raw_name.lower() == "ip discovery":
            try:
                gv = int(gfx_ver)
                gfx_tag = f"gfx{gv // 10000}{(gv // 100) % 100:x}{gv % 100:x}"
            except ValueError:
                gfx_tag = gfx_ver
            name = f"AMD Radeon ({gfx_tag})" if gfx_tag else "AMD Radeon GPU"
        else:
            name = raw_name
        out.append((name, vram))
    return out


def discover_gpus(
    sysfs_drm: Path | str = "/sys/class/drm",
    kfd_nodes: Path | str = "/sys/class/kfd/kfd/topology/nodes",
    nvidia_smi_rows: Sequence[str] | None = None,
) -> list[DiscoveredGpu]:
    """Discovers AMD (hip), Intel (sycl), and NVIDIA (cuda) GPUs on Linux."""
    drm_root = Path(sysfs_drm)
    kfd_root = Path(kfd_nodes)
    kfd_gpus = _read_kfd_gpu_vram(kfd_root)
    gpus: list[DiscoveredGpu] = []
    vendor_counts = {"hip": 0, "sycl": 0, "cuda": 0}

    if drm_root.is_dir():
        for entry in sorted(drm_root.iterdir()):
            name = entry.name
            if not name.startswith("card") or "-" in name:
                continue
            dev = entry / "device"
            vendor_hex = _read_text(dev / "vendor").lower()
            if vendor_hex not in (PCI_VENDOR_AMD, PCI_VENDOR_INTEL, PCI_VENDOR_NVIDIA):
                continue
            pci_id = _read_text(dev / "device").lower()
            driver = ""
            try:
                driver = (dev / "driver").resolve().name
            except OSError:
                driver = ""

            if vendor_hex == PCI_VENDOR_AMD:
                vram = _read_int(dev / "mem_info_vram_total")
                idx = vendor_counts["hip"]
                vendor_counts["hip"] += 1
                kfd_name = kfd_gpus[idx][0] if idx < len(kfd_gpus) else ""
                if vram == 0 and idx < len(kfd_gpus):
                    vram = kfd_gpus[idx][1]
                # Integrated Radeon GPUs typically expose < 2 GiB dedicated VRAM in mem_info_vram_total.
                discrete = vram == 0 or vram >= (2 << 30)
                gpus.append(DiscoveredGpu(
                    vendor="hip",
                    index=idx,
                    name=kfd_name or f"AMD Radeon ({pci_id or name})",
                    vram_bytes=vram,
                    driver=driver or "amdgpu",
                    pci_id=pci_id,
                    drm_card=name,
                    discrete=discrete,
                ))
            elif vendor_hex == PCI_VENDOR_INTEL:
                vram = _read_int(dev / "mem_info_vram_total") or _read_int(dev / "lmem_total_bytes")
                discrete = any(pci_id.startswith(p) for p in INTEL_DISCRETE_PREFIXES) or vram >= (4 << 30)
                idx = vendor_counts["sycl"]
                vendor_counts["sycl"] += 1
                label = "Intel Arc Battlemage" if pci_id.startswith("0xe2") else (
                    "Intel Arc Alchemist" if pci_id.startswith("0x56") else "Intel Graphics"
                )
                gpus.append(DiscoveredGpu(
                    vendor="sycl",
                    index=idx,
                    name=f"{label} [{pci_id}]" if pci_id else label,
                    vram_bytes=vram,
                    driver=driver or "xe",
                    pci_id=pci_id,
                    drm_card=name,
                    discrete=discrete,
                ))
            elif vendor_hex == PCI_VENDOR_NVIDIA:
                idx = vendor_counts["cuda"]
                vendor_counts["cuda"] += 1
                gpus.append(DiscoveredGpu(
                    vendor="cuda",
                    index=idx,
                    name=f"NVIDIA GPU [{pci_id}]" if pci_id else f"NVIDIA GPU {idx}",
                    vram_bytes=0,
                    driver=driver or "nvidia",
                    pci_id=pci_id,
                    drm_card=name,
                    discrete=True,
                ))

    # Enrich NVIDIA GPUs via nvidia-smi when available (or injected rows in tests).
    nv_rows: list[str] = []
    if nvidia_smi_rows is not None:
        nv_rows = list(nvidia_smi_rows)
    elif any(g.vendor == "cuda" for g in gpus) and shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if r.returncode == 0:
                nv_rows = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        except OSError:
            pass
    for row in nv_rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            continue
        try:
            n_idx = int(parts[0])
            n_vram = int(float(parts[2])) * (1024 * 1024)
        except ValueError:
            continue
        matched = False
        for g in gpus:
            if g.vendor == "cuda" and g.index == n_idx:
                g.name = parts[1]
                g.vram_bytes = n_vram
                matched = True
                break
        if not matched:
            gpus.append(DiscoveredGpu(
                vendor="cuda", index=n_idx, name=parts[1], vram_bytes=n_vram,
                driver="nvidia", discrete=True,
            ))

    return gpus


def _load_profile_scores(profile_path: Path | None) -> tuple[dict[str, float], dict[str, int]]:
    """Returns (vendor_or_id -> T(4)_ms, vendor_or_id -> recommended_slots) from franken-hw-profile.json."""
    scores: dict[str, float] = {}
    slots: dict[str, int] = {}
    if profile_path is None or not profile_path.is_file():
        return scores, slots
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return scores, slots
    for g in data.get("gpus", []):
        vendor = str(g.get("vendor", "")).lower()
        gid = str(g.get("id", ""))
        a = float(g.get("a_ipc_ms", 0.0))
        b = float(g.get("b_per_expert_ms", 0.0))
        t4 = a + 4.0 * b
        rec_slots = int(g.get("recommended_slots", 0))
        if t4 > 0.0:
            if vendor:
                scores[vendor] = min(scores.get(vendor, 1e9), t4)
            if gid:
                scores[gid] = t4
        if rec_slots > 0:
            if vendor:
                slots[vendor] = max(slots.get(vendor, 0), rec_slots)
            if gid:
                slots[gid] = rec_slots
    return scores, slots


def select_tier_plan(
    gpus: Sequence[DiscoveredGpu],
    profile_path: Path | None = None,
    preferred_primary: str | None = None,
    shm_prefix: str = "/franken",
    agent_slots_override: int = 0,
    cost_model: str = "on",
    adapt_swaps: int | None = None,
    deadline_ms: float | None = None,
    include_cpu_fallback: bool = False,
) -> LaunchPlan:
    """Deterministically selects Primary GPU and ordered foreign-vendor Agent tiers."""
    discrete = [g for g in gpus if g.discrete]
    candidates = discrete if discrete else list(gpus)
    if not candidates:
        agents: list[AgentLaunchSpec] = []
        if include_cpu_fallback:
            agents.append(AgentLaunchSpec(
                vendor="cpu", device_index=0, shm_name=f"{shm_prefix}_cpu0",
                slots=agent_slots_override, device_name="CPU Conformance Agent",
            ))
        return LaunchPlan(primary=None, agents=agents, profile_path=profile_path,
                          cost_model=cost_model, adapt_swaps=adapt_swaps, deadline_ms=deadline_ms)

    prof_t4, prof_slots = _load_profile_scores(profile_path)
    vendor_rank = {"cuda": 0, "hip": 1, "sycl": 2}

    def sort_key(g: DiscoveredGpu):
        # Lower key = better Primary candidate.
        pref = 0 if (preferred_primary and g.vendor == preferred_primary.lower()) else 1
        if g.vendor in prof_t4:
            return (pref, 0, prof_t4[g.vendor], -g.vram_bytes, g.index)
        return (pref, 1, vendor_rank.get(g.vendor, 9), -g.vram_bytes, g.index)

    ranked = sorted(candidates, key=sort_key)
    primary = ranked[0]
    agents = []
    for i, ag in enumerate(ranked[1:]):
        shm = f"{shm_prefix}_{ag.vendor}{i}"
        slots = agent_slots_override if agent_slots_override > 0 else 0
        # If profile gave a recommended slot cap and caller didn't override, leave 0 so
        # primary assigns all remaining unclaimed experts up to agent HELLO slot_count.
        agents.append(AgentLaunchSpec(
            vendor=ag.vendor,
            device_index=ag.index,
            shm_name=shm,
            slots=slots,
            device_name=ag.name,
        ))
    return LaunchPlan(
        primary=primary,
        agents=agents,
        profile_path=profile_path if (profile_path and profile_path.is_file()) else None,
        cost_model=cost_model,
        adapt_swaps=adapt_swaps,
        deadline_ms=deadline_ms,
    )


def _therock_env(repo_root: Path) -> dict[str, str]:
    """Adds .venv TheRock ROCm library directories to LD_LIBRARY_PATH when present."""
    env = dict(os.environ)
    venv = repo_root / ".venv"
    rocm_sdk = venv / "bin" / "rocm-sdk"
    if not rocm_sdk.is_file():
        return env
    try:
        root = Path(subprocess.check_output([str(rocm_sdk), "path", "--root"], text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        return env
    lib_dirs = [str(root / "lib")]
    for sp in venv.glob("lib/python*/site-packages"):
        for pkg in sorted(sp.glob("_rocm_sdk_libraries_*")):
            ld = pkg / "lib"
            if ld.is_dir():
                lib_dirs.append(str(ld))
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def resolve_binary(
    name: str,
    bundle_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path | None, dict[str, str]]:
    """Resolves an engine/agent binary from a boxed bundle first, then local build trees."""
    search_dirs: list[Path] = []
    if bundle_dir is not None:
        search_dirs.extend([bundle_dir, bundle_dir / "hip", bundle_dir / "sycl", bundle_dir / "cuda", bundle_dir / "bin"])
    search_dirs.extend([
        repo_root / "build-hip-franken",
        repo_root / "build-sycl-franken",
        repo_root / "build-hip",
        repo_root / "build-cuda",
        repo_root / "build",
        repo_root / "engine",
    ])
    for d in search_dirs:
        cand = d / name
        if cand.is_file() and os.access(cand, os.X_OK):
            # Boxed bundle binaries carry RUNPATH $ORIGIN; source builds use .venv TheRock libs.
            in_bundle = bundle_dir is not None and (bundle_dir == d or bundle_dir in d.parents)
            env = dict(os.environ) if in_bundle else _therock_env(repo_root)
            return cand, env
    return None, dict(os.environ)


def build_agent_command(
    spec: AgentLaunchSpec,
    pack_dir: str,
    native_gguf: str | None = None,
    wait_ms: int = 90000,
    bundle_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    docker_container_name: str | None = None,
) -> tuple[list[str], dict[str, str], bool]:
    """Builds the argv, env, and is_docker flag for one agent subprocess."""
    exe_name = f"franken-agent-{spec.vendor}"
    binary = spec.binary
    env = dict(spec.env) if spec.env else {}
    if binary is None:
        binary, resolved_env = resolve_binary(exe_name, bundle_dir=bundle_dir, repo_root=repo_root)
        if not env:
            env = resolved_env

    # Native SYCL binary works directly when bundled with its SYCL runtime libs
    # (libsycl.so* in same folder), otherwise fall back to strata-sycl-dev Docker image.
    use_docker = spec.use_docker
    if spec.vendor == "sycl" and not use_docker:
        if binary is None:
            use_docker = True
        else:
            has_sycl_rt = any(binary.parent.glob("libsycl.so*")) or bool(shutil.which("icpx"))
            if not has_sycl_rt:
                use_docker = True

    agent_args = ["--shm", spec.shm_name, "--pack", pack_dir, "--wait-ms", str(wait_ms)]
    if native_gguf:
        agent_args.extend(["--native", native_gguf])
    if spec.slots > 0:
        agent_args.extend(["--slots", str(spec.slots)])
    if spec.vendor in ("hip", "cuda"):
        agent_args.extend(["--device", str(spec.device_index)])

    if use_docker:
        cname = docker_container_name or f"franken_agent_{os.getpid()}_{spec.shm_name.lstrip('/')}"
        # Mount repo_root (containing build-sycl-franken), pack_dir, native_gguf parent, and /dev/shm.
        mounts: list[tuple[str, str, bool]] = [(str(repo_root), "/src", False)]
        pack_parent = str(Path(pack_dir).resolve().parent)
        mounts.append((pack_parent, pack_parent, True))
        if native_gguf:
            gguf_parent = str(Path(native_gguf).resolve().parent)
            if not any(gguf_parent == m[0] or gguf_parent.startswith(m[0].rstrip("/") + "/") for m in mounts):
                mounts.append((gguf_parent, gguf_parent, True))
        if binary is not None:
            try:
                rel = binary.resolve().relative_to(repo_root.resolve())
                container_bin = f"/src/{rel.as_posix()}"
            except ValueError:
                mounts.append((str(binary.parent), "/agent_bin", True))
                container_bin = f"/agent_bin/{binary.name}"
        else:
            container_bin = f"/src/build-sycl-franken/{exe_name}"

        cmd = ["docker", "run", "--rm", "--name", cname, "--device", "/dev/dri", "-v", "/dev/shm:/dev/shm"]
        for host_p, cont_p, ro in mounts:
            cmd.extend(["-v", f"{host_p}:{cont_p}{':ro' if ro else ''}"])
        quoted = " ".join(subprocess.list2cmdline([a]) if " " in a else a for a in [container_bin, *agent_args])
        cmd.extend([spec.docker_image, quoted])
        return cmd, dict(os.environ), True

    if binary is None:
        raise FileNotFoundError(f"cannot find {exe_name} (checked bundle and build directories)")
    return [str(binary), *agent_args], env, False


def build_primary_franken_args(plan: LaunchPlan) -> list[str]:
    """Builds the --franken-* CLI flags appended to the primary `strata` command."""
    args: list[str] = []
    for ag in plan.agents:
        val = f"{ag.shm_name},slots={ag.slots}" if ag.slots > 0 else ag.shm_name
        args.extend(["--franken-agent", val])
    if plan.agents:
        args.extend(["--franken-cost-model", plan.cost_model])
        if plan.profile_path is not None and plan.profile_path.is_file():
            args.extend(["--franken-profile", str(plan.profile_path)])
        if plan.adapt_swaps is not None:
            args.extend(["--franken-adapt-swaps", str(plan.adapt_swaps)])
        if plan.deadline_ms is not None and plan.deadline_ms > 0:
            args.extend(["--franken-deadline-ms", f"{plan.deadline_ms:g}"])
    return args


def cleanup_shm_paths(shm_names: Iterable[str], shm_dir: Path | str = "/dev/shm") -> list[Path]:
    """Unlinks any leftover POSIX /dev/shm files for the given shm names."""
    root = Path(shm_dir)
    removed: list[Path] = []
    for raw in shm_names:
        clean = raw.lstrip("/")
        if not clean or "/" in clean:
            continue
        p = root / clean
        try:
            if p.exists() or p.is_symlink():
                p.unlink()
                removed.append(p)
        except OSError:
            pass
    return removed


class AgentSupervisor:
    """Spawns and supervises 1..N FrankenStrata agent subprocesses with guaranteed teardown."""

    def __init__(
        self,
        specs: Sequence[AgentLaunchSpec],
        pack_dir: str,
        native_gguf: str | None = None,
        bundle_dir: Path | None = None,
        repo_root: Path = REPO_ROOT,
        shm_dir: Path | str = "/dev/shm",
        wait_ms: int = 90000,
        log_dir: Path | None = None,
    ) -> None:
        self.specs = list(specs)
        self.pack_dir = pack_dir
        self.native_gguf = native_gguf
        self.bundle_dir = bundle_dir
        self.repo_root = repo_root
        self.shm_dir = Path(shm_dir)
        self.wait_ms = wait_ms
        self.log_dir = log_dir
        self._procs: list[tuple[AgentLaunchSpec, subprocess.Popen, str | None, object | None]] = []

    def start(self) -> None:
        for i, spec in enumerate(self.specs):
            cname = f"franken_agent_{os.getpid()}_{i}_{spec.vendor}"
            cmd, env, is_docker = build_agent_command(
                spec,
                pack_dir=self.pack_dir,
                native_gguf=self.native_gguf,
                wait_ms=self.wait_ms,
                bundle_dir=self.bundle_dir,
                repo_root=self.repo_root,
                docker_container_name=cname,
            )
            log_fh = None
            stderr_dst = sys.stderr
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log_fh = (self.log_dir / f"franken-agent-{spec.vendor}-{i}.log").open("w", encoding="utf-8")
                stderr_dst = log_fh
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_dst,
                start_new_session=True,
            )
            self._procs.append((spec, proc, cname if is_docker else None, log_fh))

    def stop(self, grace_s: float = 3.0) -> None:
        deadline = time.monotonic() + grace_s
        for _, proc, _, _ in self._procs:
            rem = max(0.05, deadline - time.monotonic())
            if proc.poll() is None:
                try:
                    proc.wait(timeout=rem)
                except subprocess.TimeoutExpired:
                    pass

        for _, proc, cname, _ in self._procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    if cname:
                        subprocess.run(["docker", "rm", "-f", cname],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        try:
                            proc.kill()
                        except OSError:
                            pass
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass

        for _, _, _, fh in self._procs:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        self._procs.clear()
        cleanup_shm_paths([s.shm_name for s in self.specs], shm_dir=self.shm_dir)

    def __enter__(self) -> "AgentSupervisor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def run_hardware_profiler(
    plan: LaunchPlan,
    pack_dir: str,
    native_gguf: str | None,
    out_path: Path,
    bundle_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Runs franken-profile against the planned agents and writes franken-hw-profile.json."""
    prof_bin, prof_env = resolve_binary("franken-profile", bundle_dir=bundle_dir, repo_root=repo_root)
    if prof_bin is None:
        print("franken-strata: cannot find franken-profile binary", file=sys.stderr)
        return 1

    probe_specs = [
        AgentLaunchSpec(
            vendor=ag.vendor,
            device_index=ag.device_index,
            shm_name=f"/franken_prof_{os.getpid()}_{i}",
            slots=ag.slots,
            device_name=ag.device_name,
            use_docker=ag.use_docker,
            docker_image=ag.docker_image,
            binary=ag.binary,
        )
        for i, ag in enumerate(plan.agents)
    ]

    cmd = [str(prof_bin), "--pack", pack_dir, "--out", str(out_path)]
    if native_gguf:
        cmd.extend(["--native", native_gguf])
    if plan.primary is not None:
        cmd.extend([
            "--primary-vendor", plan.primary.vendor.upper(),
            "--primary-name", plan.primary.name,
            "--primary-vram-gib", f"{plan.primary.vram_bytes / (1024**3):.2f}",
        ])
    for ps in probe_specs:
        cmd.extend(["--agent", ps.shm_name])

    with AgentSupervisor(probe_specs, pack_dir=pack_dir, native_gguf=native_gguf,
                         bundle_dir=bundle_dir, repo_root=repo_root):
        res = subprocess.run(cmd, env=prof_env, check=False)
        return res.returncode


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="franken-strata",
        description="Unified mixed-hardware launcher for FrankenStrata (AMD HIP + Intel SYCL + NVIDIA CUDA + CPU).",
    )
    ap.add_argument("--bundle-dir", type=Path, default=None, help="Boxed FrankenStrata bundle directory")
    ap.add_argument("--primary-bin", type=Path, default=None, help="Override path to primary strata executable")
    ap.add_argument("--primary-vendor", choices=("hip", "cuda", "sycl"), default=None,
                    help="Prefer this GPU vendor as Primary")
    ap.add_argument("--agents", default="auto",
                    help="Comma-separated agent vendors (e.g. 'sycl', 'cpu', 'none') or 'auto' (default)")
    ap.add_argument("--agent-slots", type=int, default=0, help="Cap slots per agent (0 = auto)")
    ap.add_argument("--docker-sycl", action="store_true", help="Force Docker strata-sycl-dev:latest for SYCL agents")
    ap.add_argument("--docker-image", default="strata-sycl-dev:latest", help="Docker image for SYCL fallback")
    ap.add_argument("--hw-profile", type=Path, default=Path("franken-hw-profile.json"),
                    help="Path to franken-hw-profile.json")
    ap.add_argument("--profile", action="store_true",
                    help="Run franken-profile first if --hw-profile does not exist")
    ap.add_argument("--reprofile", action="store_true", help="Force re-running franken-profile before launch")
    ap.add_argument("--profile-only", action="store_true", help="Run franken-profile and exit")
    ap.add_argument("--cost-model", choices=("on", "off"), default="on",
                    help="Online latency cost-model dispatch gating (default: on)")
    ap.add_argument("--adapt-swaps", type=int, default=None, help="Override --franken-adapt-swaps N")
    ap.add_argument("--deadline-ms", type=float, default=None, help="Override --franken-deadline-ms MS")
    ap.add_argument("--plan", action="store_true", help="Print discovered GPUs and launch plan without running")
    ap.add_argument("--json", action="store_true", help="Output --plan as JSON")
    return ap


def _extract_flag(argv: Sequence[str], flag: str) -> str | None:
    for i in range(len(argv) - 1):
        if argv[i] == flag:
            return argv[i + 1]
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    ns, passthrough = ap.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    gpus = discover_gpus()
    shm_prefix = f"/franken_{os.getpid()}"
    plan = select_tier_plan(
        gpus,
        profile_path=ns.hw_profile if ns.hw_profile.is_file() else None,
        preferred_primary=ns.primary_vendor,
        shm_prefix=shm_prefix,
        agent_slots_override=ns.agent_slots,
        cost_model=ns.cost_model,
        adapt_swaps=ns.adapt_swaps,
        deadline_ms=ns.deadline_ms,
    )

    # Apply explicit --agents override (e.g. --agents sycl, --agents cpu, --agents none).
    if ns.agents.lower() == "none":
        plan.agents = []
    elif ns.agents.lower() != "auto":
        plan.agents = []
        for idx, raw_v in enumerate(ns.agents.split(",")):
            v = raw_v.strip().lower()
            if not v:
                continue
            plan.agents.append(AgentLaunchSpec(
                vendor=v,
                device_index=0,
                shm_name=f"{shm_prefix}_{v}{idx}",
                slots=ns.agent_slots,
                device_name=f"{v.upper()} agent",
            ))

    for ag in plan.agents:
        if ag.vendor == "sycl":
            ag.use_docker = ns.docker_sycl
            ag.docker_image = ns.docker_image

    if ns.plan:
        payload = {
            "gpus": [g.__dict__ for g in gpus],
            "primary": plan.primary.__dict__ if plan.primary else None,
            "agents": [
                {"vendor": a.vendor, "shm_name": a.shm_name, "slots": a.slots, "device_name": a.device_name}
                for a in plan.agents
            ],
            "hw_profile": str(plan.profile_path) if plan.profile_path else None,
            "franken_args": build_primary_franken_args(plan),
        }
        if ns.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Discovered GPUs ({len(gpus)}):")
            for g in gpus:
                print(f"  - [{g.vendor.upper()}:{g.index}] {g.name} ({g.vram_bytes / (1024**3):.1f} GiB, driver={g.driver})")
            if plan.primary:
                print(f"Primary : [{plan.primary.vendor.upper()}] {plan.primary.name}")
            for i, a in enumerate(plan.agents):
                print(f"Tier {i + 1}  : [{a.vendor.upper()}] {a.device_name} shm={a.shm_name} slots={a.slots or 'auto'}")
            print(f"Flags   : {' '.join(build_primary_franken_args(plan)) or '(solo primary)'}")
        return 0

    pack_dir = _extract_flag(passthrough, "--pack")
    native_gguf = _extract_flag(passthrough, "--native")

    if (ns.reprofile or ns.profile or ns.profile_only) and (ns.reprofile or not ns.hw_profile.is_file()):
        if not pack_dir:
            print("franken-strata: --profile/--reprofile requires --pack <dir>", file=sys.stderr)
            return 2
        rc = run_hardware_profiler(plan, pack_dir=pack_dir, native_gguf=native_gguf,
                                   out_path=ns.hw_profile, bundle_dir=ns.bundle_dir)
        if rc != 0:
            return rc
        plan.profile_path = ns.hw_profile
        if ns.profile_only:
            return 0

    primary_bin = ns.primary_bin
    primary_env = dict(os.environ)
    if primary_bin is None:
        primary_bin, primary_env = resolve_binary("strata", bundle_dir=ns.bundle_dir)
    if primary_bin is None:
        print("franken-strata: cannot find primary 'strata' executable (pass --primary-bin or --bundle-dir)",
              file=sys.stderr)
        return 1

    if plan.agents and not pack_dir:
        print("franken-strata: multi-GPU launch requires --pack <dir> so agents open the same pack",
              file=sys.stderr)
        return 2

    primary_cmd = [str(primary_bin), *passthrough, *build_primary_franken_args(plan)]

    if not plan.agents:
        return subprocess.run(primary_cmd, env=primary_env, check=False).returncode

    primary_proc: subprocess.Popen | None = None

    def _forward_signal(signum, _frame):
        if primary_proc is not None and primary_proc.poll() is None:
            try:
                primary_proc.send_signal(signum)
            except OSError:
                pass

    prev_int = signal.signal(signal.SIGINT, _forward_signal)
    prev_term = signal.signal(signal.SIGTERM, _forward_signal)
    try:
        with AgentSupervisor(
            plan.agents,
            pack_dir=pack_dir or "",
            native_gguf=native_gguf,
            bundle_dir=ns.bundle_dir,
        ):
            primary_proc = subprocess.Popen(primary_cmd, env=primary_env)
            return primary_proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


if __name__ == "__main__":
    raise SystemExit(main())
