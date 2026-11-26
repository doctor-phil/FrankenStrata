#!/usr/bin/env python3
"""package_franken.py — Stage a self-contained boxed Linux FrankenStrata bundle.

Produces a standalone directory (and optional .tar.gz) containing:
  - hip/      : primary `strata` + `franken-agent-hip` + bundled TheRock HIP/rocBLAS/hipBLASLt libs ($ORIGIN)
  - sycl/     : `franken-agent-sycl` + bundled oneAPI SYCL/UR Level-Zero/OpenCL runtime libs ($ORIGIN)
  - bin/      : vendor-free `franken-agent-cpu` + `franken-profile`
  - launcher/ : `franken_launch.py`
  - franken-strata : executable mixed-hardware launcher wrapper
  - BUILD.json     : bundle manifest

End users only need standard Linux kernel GPU drivers (amdgpu + xe/i915 + libze1) —
no ROCm install and no Intel oneAPI SDK install required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGING_DIR = REPO_ROOT / "tools" / "packaging"


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _find_patchelf() -> str:
    for cand in (REPO_ROOT / ".venv" / "bin" / "patchelf", shutil.which("patchelf")):
        if cand and Path(str(cand)).is_file():
            return str(cand)
    raise RuntimeError("patchelf not found (install via 'pip install patchelf' or 'apt install patchelf')")


def _detect_therock(repo_root: Path) -> tuple[Path, Path, Path]:
    """Returns (devel_root, core_root, libs_root) from .venv TheRock ROCm wheels."""
    venv = repo_root / ".venv"
    rocm_sdk = venv / "bin" / "rocm-sdk"
    if not rocm_sdk.is_file():
        raise RuntimeError(f"cannot find {rocm_sdk}; pass --rocm-devel/--rocm-core/--rocm-libs")
    devel = Path(subprocess.check_output([str(rocm_sdk), "path", "--root"], text=True).strip())
    sp = devel.parent
    core = sp / "_rocm_sdk_core"
    if not core.is_dir():
        core = devel
    libs_list = sorted(sp.glob("_rocm_sdk_libraries_*"))
    libs = libs_list[0] if libs_list else devel
    return devel, core, libs


def _patch_rpath_origin(folder: Path, patchelf: str) -> None:
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.is_symlink():
            continue
        with p.open("rb") as f:
            if f.read(4) != b"\x7fELF":
                continue
        _run([patchelf, "--set-rpath", "$ORIGIN", str(p)])


def _stage_hip(
    hip_build: Path,
    out_hip: Path,
    devel: Path,
    core: Path,
    libs: Path,
    arch: str,
    patchelf: str,
) -> list[str]:
    out_hip.mkdir(parents=True, exist_ok=True)
    bins = [hip_build / "strata"]
    hip_agent = hip_build / "franken-agent-hip"
    if hip_agent.is_file():
        bins.append(hip_agent)

    search_dirs = [libs / "lib", devel / "lib", core / "lib", core / "lib" / "llvm" / "lib"]
    sysdeps = devel / "lib" / "rocm_sysdeps" / "lib"
    if sysdeps.is_dir():
        search_dirs.append(sysdeps)

    data_args: list[str] = []
    for sub in ("rocblas/library", "hipblaslt/library"):
        src = libs / "lib" / sub
        if src.is_dir():
            data_args.extend(["--data", f"{src}={sub}"])
    if data_args and arch:
        data_args.extend(["--data-arch", arch])

    collected_bins: list[str] = []
    for idx, b in enumerate(bins):
        if not b.is_file():
            raise FileNotFoundError(f"missing HIP binary: {b}")
        cmd = [sys.executable, str(PACKAGING_DIR / "collect_libs.py"), "--binary", str(b)]
        for sd in search_dirs:
            if sd.is_dir():
                cmd.extend(["--search", str(sd)])
        cmd.extend(["--also", "libamd_comgr.so*", "--also", "libhiprtc*.so*", "--out", str(out_hip)])
        if idx == 0 and data_args:
            cmd.extend(data_args)
        _run(cmd)
        shutil.copy2(b, out_hip / b.name)
        collected_bins.append(b.name)

    _patch_rpath_origin(out_hip, patchelf)
    return collected_bins


def _stage_sycl(
    sycl_build: Path,
    out_sycl: Path,
    oneapi_root: Path | None,
    docker_image: str | None,
    patchelf: str,
) -> list[str]:
    out_sycl.mkdir(parents=True, exist_ok=True)
    agent_bin = sycl_build / "franken-agent-sycl"
    if not agent_bin.is_file():
        raise FileNotFoundError(f"missing SYCL agent binary: {agent_bin}")

    if oneapi_root is not None and oneapi_root.is_dir():
        cmd = [
            sys.executable, str(PACKAGING_DIR / "collect_libs.py"),
            "--binary", str(agent_bin),
            "--search", str(oneapi_root / "compiler" / "latest" / "lib"),
            "--search", str(oneapi_root / "mkl" / "latest" / "lib"),
            "--also", "libur_adapter_level_zero*.so*",
            "--also", "libur_adapter_opencl*.so*",
            "--out", str(out_sycl),
        ]
        _run(cmd)
    elif docker_image:
        # Run collect_libs.py inside the SYCL dev container so oneAPI runtime .so files are collected cleanly.
        uid, gid = os.getuid(), os.getgid()
        rel_bin = agent_bin.resolve().relative_to(REPO_ROOT.resolve())
        inner = (
            f"set -euo pipefail; "
            f"python3 /src/tools/packaging/collect_libs.py "
            f"--binary /src/{rel_bin.as_posix()} "
            f"--search /opt/intel/oneapi/compiler/latest/lib "
            f"--search /opt/intel/oneapi/mkl/latest/lib "
            f"--also 'libur_adapter_level_zero*.so*' "
            f"--also 'libur_adapter_opencl*.so*' "
            f"--out /stage_sycl; "
            f"chown -R {uid}:{gid} /stage_sycl"
        )
        _run([
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/src:ro",
            "-v", f"{out_sycl.resolve()}:/stage_sycl",
            docker_image, inner,
        ])
    else:
        raise RuntimeError("either --oneapi-root or --sycl-docker-image is required to bundle SYCL runtime libs")

    shutil.copy2(agent_bin, out_sycl / agent_bin.name)
    _patch_rpath_origin(out_sycl, patchelf)
    return [agent_bin.name]


def _stage_host_tools(hip_build: Path, out_bin: Path, patchelf: str) -> list[str]:
    out_bin.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for name in ("franken-agent-cpu", "franken-profile"):
        src = hip_build / name
        if src.is_file():
            dst = out_bin / name
            shutil.copy2(src, dst)
            staged.append(name)
    _patch_rpath_origin(out_bin, patchelf)
    return staged


def _stage_launcher(out_root: Path) -> None:
    ldir = out_root / "launcher"
    ldir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "franken" / "launcher" / "__init__.py", ldir / "__init__.py")
    shutil.copy2(REPO_ROOT / "franken" / "launcher" / "franken_launch.py", ldir / "franken_launch.py")

    wrapper = out_root / "franken-strata"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "exec python3 \"$DIR/launcher/franken_launch.py\" --bundle-dir \"$DIR\" \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _project_version() -> str:
    txt = (REPO_ROOT / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        if line.strip().startswith("project(strata VERSION "):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return "0.1.55"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Package a self-contained boxed Linux FrankenStrata bundle.")
    ap.add_argument("--hip-build", type=Path, default=REPO_ROOT / "build-hip-franken")
    ap.add_argument("--sycl-build", type=Path, default=REPO_ROOT / "build-sycl-franken")
    ap.add_argument("--rocm-devel", type=Path, default=None)
    ap.add_argument("--rocm-core", type=Path, default=None)
    ap.add_argument("--rocm-libs", type=Path, default=None)
    ap.add_argument("--oneapi-root", type=Path, default=None)
    ap.add_argument("--sycl-docker-image", default="strata-sycl-dev:latest")
    ap.add_argument("--arch", default="gfx1100", help="HIP target architecture(s) for rocBLAS/hipBLASLt data")
    ap.add_argument("--sycl-aot", default="bmg-g31", help="SYCL AOT target recorded in BUILD.json")
    ap.add_argument("--out", type=Path, required=True, help="Output staged bundle directory")
    ap.add_argument("--tarball", type=Path, default=None, help="Optional .tar.gz archive path")
    ap.add_argument("--verify", action="store_true", default=True, help="Verify ELF dependency closure (default on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ns = ap.parse_args(argv)

    patchelf = _find_patchelf()
    if ns.rocm_devel is None:
        devel, core, libs = _detect_therock(REPO_ROOT)
    else:
        devel = ns.rocm_devel
        core = ns.rocm_core or devel
        libs = ns.rocm_libs or devel

    out_root = ns.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    hip_bins = _stage_hip(ns.hip_build, out_root / "hip", devel, core, libs, ns.arch, patchelf)
    sycl_bins = _stage_sycl(ns.sycl_build, out_root / "sycl", ns.oneapi_root, ns.sycl_docker_image, patchelf)
    host_bins = _stage_host_tools(ns.hip_build, out_root / "bin", patchelf)
    _stage_launcher(out_root)

    manifest = {
        "product": "FrankenStrata",
        "version": _project_version(),
        "commit": _git_commit(),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": "linux-x86_64",
        "primary": {"backend": "hip", "archs": [a for a in ns.arch.split(",") if a], "binaries": hip_bins},
        "agents": {
            "sycl": {"aot": ns.sycl_aot, "binaries": sycl_bins},
            "host": {"binaries": host_bins},
        },
        "launcher": "franken-strata",
        "rpath_origin": True,
    }
    (out_root / "BUILD.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if ns.verify:
        for sub in ("hip", "sycl", "bin"):
            _run([sys.executable, str(PACKAGING_DIR / "binary_deps.py"), str(out_root / sub)])
        # Verify clean-environment execution (env -i) of bundled binaries.
        _run(["env", "-i", str(out_root / "hip" / "strata"), "--version"])
        _run(["env", "-i", "PATH=/usr/bin:/bin", str(out_root / "franken-strata"), "--plan"])

    if ns.tarball is not None:
        ns.tarball.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ns.tarball, "w:gz") as tf:
            tf.add(out_root, arcname=out_root.name)
        print(f"wrote tarball: {ns.tarball} ({ns.tarball.stat().st_size / (1024**2):.1f} MiB)")

    print(f"staged FrankenStrata bundle: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
