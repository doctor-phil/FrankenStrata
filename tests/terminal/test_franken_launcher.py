"""Unit tests for franken/launcher/franken_launch.py (GPU discovery, tier plan, CLI assembly, teardown)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from franken.launcher.franken_launch import (
    AgentLaunchSpec,
    AgentSupervisor,
    DiscoveredGpu,
    LaunchPlan,
    build_agent_command,
    build_primary_franken_args,
    cleanup_shm_paths,
    discover_gpus,
    select_tier_plan,
)


def _mk_drm_card(root: Path, card: str, vendor: str, device: str, driver: str, vram_bytes: int = 0) -> None:
    dev = root / card / "device"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "vendor").write_text(vendor + "\n", encoding="utf-8")
    (dev / "device").write_text(device + "\n", encoding="utf-8")
    if vram_bytes > 0:
        (dev / "mem_info_vram_total").write_text(f"{vram_bytes}\n", encoding="utf-8")
    drv_target = root / "_drivers" / driver
    drv_target.mkdir(parents=True, exist_ok=True)
    (dev / "driver").symlink_to(drv_target)


def test_discover_mixed_amd_intel_nvidia(tmp_path: Path) -> None:
    drm = tmp_path / "drm"
    kfd = tmp_path / "kfd"
    _mk_drm_card(drm, "card1", "0x1002", "0x744c", "amdgpu", vram_bytes=24 << 30)
    _mk_drm_card(drm, "card2", "0x8086", "0xe223", "xe", vram_bytes=32 << 30)
    _mk_drm_card(drm, "card3", "0x10de", "0x2684", "nvidia", vram_bytes=0)

    # KFD node for the AMD 7900 XTX
    n1 = kfd / "1"
    (n1 / "mem_banks" / "0").mkdir(parents=True)
    (n1 / "properties").write_text("simd_count 192\ngfx_target_version 110000\n", encoding="utf-8")
    (n1 / "name").write_text("AMD Radeon RX 7900 XTX\n", encoding="utf-8")
    (n1 / "mem_banks" / "0" / "properties").write_text(
        f"heap_type 1\nsize_in_bytes {24 << 30}\n", encoding="utf-8"
    )

    gpus = discover_gpus(
        sysfs_drm=drm,
        kfd_nodes=kfd,
        nvidia_smi_rows=["0, NVIDIA GeForce RTX 4090, 24564"],
    )
    assert len(gpus) == 3
    by_vendor = {g.vendor: g for g in gpus}
    assert by_vendor["hip"].name == "AMD Radeon RX 7900 XTX"
    assert by_vendor["hip"].vram_bytes == (24 << 30)
    assert by_vendor["sycl"].driver == "xe"
    assert by_vendor["sycl"].discrete is True
    assert by_vendor["cuda"].name == "NVIDIA GeForce RTX 4090"
    assert by_vendor["cuda"].vram_bytes == 24564 * 1024 * 1024


def test_select_tier_plan_heuristics_and_profile(tmp_path: Path) -> None:
    xtx = DiscoveredGpu("hip", 0, "RX 7900 XTX", 24 << 30, "amdgpu", "0x744c", "card1", True)
    b70 = DiscoveredGpu("sycl", 0, "Arc Pro B70", 32 << 30, "xe", "0xe223", "card2", True)

    # Without profile: HIP beats SYCL as Primary; B70 becomes Tier 1 Agent.
    plan = select_tier_plan([b70, xtx], shm_prefix="/franken_unit")
    assert plan.primary is not None and plan.primary.vendor == "hip"
    assert len(plan.agents) == 1 and plan.agents[0].vendor == "sycl"
    assert plan.agents[0].shm_name == "/franken_unit_sycl0"

    # With profile where SYCL is faster than HIP: profile overrides heuristic!
    prof = tmp_path / "franken-hw-profile.json"
    prof.write_text(json.dumps({
        "schema_version": 1,
        "gpus": [
            {"id": "hip:0", "vendor": "HIP", "a_ipc_ms": 0.20, "b_per_expert_ms": 0.10, "recommended_slots": 9000},
            {"id": "/franken0", "vendor": "SYCL", "a_ipc_ms": 0.02, "b_per_expert_ms": 0.01, "recommended_slots": 12288},
        ],
    }), encoding="utf-8")
    plan_prof = select_tier_plan([xtx, b70], profile_path=prof, shm_prefix="/franken_unit")
    assert plan_prof.primary is not None and plan_prof.primary.vendor == "sycl"
    assert plan_prof.agents[0].vendor == "hip"
    assert plan_prof.profile_path == prof


def test_command_assembly_native_and_docker(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    sycl_dir = bundle / "sycl"
    sycl_dir.mkdir(parents=True)
    agent_bin = sycl_dir / "franken-agent-sycl"
    agent_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    agent_bin.chmod(0o755)
    # Place bundled libsycl.so.8 so native mode is selected (no Docker).
    (sycl_dir / "libsycl.so.8").write_bytes(b"ELF")

    spec = AgentLaunchSpec(vendor="sycl", device_index=0, shm_name="/franken_t0", slots=3000)
    cmd, _, is_docker = build_agent_command(
        spec, pack_dir="/work/pack", native_gguf="/work/model.gguf", bundle_dir=bundle
    )
    assert is_docker is False
    assert cmd[0] == str(agent_bin)
    assert "--shm" in cmd and "/franken_t0" in cmd
    assert "--slots" in cmd and "3000" in cmd
    assert "--native" in cmd and "/work/model.gguf" in cmd

    # Force Docker mode
    spec.use_docker = True
    dcmd, _, is_docker2 = build_agent_command(
        spec, pack_dir=str(tmp_path / "pack"), native_gguf=str(tmp_path / "m.gguf"),
        bundle_dir=bundle, repo_root=tmp_path, docker_container_name="test_c",
    )
    assert is_docker2 is True
    assert dcmd[:6] == ["docker", "run", "--rm", "--name", "test_c", "--device"]
    assert "/dev/shm:/dev/shm" in dcmd

    # Primary CLI flags
    prof = tmp_path / "hw.json"
    prof.write_text("{}", encoding="utf-8")
    p_args = build_primary_franken_args(LaunchPlan(
        primary=None, agents=[spec], profile_path=prof, cost_model="on", adapt_swaps=16, deadline_ms=25.0,
    ))
    assert p_args == [
        "--franken-agent", "/franken_t0,slots=3000",
        "--franken-cost-model", "on",
        "--franken-profile", str(prof),
        "--franken-adapt-swaps", "16",
        "--franken-deadline-ms", "25",
    ]


def test_agent_supervisor_teardown_and_shm_cleanup(tmp_path: Path) -> None:
    fake_shm_dir = tmp_path / "shm"
    fake_shm_dir.mkdir()
    shm_name = "/franken_sup_test"
    shm_file = fake_shm_dir / "franken_sup_test"
    shm_file.write_bytes(b"FRANKEN1")

    # Spawn a long-lived dummy agent script in its own process group.
    dummy_agent = tmp_path / "dummy_agent.sh"
    dummy_agent.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    dummy_agent.chmod(0o755)

    spec = AgentLaunchSpec(vendor="cpu", device_index=0, shm_name=shm_name, binary=dummy_agent)
    sup = AgentSupervisor([spec], pack_dir=str(tmp_path), shm_dir=fake_shm_dir)
    sup.start()
    assert len(sup._procs) == 1
    pid = sup._procs[0][1].pid
    assert sup._procs[0][1].poll() is None

    # Teardown must terminate the process group and unlink the shm segment.
    sup.stop(grace_s=0.1)
    assert not shm_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    # Extra cleanup helper check
    shm_file.write_bytes(b"stale")
    removed = cleanup_shm_paths([shm_name], shm_dir=fake_shm_dir)
    assert removed == [shm_file]
    assert not shm_file.exists()
