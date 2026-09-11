"""Which inference backend to use, and why.

The choice is not a preference. Docker Desktop on macOS runs a Linux VM with
no Metal passthrough, so an ollama *container* on a Mac is CPU-only however
much GPU the machine has. A host ollama therefore beats a bundled one on
Apple silicon by a wide margin, and the bundled container is the fallback
rather than the default.

`decide` takes the probe results as arguments instead of running them, so
every branch is tested without a Mac, an NVIDIA card or a running daemon.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str  # "metal" | "cuda" | "cpu"
    host: str
    reason: str


@dataclass(frozen=True)
class Probe:
    system: str  # platform.system()
    machine: str  # platform.machine()
    host_ollama: bool
    nvidia: bool
    host_url: str = "http://localhost:11434"
    bundled_url: str = "http://ollama:11434"


def decide(probe: Probe) -> Backend:
    if probe.host_ollama and probe.system == "Darwin" and probe.machine in {"arm64", "aarch64"}:
        return Backend(
            "metal",
            probe.host_url,
            "ollama is running on this Mac; using it directly for Metal acceleration "
            "(a container on macOS gets no GPU at all)",
        )
    if probe.host_ollama and probe.nvidia:
        return Backend("cuda", probe.host_url, "ollama is running on the host with an NVIDIA GPU")
    if probe.nvidia:
        return Backend(
            "cuda", probe.bundled_url, "no host ollama; the bundled container gets the NVIDIA GPU"
        )
    if probe.host_ollama:
        return Backend("cpu", probe.host_url, "ollama is running on the host, without a GPU")
    return Backend(
        "cpu",
        probe.bundled_url,
        "no GPU and no host ollama; the bundled container will run on CPU and will be slow",
    )


def nvidia_present() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],  # noqa: S603, S607 - fixed argv, resolved by which()
            capture_output=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform dependent
        return False


def host_ollama_present(url: str) -> bool:
    from .errors import BackendUnavailable
    from .ollama import version

    try:
        version(url, timeout=2.0)
    except BackendUnavailable:
        return False
    return True


def detect(  # pragma: no cover - probes the real machine
    host_url: str | None = None, bundled_url: str | None = None
) -> Backend:
    host_url = host_url or os.environ.get("HOST_OLLAMA", "http://localhost:11434")
    return decide(
        Probe(
            system=platform.system(),
            machine=platform.machine(),
            host_ollama=host_ollama_present(host_url),
            nvidia=nvidia_present(),
            host_url=host_url,
            bundled_url=bundled_url or os.environ.get("BUNDLED_OLLAMA", "http://ollama:11434"),
        )
    )
