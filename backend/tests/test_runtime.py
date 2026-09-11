from __future__ import annotations

import pytest

from rag_assistant.runtime import Probe, decide

MAC = {"system": "Darwin", "machine": "arm64"}
LINUX = {"system": "Linux", "machine": "x86_64"}


def test_a_mac_with_a_host_ollama_uses_metal():
    backend = decide(Probe(**MAC, host_ollama=True, nvidia=False))
    assert backend.name == "metal"
    assert backend.host == "http://localhost:11434"


def test_a_mac_without_a_host_ollama_falls_back_to_the_cpu_container():
    # Docker Desktop on macOS has no Metal passthrough, so a bundled ollama on
    # a Mac is CPU-only however much GPU the machine has.
    backend = decide(Probe(**MAC, host_ollama=False, nvidia=False))
    assert backend.name == "cpu"
    assert backend.host == "http://ollama:11434"


def test_an_intel_mac_does_not_claim_metal():
    backend = decide(Probe(system="Darwin", machine="x86_64", host_ollama=True, nvidia=False))
    assert backend.name == "cpu"


def test_a_linux_host_with_an_nvidia_gpu_and_host_ollama_uses_it():
    backend = decide(Probe(**LINUX, host_ollama=True, nvidia=True))
    assert backend.name == "cuda"
    assert backend.host == "http://localhost:11434"


def test_a_linux_box_with_a_gpu_and_no_host_ollama_uses_the_container():
    backend = decide(Probe(**LINUX, host_ollama=False, nvidia=True))
    assert backend.name == "cuda"
    assert backend.host == "http://ollama:11434"


def test_a_plain_linux_box_lands_on_cpu_and_says_it_will_be_slow():
    backend = decide(Probe(**LINUX, host_ollama=False, nvidia=False))
    assert backend.name == "cpu"
    assert "slow" in backend.reason


def test_a_host_ollama_without_a_gpu_is_still_preferred_over_a_container():
    backend = decide(Probe(**LINUX, host_ollama=True, nvidia=False))
    assert backend.host == "http://localhost:11434"


@pytest.mark.parametrize("machine", ["arm64", "aarch64"])
def test_both_spellings_of_apple_silicon_are_recognised(machine):
    assert (
        decide(Probe(system="Darwin", machine=machine, host_ollama=True, nvidia=False)).name
        == "metal"
    )


def test_every_decision_explains_itself():
    for probe in [
        Probe(**MAC, host_ollama=True, nvidia=False),
        Probe(**LINUX, host_ollama=True, nvidia=True),
        Probe(**LINUX, host_ollama=False, nvidia=True),
        Probe(**LINUX, host_ollama=True, nvidia=False),
        Probe(**LINUX, host_ollama=False, nvidia=False),
    ]:
        assert len(decide(probe).reason) > 20
