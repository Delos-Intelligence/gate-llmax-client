"""`gate-llmax agent install` reading gateway credentials out of a project's env files."""

from pathlib import Path

from gate_llmax.agent.env_discovery import discover


def test_finds_prefixed_keys_in_a_nested_env_file(tmp_path: Path) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / ".env").write_text("COSMOS_GATE_BASE_URL=https://gate.example.com\nCOSMOS_GATE_API_KEY='secret'\n")

    found = discover(tmp_path)

    assert found is not None
    assert (found.base_url, found.api_key) == ("https://gate.example.com", "secret")
    assert found.source == backend / ".env"


def test_prefers_the_local_file_over_the_example(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("GATE_BASE_URL=\nGATE_API_KEY=\n")
    (tmp_path / ".env").write_text("GATE_BASE_URL=https://placeholder\nGATE_API_KEY=placeholder\n")
    (tmp_path / ".env.local").write_text("export GATE_BASE_URL=https://real\nGATE_API_KEY=real\n")

    found = discover(tmp_path)

    assert found is not None
    assert found.source == tmp_path / ".env.local"
    assert found.api_key == "real"


def test_ignores_half_a_pair_and_skipped_directories(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("# GATE_API_KEY=commented\nGATE_BASE_URL=https://gate.example.com\n")
    vendored = tmp_path / "node_modules"
    vendored.mkdir()
    (vendored / ".env").write_text("GATE_BASE_URL=https://vendored\nGATE_API_KEY=vendored\n")

    assert discover(tmp_path) is None
