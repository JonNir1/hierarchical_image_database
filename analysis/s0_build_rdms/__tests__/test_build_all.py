"""Tests for analysis.s0_build_rdms.build_all: the --validate flag."""
from __future__ import annotations

import pytest

import analysis.s0_build_rdms.build_all as ba
import analysis.s0_build_rdms.validate_rdms as vr


@pytest.fixture()
def two_builders(monkeypatch):
    """Replace the real builders with one that succeeds and one that raises."""
    def ok() -> None:
        return None

    def boom() -> None:
        raise RuntimeError("builder broke")

    monkeypatch.setattr(ba, "_BUILDERS", {
        "good": (__name__, "ok", [], {}), "bad": (__name__, "boom", [], {}),
    })
    monkeypatch.setattr(ba, "_ALL_NAMES", ["good", "bad"])
    globals().update(ok=ok, boom=boom)


def _run(monkeypatch, argv: list[str], validate_returns: bool = True) -> list[list[str]]:
    seen: list[list[str]] = []
    monkeypatch.setattr(vr, "run_all_checks", lambda names: seen.append(list(names)) or validate_returns)
    monkeypatch.setattr("sys.argv", ["build_all", *argv])
    return seen


def test_validate_runs_only_on_built_targets_and_sets_exit_code(two_builders, monkeypatch):
    seen = _run(monkeypatch, ["--validate"])
    with pytest.raises(SystemExit) as failed_build:
        ba.main()
    assert seen == [["good"]] and failed_build.value.code == 1

    seen = _run(monkeypatch, ["--only", "good", "--validate"], validate_returns=False)
    with pytest.raises(SystemExit) as failed_validation:
        ba.main()
    assert seen == [["good"]] and failed_validation.value.code == 1

    seen = _run(monkeypatch, ["--only", "good"])
    ba.main()  # no --validate: checks are not run, and a clean build does not exit
    assert seen == []
