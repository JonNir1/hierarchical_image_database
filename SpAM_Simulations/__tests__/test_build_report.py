"""Tests for the report writer's overwrite guard.

The generated report is a starting point that gets hand-edited afterwards, so a later build would
silently destroy that work. These pin the guard's decisions: refuse by default, allow when the file
is provably still ours, and never refuse so late that the caller has already paid for the work.
"""
import pytest

from SpAM_Simulations.reporting.build_report import (
    ReportEditedError, guard_output, write_report,
)


def test_a_fresh_target_is_written_without_complaint(tmp_path):
    out = tmp_path / "report.html"
    write_report(out, "<h1>first</h1>")
    assert out.read_text(encoding="utf-8") == "<h1>first</h1>"


def test_rewriting_an_untouched_report_is_allowed(tmp_path):
    """Regenerating is the normal path and must not need --force."""
    out = tmp_path / "report.html"
    write_report(out, "<h1>first</h1>")
    write_report(out, "<h1>second</h1>")
    assert out.read_text(encoding="utf-8") == "<h1>second</h1>"


def test_a_hand_edited_report_is_never_overwritten(tmp_path):
    """The case this exists for: someone edited the HTML and a rebuild would erase it."""
    out = tmp_path / "report.html"
    write_report(out, "<h1>generated</h1>")
    out.write_text("<h1>generated</h1><p>a paragraph I wrote by hand</p>", encoding="utf-8")
    with pytest.raises(ReportEditedError, match="contents have changed"):
        write_report(out, "<h1>regenerated</h1>")
    assert "by hand" in out.read_text(encoding="utf-8"), "the edit must survive the refusal"


def test_a_report_with_no_stamp_is_never_overwritten(tmp_path):
    """A file written before the guard existed cannot be proven to be ours, so assume it is not."""
    out = tmp_path / "report.html"
    out.write_text("<h1>from an older version</h1>", encoding="utf-8")
    with pytest.raises(ReportEditedError, match="no generation stamp"):
        write_report(out, "<h1>new</h1>")
    assert "older version" in out.read_text(encoding="utf-8")


def test_force_is_the_deliberate_escape_hatch(tmp_path):
    out = tmp_path / "report.html"
    out.write_text("<h1>hand written</h1>", encoding="utf-8")
    write_report(out, "<h1>forced</h1>", force=True)
    assert out.read_text(encoding="utf-8") == "<h1>forced</h1>"


def test_forcing_restamps_so_the_next_build_is_clean(tmp_path):
    """After an intentional overwrite the file is ours again, so the guard must stop objecting."""
    out = tmp_path / "report.html"
    out.write_text("<h1>hand written</h1>", encoding="utf-8")
    write_report(out, "<h1>forced</h1>", force=True)
    write_report(out, "<h1>next</h1>")          # no force needed
    assert out.read_text(encoding="utf-8") == "<h1>next</h1>"


def test_guard_is_callable_before_the_page_is_built(tmp_path):
    """`main` checks the target first so a refusal costs nothing and appears at the top of output.

    Guarding only inside `write_report` would mean building the whole page - loading every table,
    rendering every figure - before discovering the write was never permitted.
    """
    out = tmp_path / "report.html"
    out.write_text("<h1>hand written</h1>", encoding="utf-8")
    with pytest.raises(ReportEditedError):
        guard_output(out)
    guard_output(out, force=True)               # must not raise


def test_the_stamp_does_not_sit_next_to_the_report_as_clutter(tmp_path):
    """It is a dotfile so a directory listing of a run still shows only real artifacts."""
    out = tmp_path / "report.html"
    write_report(out, "<h1>x</h1>")
    visible = [p.name for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert visible == ["report.html"]
