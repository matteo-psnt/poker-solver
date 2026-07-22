"""Tests for the shared run-selector choice logic."""

from questionary import Choice, Separator

from src.interfaces.cli.flows import run_picker
from src.pipeline.services import RunSummary


def _summary(name, *, commits_ago, loadable, blocker=None, dirty=False) -> RunSummary:
    return RunSummary(
        name=name,
        commits_ago=commits_ago,
        git_dirty=dirty,
        representation_version=3,
        current_version=3,
        has_checkpoint=loadable,
        loadable=loadable,
        blocker=blocker,
        iterations=6000,
        num_infosets=43041,
        config_name="quick_test",
        status="completed",
    )


def _values(choices) -> list[object]:
    return [c.value for c in choices if isinstance(c, Choice)]


def _by_value(choices, value) -> Choice:
    return next(c for c in choices if isinstance(c, Choice) and c.value == value)


def test_age_formatting():
    assert run_picker._format_age(_summary("r", commits_ago=0, loadable=True)) == "HEAD"
    assert run_picker._format_age(_summary("r", commits_ago=1, loadable=True)) == "1 commit ago"
    assert run_picker._format_age(_summary("r", commits_ago=7, loadable=True)) == "7 commits ago"
    assert (
        run_picker._format_age(_summary("r", commits_ago=None, loadable=True)) == "commit unknown"
    )
    dirty = run_picker._format_age(_summary("r", commits_ago=2, loadable=True, dirty=True))
    assert dirty.endswith("· dirty")


def test_title_shows_run_stats():
    title = run_picker.run_title(
        _summary("run-x", commits_ago=5, loadable=True), note_blocker=False
    )
    assert "6k it" in title
    assert "43k infosets" in title
    assert "quick_test" in title


def test_human_count_scales():
    assert run_picker._human_count(6000) == "6k"
    assert run_picker._human_count(10_600_000) == "10.6M"
    assert run_picker._human_count(None) is None


def test_collapsed_hides_blocked_behind_toggle():
    summaries = [
        _summary("run-new", commits_ago=0, loadable=True),
        _summary("run-old", commits_ago=5, loadable=False, blocker="no checkpoint"),
    ]
    choices = run_picker.build_choices(
        summaries, show_all=False, cancel_label="Cancel", allow_unloadable=False
    )
    values = _values(choices)

    assert "run-new" in values  # loadable run is selectable
    assert "run-old" not in values  # blocked run hidden while collapsed
    assert run_picker._SHOW_ALL in values  # toggle offered
    assert "Cancel" in values


def test_expanded_greys_out_blocked_runs():
    summaries = [
        _summary("run-new", commits_ago=0, loadable=True),
        _summary("run-old", commits_ago=5, loadable=False, blocker="no checkpoint"),
    ]
    choices = run_picker.build_choices(
        summaries, show_all=True, cancel_label="Cancel", allow_unloadable=False
    )

    blocked = _by_value(choices, "run-old")
    assert blocked.disabled == "no checkpoint"  # greyed with reason
    assert run_picker._HIDE in _values(choices)  # toggle flips to hide


def test_allow_unloadable_keeps_everything_selectable():
    summaries = [
        _summary("run-new", commits_ago=0, loadable=True),
        _summary("run-old", commits_ago=5, loadable=False, blocker="no checkpoint"),
    ]
    choices = run_picker.build_choices(
        summaries, show_all=False, cancel_label="Back", allow_unloadable=True
    )

    old = _by_value(choices, "run-old")
    assert old.disabled is None  # selectable for inspection
    assert old.title is not None and "no checkpoint" in old.title  # reason surfaced
    assert run_picker._SHOW_ALL not in _values(choices)  # no toggle in this mode
    assert "Back" in _values(choices)


def test_separator_present_but_not_a_value():
    summaries = [_summary("run-new", commits_ago=0, loadable=True)]
    choices = run_picker.build_choices(
        summaries, show_all=False, cancel_label="Cancel", allow_unloadable=False
    )
    assert any(isinstance(c, Separator) for c in choices)
