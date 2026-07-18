"""Tests for chart presentation (grid naming and payload shape)."""

from src.core.game.actions import bet, call, fold
from src.interfaces.chart.data import parse_situation, render_preflop_chart
from src.pipeline.evaluation.preflop_chart import HandStrategy, PreflopChartData


def _chart(hands) -> PreflopChartData:
    return PreflopChartData(
        betting_sequence="",
        to_call=0,
        big_blind=2,
        applied_raise=False,
        hands=hands,
    )


def _cell(payload, hand):
    for row in payload["grid"]:
        for cell in row:
            if cell["hand"] == hand:
                return cell
    raise AssertionError(f"hand {hand} not in grid")


def test_parse_situation():
    assert parse_situation("first_to_act") is None
    assert parse_situation("facing_raise_2.5") == 2.5


def test_grid_cells_use_canonical_class_names():
    payload = render_preflop_chart(_chart({}), run_id="r", position=0, situation_id="first_to_act")

    cells = {cell["hand"] for row in payload["grid"] for cell in row}
    assert len(cells) == 169
    # Upper triangle suited, diagonal pairs, lower triangle offsuit (high rank first).
    assert {"AA", "AKs", "AKo", "32o"} <= cells
    assert payload["situationLabel"] == "First to act"
    assert payload["positionLabel"] == "Button (BTN)"


def test_trained_hand_renders_action_percentages():
    hands = {
        "AKs": HandStrategy(
            actions=(fold(), call(), bet(4)),
            probabilities=(0.25, 0.25, 0.5),
        )
    }
    payload = render_preflop_chart(
        _chart(hands), run_id="r", position=0, situation_id="first_to_act"
    )

    cell = _cell(payload, "AKs")
    by_id = {a["id"]: a["pct"] for a in cell["actions"]}
    assert by_id["fold"] == 25.0
    assert by_id["call"] == 25.0
    # bet(4) at big_blind=2 is a 2bb open.
    assert by_id["size_2.00"] == 50.0
    # Untrained cells render empty, and the action legend covers the seen actions.
    assert _cell(payload, "72o")["actions"] == []
    assert [a["id"] for a in payload["actions"]] == ["size_2.00", "call", "fold"]


def test_facing_raise_label_survives_fallback():
    payload = render_preflop_chart(
        _chart({}), run_id="r", position=1, situation_id="facing_raise_3"
    )
    assert payload["situationLabel"] == "Facing raise to 3bb"
