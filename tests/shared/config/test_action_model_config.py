"""What `ActionModelConfig` refuses, and the message it refuses with.

The validator guards every config load in the project and had no test at all,
so its four jobs -- required keys, referential integrity between rules and
templates, preflop token grammar, postflop token grammar -- were free to drift.
These pin the behaviour, message text included: the messages name the offending
key, and a config error that does not say which key is wrong costs a round of
guessing at the one moment a run is being set up.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.config import ActionModelConfig

PREFLOP: dict[str, list[float | str]] = {
    "sb_first_in": ["fold", "call", 2.5],
    "bb_vs_limp": ["check", 4.0],
    "sb_vs_limp_raise": ["fold", "call", "2.3x_last", "jam"],
    "bb_vs_open": ["fold", "call"],
    "sb_vs_3bet": ["fold", "call"],
    "bb_vs_4bet": ["fold", "call", "jam"],
    "sb_vs_5bet": ["fold", "call"],
}
POSTFLOP: dict[str, list[float | str]] = {
    "first_aggressive": [0.33],
    "facing_bet": ["min_raise", "jam"],
    "after_one_raise": ["pot_raise", "jam"],
    "after_two_raises": ["jam"],
}
RULES = {
    "facing_1": "facing_bet",
    "facing_2": "after_one_raise",
    "facing_3_plus": "after_two_raises",
}


def _build(
    *,
    preflop_templates: dict[str, list[float | str]] | None = None,
    postflop_templates: dict[str, list[float | str]] | None = None,
    raise_count_rules: dict[str, str] | None = None,
) -> ActionModelConfig:
    """A valid action model, with one part swapped out per test."""
    return ActionModelConfig(
        preflop_templates=dict(PREFLOP) if preflop_templates is None else preflop_templates,
        postflop_templates=dict(POSTFLOP) if postflop_templates is None else postflop_templates,
        raise_count_rules=dict(RULES) if raise_count_rules is None else raise_count_rules,
    )


def test_the_defaults_validate():
    """The default action model is itself a valid one."""
    assert ActionModelConfig().version == 1


def test_an_explicit_but_valid_model_validates():
    assert _build().off_tree_mapping == "probabilistic"


class TestRequiredKeys:
    def test_a_missing_preflop_template_names_it(self):
        incomplete = {k: v for k, v in PREFLOP.items() if k != "bb_vs_open"}
        with pytest.raises(ValidationError, match=r"preflop_templates missing required keys"):
            _build(preflop_templates=incomplete)

    def test_a_missing_postflop_template_names_it(self):
        incomplete = {k: v for k, v in POSTFLOP.items() if k != "facing_bet"}
        with pytest.raises(ValidationError, match=r"postflop_templates missing required keys"):
            _build(postflop_templates=incomplete)

    def test_a_missing_rule_key_names_it(self):
        with pytest.raises(ValidationError, match=r"raise_count_rules missing required keys"):
            _build(raise_count_rules={"facing_1": "facing_bet"})

    def test_an_unknown_rule_key_is_refused(self):
        with pytest.raises(ValidationError, match=r"raise_count_rules has unknown keys"):
            _build(raise_count_rules={**RULES, "facing_9": "facing_bet"})


class TestRulesPointAtRealTemplates:
    def test_a_rule_naming_no_template_is_refused(self):
        with pytest.raises(ValidationError, match=r"points to unknown postflop template"):
            _build(raise_count_rules={**RULES, "facing_1": "no_such_template"})


class TestPreflopTokens:
    def test_an_empty_template_is_refused(self):
        with pytest.raises(ValidationError, match=r"must not be empty"):
            _build(preflop_templates={**PREFLOP, "bb_vs_open": []})

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(ValidationError, match=r"numeric sizes must be > 0"):
            _build(preflop_templates={**PREFLOP, "bb_vs_open": ["fold", 0.0]})

    def test_passive_and_jam_words_are_accepted(self):
        assert _build(preflop_templates={**PREFLOP, "bb_vs_open": ["fold", "call", "jam"]})

    def test_a_word_carrying_no_multiplier_is_refused(self):
        """`parse_multiplier_token` answers None for a token with no suffix."""
        with pytest.raises(ValidationError, match=r"Invalid preflop token 'wat'"):
            _build(preflop_templates={**PREFLOP, "bb_vs_open": ["wat"]})

    def test_a_multiplier_suffix_with_a_bad_prefix_is_refused(self):
        """The other branch: the suffix is there, so float() raises rather than
        returning None. Both spellings must reach the same refusal."""
        with pytest.raises(ValidationError, match=r"Invalid preflop token 'xx_open'"):
            _build(preflop_templates={**PREFLOP, "bb_vs_open": ["xx_open"]})

    def test_a_non_positive_multiplier_is_refused(self):
        with pytest.raises(ValidationError, match=r"Invalid preflop token '0x_open'"):
            _build(preflop_templates={**PREFLOP, "bb_vs_open": ["0x_open"]})

    def test_a_valid_multiplier_is_accepted(self):
        assert _build(preflop_templates={**PREFLOP, "bb_vs_open": ["fold", "2.5x_open"]})


class TestPostflopTokens:
    def test_an_empty_template_is_refused(self):
        with pytest.raises(ValidationError, match=r"must not be empty"):
            _build(postflop_templates={**POSTFLOP, "after_two_raises": []})

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(ValidationError, match=r"numeric sizes must be > 0"):
            _build(postflop_templates={**POSTFLOP, "first_aggressive": [-1.0]})

    def test_an_unknown_word_is_refused(self):
        """Postflop takes a closed vocabulary, not preflop's multipliers."""
        with pytest.raises(ValidationError, match=r"Invalid postflop token '2.5x'"):
            _build(postflop_templates={**POSTFLOP, "first_aggressive": ["2.5x"]})

    def test_the_vocabulary_is_case_and_space_insensitive(self):
        assert _build(postflop_templates={**POSTFLOP, "after_two_raises": ["  JAM "]})
