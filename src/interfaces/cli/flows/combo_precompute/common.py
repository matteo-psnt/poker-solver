"""Shared helpers for combo abstraction CLI flows."""

import json
from dataclasses import dataclass
from pathlib import Path

from questionary import Choice

from src.core.game.state import Card, Street
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext

# Board card counts for each street
BOARD_CARDS_BY_STREET: dict[Street, int] = {
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}

# Full deck as rank+suit strings, for random-sampling flows.
ALL_CARDS: list[str] = [r + s for r in "23456789TJQKA" for s in "hdcs"]


@dataclass(frozen=True)
class AbstractionEntry:
    """An on-disk combo abstraction directory with its parsed metadata."""

    path: Path
    metadata: dict

    @property
    def label(self) -> str:
        return f"{self.path.name} ({_get_config_name_from_metadata(self.metadata)})"


def _get_config_name_from_metadata(metadata: dict) -> str:
    """Extract config name from metadata JSON."""
    if "config" in metadata and isinstance(metadata["config"], dict):
        config_name = metadata["config"].get("config_name")
        if config_name:
            return config_name

    return "unknown"


def _list_existing_abstractions(base_path: Path) -> list[AbstractionEntry]:
    """Return all abstraction directories with their parsed metadata."""
    abstractions: list[AbstractionEntry] = []
    for path in base_path.iterdir():
        if path.is_dir() and (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
            abstractions.append(AbstractionEntry(path=path, metadata=metadata))
    return abstractions


def _select_abstraction(ctx: CliContext) -> AbstractionEntry | None:
    """Prompt the user to select an existing abstraction; None if cancelled or none found."""
    base_path = ctx.base_dir / "data" / "combo_abstraction"

    if not base_path.exists():
        print("\nNo combo abstractions found.")
        print("Run 'Precompute Combo Abstraction' to create one.")
        return None

    abstractions = _list_existing_abstractions(base_path)
    if not abstractions:
        print("\nNo combo abstractions found.")
        return None

    choices: list[Choice] = [Choice(title=entry.label, value=entry) for entry in abstractions]
    choices.append(Choice(title="Cancel", value=None))

    return prompts.select(
        ctx,
        "Select abstraction to examine:",
        choices=choices,
    )


def _parse_cards(card_str: str, expected: int) -> list:
    """Parse card string like 'AsKh' into list of Card objects."""
    card_str = card_str.replace(" ", "").replace(",", "")

    cards = []
    for i in range(0, len(card_str), 2):
        if i + 1 >= len(card_str):
            raise ValueError(f"Invalid card string: '{card_str}' (incomplete card)")

        rank = card_str[i].upper()
        suit = card_str[i + 1].lower()

        card = Card.new(rank + suit)
        cards.append(card)

    if len(cards) != expected:
        raise ValueError(f"Expected {expected} cards, got {len(cards)}")

    return cards
