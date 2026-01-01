"""Shared helpers for combo abstraction CLI flows."""

import json
from pathlib import Path

from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Card, Street

# Board card counts for each street
BOARD_CARDS_BY_STREET: dict[Street, int] = {
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}


def _get_config_name_from_metadata(metadata: dict) -> str:
    """Extract config name from metadata JSON."""
    if "config" in metadata and isinstance(metadata["config"], dict):
        config_name = metadata["config"].get("config_name")
        if config_name:
            return config_name

    return "unknown"


def _list_existing_abstractions(base_path: Path) -> list[tuple[Path, dict]]:
    """Return all abstraction directories and parsed metadata."""
    abstractions: list[tuple[Path, dict]] = []
    for path in base_path.iterdir():
        if path.is_dir() and (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
            abstractions.append((path, metadata))
    return abstractions


def _select_abstraction(ctx: CliContext) -> tuple:
    """
    Prompt user to select an existing abstraction.

    Returns:
        (path, metadata) or (None, None) if cancelled or none found
    """
    base_path = ctx.base_dir / "data" / "combo_abstraction"

    if not base_path.exists():
        print("\nNo combo abstractions found.")
        print("Run 'Precompute Combo Abstraction' to create one.")
        return None, None

    abstractions = _list_existing_abstractions(base_path)
    if not abstractions:
        print("\nNo combo abstractions found.")
        return None, None

    choices = []
    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        choices.append(f"{path.name} ({config_name})")

    choices.append("Cancel")

    choice = prompts.select(
        ctx,
        "Select abstraction to examine:",
        choices=choices,
    )

    if choice is None or choice == "Cancel":
        return None, None

    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        if f"{path.name} ({config_name})" == choice:
            return path, metadata

    return None, None


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
