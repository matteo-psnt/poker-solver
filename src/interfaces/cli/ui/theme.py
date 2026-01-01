"""Shared CLI styling."""

from questionary import Style

STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:#6C6C6C"),
        ("instruction", ""),
        ("text", ""),
    ]
)
