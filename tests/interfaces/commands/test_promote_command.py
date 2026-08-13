"""Promotion refuses with a cause, not a guess.

`_publish` used to answer a bare ``False``, so the handler's advice -- "check
`az login` and Terraform state" -- was a guess at a reason it had just thrown
away. Unreachable share, bad credential and missing container all arrived
looking identical, at the one moment the lineage is being moved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.interfaces.commands import promote

if TYPE_CHECKING:
    from pathlib import Path


class TestPublishReportsWhyItFailed:
    def test_the_exception_type_and_message_survive(self, monkeypatch, tmp_path: Path):
        def _no_credential():
            raise RuntimeError("DefaultAzureCredential failed to retrieve a token")

        monkeypatch.setattr(promote.CloudConfig, "load", staticmethod(_no_credential))
        local = tmp_path / "baseline.json"
        local.write_text("{}")

        refused = promote._publish(local)

        assert refused is not None
        assert "RuntimeError" in refused
        assert "failed to retrieve a token" in refused

    def test_success_is_the_absence_of_a_reason(self, monkeypatch, tmp_path: Path):
        written: dict[str, str] = {}
        monkeypatch.setattr(
            promote.CloudConfig, "load", staticmethod(lambda: _Config("share-name"))
        )
        monkeypatch.setattr(promote, "share_client", lambda _config: object())
        monkeypatch.setattr(
            promote,
            "write_baseline",
            lambda _client, share, text: written.update(share=share, text=text),
        )
        local = tmp_path / "baseline.json"
        local.write_text('{"run_id": "run-a"}')

        assert promote._publish(local) is None
        assert written == {"share": "share-name", "text": '{"run_id": "run-a"}'}


class _Config:
    """The two attributes `_publish` reads off a loaded CloudConfig."""

    def __init__(self, share_name: str) -> None:
        self.share_name = share_name
