"""Resolution and loading of precomputed combo abstractions."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


@dataclass(frozen=True, slots=True)
class _AbstractionCandidate:
    path: Path
    config_hash: str | None


def _read_metadata(path: Path) -> dict | None:
    """Read abstraction metadata.json; return None for unreadable files."""
    metadata_file = path / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file) as f:
            return json.load(f)
    except Exception:
        return None


class ComboAbstractionResolver:
    """Resolves configured abstraction references into concrete filesystem paths."""

    def __init__(
        self,
        abstractions_dir: Path | None = None,
        loader: Callable[[Path], BucketingStrategy] | None = None,
    ):
        self.abstractions_dir = abstractions_dir or Path("data/combo_abstraction")
        self._loader = loader or PostflopPrecomputer.load

    def load(
        self, *, abstraction_config: str, abstraction_hash: str | None = None
    ) -> BucketingStrategy:
        """Load abstraction by config name, optionally pinned to an exact config hash.

        A config name is a mutable pointer: recomputing an abstraction under changed
        parameters reuses the name but produces different buckets. ``abstraction_hash``
        pins resolution to the exact abstraction a checkpoint was trained against, so
        evaluating that checkpoint cannot silently bucket hands under a newer
        abstraction whose bucket ids mean something else.
        """
        resolved_path = self._resolve_config_path(abstraction_config, abstraction_hash)
        return self._loader(resolved_path)

    def resolved_hash(
        self, *, abstraction_config: str, abstraction_hash: str | None = None
    ) -> str | None:
        """Config hash of the abstraction this name currently resolves to.

        Recorded on a run at training time so the run can later be evaluated under the
        exact abstraction it was trained against, rather than whatever the (mutable)
        config name happens to point at by then.
        """
        resolved_path = self._resolve_config_path(abstraction_config, abstraction_hash)
        metadata = _read_metadata(resolved_path) or {}
        config_hash = metadata.get("config_hash")
        return config_hash if isinstance(config_hash, str) else None

    def _resolve_config_path(self, config_name: str, abstraction_hash: str | None = None) -> Path:
        if abstraction_hash is not None:
            expected_hash = abstraction_hash
        else:
            # Validate config file early so users get actionable errors.
            expected_config = PrecomputeConfig.from_yaml(config_name)
            expected_hash = expected_config.get_config_hash()

        if not self.abstractions_dir.exists():
            raise FileNotFoundError(
                f"No combo abstractions found (directory doesn't exist: {self.abstractions_dir}).\n"
                f"Please run 'Precompute Combo Abstraction' from the CLI with config '{config_name}'.yaml first."
            )

        matching = self._find_candidates(config_name)
        if not matching:
            raise FileNotFoundError(
                f"No combo abstraction found for config '{config_name}'.\n"
                f"Please run 'Precompute Combo Abstraction' from the CLI first with {config_name}.yaml."
            )

        hash_matches = [
            candidate for candidate in matching if candidate.config_hash == expected_hash
        ]
        if len(hash_matches) == 1:
            return hash_matches[0].path

        if len(hash_matches) > 1:
            options = ", ".join(sorted(str(candidate.path) for candidate in hash_matches))
            raise ValueError(
                f"Multiple combo abstractions found for config '{config_name}' "
                f"with matching hash {expected_hash}:\n"
                f"  {options}\n"
                "Delete duplicate abstraction directories to disambiguate."
            )

        if abstraction_hash is not None:
            options = ", ".join(sorted(str(candidate.path) for candidate in matching))
            raise FileNotFoundError(
                f"No combo abstraction for config '{config_name}' with pinned hash "
                f"{expected_hash}.\n"
                f"  Available: {options}\n"
                "The abstraction this checkpoint was trained against is unavailable, so it "
                "cannot be evaluated faithfully. Recompute that abstraction or evaluate a "
                "run trained against an available one."
            )

        if len(matching) == 1:
            candidate = matching[0]
            if candidate.config_hash and candidate.config_hash != expected_hash:
                raise ValueError(
                    f"Card abstraction config hash mismatch for '{config_name}':\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Saved:    {candidate.config_hash}\n"
                    f"The saved abstraction was computed with different parameters.\n"
                    f"Please re-run 'Precompute Combo Abstraction' with the current config."
                )
            return candidate.path

        options = ", ".join(sorted(str(candidate.path) for candidate in matching))
        raise ValueError(
            f"Multiple combo abstractions found for config '{config_name}' "
            f"but none match expected hash {expected_hash}.\n"
            f"  {options}\n"
            "Recompute abstractions to match the current config."
        )

    def _find_candidates(self, config_name: str) -> list[_AbstractionCandidate]:
        matching: list[_AbstractionCandidate] = []
        for path in self.abstractions_dir.iterdir():
            if not path.is_dir():
                continue
            metadata = _read_metadata(path)
            if not metadata:
                continue
            config_data = metadata.get("config", {})
            if not isinstance(config_data, dict):
                continue
            # Match on the abstraction's name only. A precomputed abstraction is
            # identified by name here; the actual artifact is validated on load.
            # Requiring a full strict re-parse of the saved config would silently
            # skip abstractions whose metadata predates a schema change, surfacing
            # as a misleading "not found" instead of a real match.
            if config_data.get("config_name") != config_name:
                continue
            saved_hash = metadata.get("config_hash")
            matching.append(
                _AbstractionCandidate(
                    path=path,
                    config_hash=saved_hash if isinstance(saved_hash, str) else None,
                )
            )
        return matching
