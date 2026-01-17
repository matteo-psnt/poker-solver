from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import Config


@dataclass
class RunMetadata:
    run_id: str
    config_name: str
    started_at: str
    resumed_at: str | None
    completed_at: str | None
    status: str
    iterations: int
    runtime_seconds: float
    num_infosets: int
    storage_capacity: int
    action_config_hash: str
    config: Config

    @classmethod
    def new(
        cls,
        run_id: str,
        config_name: str,
        config: Config,
        action_config_hash: str,
    ) -> "RunMetadata":
        storage_capacity = config.storage.initial_capacity if config else 0
        return cls(
            run_id=run_id,
            config_name=config_name,
            started_at=datetime.now().isoformat(),
            resumed_at=None,
            completed_at=None,
            status="running",
            iterations=0,
            runtime_seconds=0.0,
            num_infosets=0,
            storage_capacity=storage_capacity,
            action_config_hash=action_config_hash,
            config=config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        config_dict = data.get("config")
        if not isinstance(config_dict, dict) or not config_dict:
            raise ValueError("Run metadata missing required config")
        action_config_hash = data.get("action_config_hash")
        if not isinstance(action_config_hash, str) or not action_config_hash:
            raise ValueError("Run metadata missing required action_config_hash")
        config = Config.from_dict(config_dict)
        return cls(
            run_id=data.get("run_id", ""),
            config_name=data.get("config_name", "default"),
            started_at=data.get("started_at", ""),
            resumed_at=data.get("resumed_at"),
            completed_at=data.get("completed_at"),
            status=data.get("status", "unknown"),
            iterations=int(data.get("iterations", 0)),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
            num_infosets=int(data.get("num_infosets", 0)),
            storage_capacity=int(data.get("storage_capacity", 0)),
            action_config_hash=action_config_hash,
            config=config,
        )

    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        config_dict = self.config.to_dict()
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "started_at": self.started_at,
            "resumed_at": self.resumed_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "num_infosets": self.num_infosets,
            "storage_capacity": self.storage_capacity,
            "action_config_hash": self.action_config_hash,
            "config": config_dict,
        }

    def update_progress(
        self,
        iterations: int,
        runtime_seconds: float,
        num_infosets: int,
        storage_capacity: int,
    ) -> None:
        self.iterations = iterations
        self.runtime_seconds = runtime_seconds
        self.num_infosets = num_infosets
        self.storage_capacity = storage_capacity

    def resolve_initial_capacity(self, default_capacity: int) -> int:
        """Return stored capacity if present, otherwise a default."""
        return self.storage_capacity or default_capacity

    def mark_resumed(self) -> None:
        self.resumed_at = datetime.now().isoformat()
        self.status = "running"

    def mark_completed(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.completed_at = datetime.now().isoformat()

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
