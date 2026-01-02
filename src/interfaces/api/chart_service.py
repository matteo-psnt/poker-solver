"""Chart API service layer."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from src.interfaces.chart.data import (
    ChartDataRuntime,
    build_chart_metadata,
    build_preflop_chart_data,
)
from src.pipeline.training import services
from src.pipeline.training.components import build_evaluation_solver


class ChartService:
    """Cached chart payload provider for API and viewer integrations."""

    def __init__(self, run_id: str, runtime: ChartDataRuntime):
        self.run_id = run_id
        self.runtime = runtime
        self._cache: OrderedDict[tuple[int, str], dict] = OrderedDict()
        self._cache_max_size = 256

    @classmethod
    def from_run_dir(cls, run_dir: Path, run_id: str | None = None) -> ChartService:
        metadata = services.load_run_metadata(run_dir)
        config = metadata.config

        solver, storage = build_evaluation_solver(
            config,
            checkpoint_dir=run_dir,
        )

        runtime = ChartDataRuntime(
            action_model=solver.action_model,
            rules=solver.rules,
            storage=storage,
            starting_stack=config.game.starting_stack,
        )
        return cls(run_id=run_id or run_dir.name, runtime=runtime)

    def get_metadata(self) -> dict:
        return build_chart_metadata(self.run_id, self.runtime.action_model)

    def get_chart(self, position: int, situation_id: str) -> dict:
        key = (position, situation_id)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        payload = build_preflop_chart_data(self.runtime, position, situation_id, self.run_id)
        self._cache[key] = payload
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return payload
