"""Chart API service layer."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from src.engine.solver.protocols import Blueprint
from src.interfaces.chart.data import (
    build_chart_metadata,
    parse_situation,
    render_preflop_chart,
)
from src.pipeline.evaluation.preflop_chart import preflop_chart_data, preflop_open_sizes_bb
from src.pipeline.training import services
from src.pipeline.training.components import build_evaluation_solver


class ChartService:
    """Cached chart payload provider for API and viewer integrations."""

    def __init__(self, run_id: str, blueprint: Blueprint):
        self.run_id = run_id
        self.blueprint = blueprint
        self._cache: OrderedDict[tuple[int, str], dict] = OrderedDict()
        self._cache_max_size = 256

    @classmethod
    def from_run_dir(cls, run_dir: Path, run_id: str | None = None) -> ChartService:
        metadata = services.load_run_metadata(run_dir)
        solver, _ = build_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
        return cls(run_id=run_id or run_dir.name, blueprint=solver)

    def num_infosets(self) -> int:
        return self.blueprint.storage.num_infosets()

    def get_metadata(self) -> dict:
        return build_chart_metadata(self.run_id, preflop_open_sizes_bb(self.blueprint))

    def get_chart(self, position: int, situation_id: str) -> dict:
        key = (position, situation_id)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        chart = preflop_chart_data(self.blueprint, position, parse_situation(situation_id))
        payload = render_preflop_chart(
            chart, run_id=self.run_id, position=position, situation_id=situation_id
        )
        self._cache[key] = payload
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return payload
