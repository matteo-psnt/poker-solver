"""Chart API service layer."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from src.actions.betting_actions import BettingActions
from src.cli.flows.chart.data import build_chart_metadata, build_preflop_chart_data
from src.game.rules import GameRules
from src.solver.storage.in_memory import InMemoryStorage
from src.training import services


@dataclass(frozen=True)
class ChartDataRuntime:
    """Minimal runtime state required to render preflop chart data."""

    action_abstraction: BettingActions
    rules: GameRules
    storage: InMemoryStorage
    starting_stack: int


class ChartService:
    """Cached chart payload provider for API and viewer integrations."""

    def __init__(self, run_id: str, runtime: ChartDataRuntime):
        self.run_id = run_id
        self.runtime = runtime
        self._cache: "OrderedDict[tuple[int, str], dict]" = OrderedDict()
        self._cache_max_size = 256

    @classmethod
    def from_run_dir(cls, run_dir: Path, run_id: str | None = None) -> "ChartService":
        metadata = services.load_run_metadata(run_dir)
        config = metadata.config

        action_abstraction = BettingActions(
            config.action_abstraction,
            big_blind=config.game.big_blind,
        )
        rules = GameRules(config.game.small_blind, config.game.big_blind)
        storage = InMemoryStorage(checkpoint_dir=run_dir)

        runtime = ChartDataRuntime(
            action_abstraction=action_abstraction,
            rules=rules,
            storage=storage,
            starting_stack=config.game.starting_stack,
        )
        return cls(run_id=run_id or run_dir.name, runtime=runtime)

    def get_metadata(self) -> dict:
        return build_chart_metadata(self.run_id, self.runtime.action_abstraction)

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
