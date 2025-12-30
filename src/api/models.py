"""Pydantic models for the chart API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ChartQuery(BaseModel):
    position: int = Field(default=0, ge=0, le=1)
    situation: str = Field(default="first_to_act", min_length=1)


class MetaResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    positions: list[dict[str, Any]]
    situations: list[dict[str, Any]]
    default_position: int = Field(alias="defaultPosition")
    default_situation: str = Field(alias="defaultSituation")


class ChartResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    position: int
    situation: str
    position_label: str = Field(alias="positionLabel")
    situation_label: str = Field(alias="situationLabel")
    betting_sequence: str = Field(alias="bettingSequence")
    ranks: str
    actions: list[dict[str, Any]]
    grid: list[list[dict[str, Any]]]
