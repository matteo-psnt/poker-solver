"""Pydantic models for the chart API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


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


class NewHandRequest(BaseModel):
    """Optional seat/button overrides for a new hand; both randomized when unset."""

    human_seat: int | None = Field(default=None, ge=0, le=1, alias="humanSeat")
    button: int | None = Field(default=None, ge=0, le=1)

    model_config = ConfigDict(populate_by_name=True)


class ActionRequest(BaseModel):
    """The human's chosen action, referenced by its id in the current legal set."""

    action_id: int = Field(ge=0, alias="actionId")

    model_config = ConfigDict(populate_by_name=True)
