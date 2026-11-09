import operator
from typing import Annotated, TypedDict


class LabState(TypedDict):
    data_summary: str
    metric: str
    best_code: str
    best_score: float
    ideas: list[str]
    results: Annotated[list[dict], operator.add]
    round: int


class WorkerInput(TypedDict):
    experiment_id: str
    idea: str
    base_code: str
    data_summary: str
    metric: str
    round: int