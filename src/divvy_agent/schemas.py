"""Strict Pydantic schemas shared by the Divvy analysis tools and the adaptive agent."""

from __future__ import annotations

import calendar
import json
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Metric(str, Enum):
    TRIP_COUNT = 'trip_count'
    AVERAGE_DURATION_MINUTES = 'average_duration_minutes'

class Dimension(str, Enum):
    START_MONTH = 'start_month'
    START_WEEKDAY = 'start_weekday'
    START_HOUR = 'start_hour'
    MEMBER_TYPE = 'member_casual'
    RIDEABLE_TYPE = 'rideable_type'
    START_STATION = 'start_station_name'
    END_STATION = 'end_station_name'

class FilterField(str, Enum):
    START_YEAR = 'start_year'
    START_MONTH = 'start_month'
    START_WEEKDAY = 'start_weekday'
    START_HOUR = 'start_hour'
    MEMBER_TYPE = 'member_casual'
    RIDEABLE_TYPE = 'rideable_type'
    START_STATION = 'start_station_name'
    END_STATION = 'end_station_name'

class FilterOperator(str, Enum):
    EQUALS = 'equals'
    IN = 'in'

class SortOrder(str, Enum):
    ASCENDING = 'ascending'
    DESCENDING = 'descending'

class ChartType(str, Enum):
    LINE = 'line'
    BAR = 'bar'

class DecisionAction(str, Enum):
    CONTINUE = 'continue'
    REPORT = 'report'
    CLARIFY = 'clarify'
    BLOCKED = 'blocked'
    FAILED = 'failed'

class TerminalStatus(str, Enum):
    COMPLETED = 'completed'
    CLARIFICATION_NEEDED = 'clarification_needed'
    BLOCKED = 'blocked'
    BUDGET_EXHAUSTED = 'budget_exhausted'
    FAILED = 'failed'

class AnalysisRequest(StrictModel):
    question: str = Field(min_length=5, max_length=1000)
    requested_output: Literal['table', 'chart', 'report'] = 'report'

class FilterSpec(StrictModel):
    field: FilterField
    operator: FilterOperator
    value: int | str | list[int] | list[str]

    @field_validator('value', mode='before')
    @classmethod
    def coerce_year_string(cls, value, info):
        # Models often send "2025" as a string; the year filter needs the integer.
        if info.data.get('field') == FilterField.START_YEAR:
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
            if isinstance(value, list):
                return [int(v) if isinstance(v, str) and v.strip().isdigit() else v for v in value]
        return value

    @model_validator(mode='after')
    def validate_operator_value(self):
        if self.operator == FilterOperator.IN and not isinstance(self.value, list):
            raise ValueError('The in operator requires a list value')
        if self.operator == FilterOperator.EQUALS and isinstance(self.value, list):
            raise ValueError('The equals operator requires a scalar value')
        if self.field == FilterField.START_YEAR:
            values = self.value if isinstance(self.value, list) else [self.value]
            if any(value != 2025 for value in values):
                raise ValueError('The frozen snapshot supports only start_year 2025')
        return self

class AnalysisPlan(StrictModel):
    objective: str = Field(min_length=5, max_length=500)
    metric: Metric
    group_by: list[Dimension] = Field(min_length=1, max_length=2)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=5)
    sort_order: SortOrder = SortOrder.ASCENDING
    limit: int | None = Field(default=None, ge=1, le=50)

    @field_validator('filters', mode='before')
    @classmethod
    def drop_empty_and_redundant_filters(cls, value):
        # An empty value filters nothing, and the snapshot holds only 2025, so a
        # start_year == 2025 filter is a no-op. Other years still fail validation.
        if not isinstance(value, list):
            return value
        kept = []
        for item in value:
            if isinstance(item, dict):
                raw = item.get('value')
                if raw is None or raw == '' or raw == []:
                    continue
                if item.get('field') == 'start_year' and str(raw).strip() == '2025':
                    continue
            kept.append(item)
        return kept

    @field_validator('group_by')
    @classmethod
    def dimensions_are_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError('group_by dimensions must be unique')
        return value

class Coverage(StrictModel):
    candidate_rows: int = Field(ge=0)
    included_rows: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    excluded_rate: float = Field(ge=0, le=1)
    eligibility_rule: str

class AnalysisResult(StrictModel):
    tool_name: Literal['analyze_data'] = 'analyze_data'
    plan: AnalysisPlan
    value_field: str
    rows: list[dict[str, int | float | str]]
    coverage: Coverage
    derived: dict[str, int | float | str] = Field(default_factory=dict)
    elapsed_ms: float = Field(ge=0)

class DatasetInspection(StrictModel):
    tool_name: Literal['inspect_dataset'] = 'inspect_dataset'
    row_count: int
    file_count: int
    fields: list[str]
    date_range: tuple[str, str]
    quality_counts: dict[str, int]
    supported_metrics: list[str]
    supported_dimensions: list[str]

class ChartRequest(StrictModel):
    chart_type: ChartType
    x: Dimension
    y: str
    series: Dimension | None = None
    title: str = Field(min_length=3, max_length=120)
    x_label: str = Field(min_length=1, max_length=60)
    y_label: str = Field(min_length=1, max_length=60)
    filename: str = Field(pattern=r'^[A-Za-z0-9_-]+\.png$')

class ChartResult(StrictModel):
    tool_name: Literal['create_visualization'] = 'create_visualization'
    path: str
    chart_type: ChartType
    plotted_rows: int = Field(ge=1)

class EvidenceClaim(StrictModel):
    key: str = Field(min_length=1)
    value: int | float | str
    unit: str

class ValidationResult(StrictModel):
    tool_name: Literal['validate_result'] = 'validate_result'
    valid: bool
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class AdaptiveDecision(StrictModel):
    action: DecisionAction
    observation: str = Field(min_length=3, max_length=500)
    reason: str = Field(min_length=3, max_length=500)
    expected_information_gain: str | None = Field(default=None, max_length=500)
    next_plan: AnalysisPlan | None = None

    @model_validator(mode='after')
    def validate_continuation(self):
        if self.action == DecisionAction.CONTINUE:
            if self.next_plan is None or not self.expected_information_gain:
                raise ValueError('continue requires next_plan and expected_information_gain')
        elif self.next_plan is not None:
            raise ValueError('Only continue may include next_plan')
        return self

class TraceEvent(StrictModel):
    run_id: str
    sequence: int = Field(ge=1)
    event_type: Literal['request', 'plan', 'tool_call', 'observation', 'decision', 'validation', 'final']
    timestamp_utc: str
    payload: dict[str, Any]

class FinalReport(StrictModel):
    status: TerminalStatus
    answer: str
    claims: list[EvidenceClaim]
    evidence_tools: list[str]
    answered_requirements: list[str] = Field(default_factory=list, max_length=6)
    unresolved_requirements: list[str] = Field(default_factory=list, max_length=6)
    limitations: list[str]
    stop_reason: str
