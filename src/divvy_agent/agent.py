"""Bounded adaptive Divvy agent: controller, dispatcher, completion gate and model adapters."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

from .schemas import *
from .tools import *


class AgentActionType(str, Enum):
    EXECUTE_ANALYSIS = 'execute_analysis'
    CREATE_CHART = 'create_chart'
    REPORT = 'report'
    CLARIFY = 'clarify'
    BLOCKED = 'blocked'
    FAILED = 'failed'

class AgentStep(StrictModel):
    action: AgentActionType
    observation: str = Field(min_length=3, max_length=500)
    reason: str = Field(min_length=3, max_length=500)
    expected_information_gain: str | None = Field(default=None, max_length=500)
    analysis_plan: AnalysisPlan | dict[str, Any] | None = None
    chart_request: ChartRequest | dict[str, Any] | None = None
    source_result_index: int | None = Field(default=None, ge=0)
    final_report: FinalReport | None = None
    message: str | None = Field(default=None, max_length=1000)

    @model_validator(mode='after')
    def validate_action_payload(self):
        if self.action == AgentActionType.EXECUTE_ANALYSIS:
            if self.analysis_plan is None or not self.expected_information_gain:
                raise ValueError('execute_analysis requires analysis_plan and expected_information_gain')
        elif self.action == AgentActionType.CREATE_CHART:
            if self.chart_request is None or self.source_result_index is None:
                raise ValueError('create_chart requires chart_request and source_result_index')
        elif self.action == AgentActionType.REPORT:
            if self.final_report is None:
                raise ValueError('report requires final_report')
        elif self.action in {AgentActionType.CLARIFY, AgentActionType.BLOCKED, AgentActionType.FAILED}:
            if not self.message:
                raise ValueError(f'{self.action.value} requires message')
        return self

class AgentLimits(StrictModel):
    max_model_turns: int = Field(default=5, ge=1, le=10)
    max_tool_calls: int = Field(default=4, ge=1, le=10)
    max_same_call: int = Field(default=1, ge=1, le=3)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_transient_tool_retries: int = Field(default=1, ge=0, le=2)
    max_report_revisions: int = Field(default=1, ge=0, le=2)
    max_elapsed_seconds: float = Field(default=60, gt=0, le=600)
    max_total_tokens: int = Field(default=45000, ge=100)
    max_estimated_cost_usd: float = Field(default=0.10, gt=0)

class ModelUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)

class AgentRunResult(StrictModel):
    run_id: str
    status: TerminalStatus
    final_report: FinalReport | None = None
    user_message: str | None = None
    trace: list[TraceEvent]
    model_turns: int
    tool_calls: int
    total_tokens: int
    estimated_cost_usd: float
    elapsed_seconds: float
    stop_reason: str
    phase_history: list[dict[str, Any]] = Field(default_factory=list)

class ModelOutputError(RuntimeError):
    """The model answered but its output was unusable; carries the usage that call still consumed."""

    def __init__(self, message: str, usage: ModelUsage):
        super().__init__(message)
        self.usage = usage

class ModelAdapter(Protocol):
    def decide(self, context: dict[str, Any]) -> tuple[AgentStep, ModelUsage]: ...

@dataclass
class RunState:
    run_id: str
    started_at: float
    trace: list[TraceEvent] = field(default_factory=list)
    results: list[AnalysisResult] = field(default_factory=list)
    charts: list[ChartResult] = field(default_factory=list)
    evidence: dict[str, int | float | str] = field(default_factory=dict)
    fingerprints: Counter = field(default_factory=Counter)
    call_outcomes: dict[str, str] = field(default_factory=dict)
    call_context_versions: dict[str, tuple[int, int]] = field(default_factory=dict)
    transient_tool_retries: Counter = field(default_factory=Counter)
    completion_feedback: list[str] = field(default_factory=list)
    model_error_feedback: list[str] = field(default_factory=list)
    report_revisions: int = 0
    guard_rejections: int = 0
    phase: str = 'created'
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    model_turns: int = 0
    tool_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0
    stop_reason: str = ''

TERMINAL_PHASES = {
    'completed', 'clarification_needed', 'blocked', 'budget_exhausted', 'failed'
}

LEGAL_PHASE_TRANSITIONS = {
    'created': {'inspected', 'failed', 'budget_exhausted'},
    'inspected': {'deciding', 'failed', 'budget_exhausted'},
    'deciding': {
        'deciding',  # a recoverable model error may be retried
        'tool_pending', 'validating', 'clarification_needed',
        'blocked', 'failed', 'budget_exhausted',
    },
    'tool_pending': {'observed', 'failed', 'budget_exhausted'},
    'observed': {'deciding', 'failed', 'budget_exhausted'},
    'validating': {'completed', 'report_revision', 'failed', 'budget_exhausted'},
    'report_revision': {'deciding', 'failed', 'budget_exhausted'},
    'completed': set(),
    'clarification_needed': set(),
    'blocked': set(),
    'budget_exhausted': set(),
    'failed': set(),
}

def transition_phase(state: RunState, target: str, reason: str) -> None:
    current = state.phase
    if current not in LEGAL_PHASE_TRANSITIONS:
        raise RuntimeError(f'Unknown workflow phase: {current}')
    if target not in LEGAL_PHASE_TRANSITIONS[current]:
        raise RuntimeError(f'Illegal workflow transition: {current} -> {target}')
    state.phase = target
    state.phase_history.append({
        'sequence': len(state.phase_history) + 1,
        'from_phase': current,
        'to_phase': target,
        'reason': reason,
        'model_turn': state.model_turns,
        'tool_calls': state.tool_calls,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    })

def assert_phase_history(history: list[dict[str, Any]]) -> None:
    for item in history:
        if item['to_phase'] not in LEGAL_PHASE_TRANSITIONS[item['from_phase']]:
            raise AssertionError(
                f"Illegal recorded transition: {item['from_phase']} -> {item['to_phase']}"
            )
    terminal_positions = [
        index for index, item in enumerate(history)
        if item['to_phase'] in TERMINAL_PHASES
    ]
    if terminal_positions and terminal_positions[-1] != len(history) - 1:
        raise AssertionError('A transition occurred after a terminal phase')

def add_result_evidence(state: RunState, result: AnalysisResult) -> None:
    prefix = f'result_{len(state.results) - 1}'
    dimensions = [dimension.value for dimension in result.plan.group_by]
    for row in result.rows:
        labels = '|'.join(f'{dimension}={row[dimension]}' for dimension in dimensions)
        state.evidence[f'{prefix}.{result.value_field}.{labels}'] = row[result.value_field]
    for key, value in result.derived.items():
        state.evidence[f'{prefix}.derived.{key}'] = value

    if dimensions == ['start_month', 'member_casual'] and result.value_field == 'trip_count':
        frame = pd.DataFrame(result.rows)
        months = sorted(frame['start_month'].unique())
        if len(months) == 2:
            pivot = frame.pivot(index='start_month', columns='member_casual', values='trip_count')
            # A filtered plan (e.g. members only) has no 'casual' column; skip the derived change then.
            if {'member', 'casual'} <= set(pivot.columns) and not pivot.isna().any().any():
                first, second = months
                member_change = int(pivot.loc[second, 'member'] - pivot.loc[first, 'member'])
                casual_change = int(pivot.loc[second, 'casual'] - pivot.loc[first, 'casual'])
                total_change = member_change + casual_change
                state.evidence.update({
                    f'{prefix}.derived.previous_month': int(first),
                    f'{prefix}.derived.current_month': int(second),
                    f'{prefix}.derived.member_increase': member_change,
                    f'{prefix}.derived.casual_increase': casual_change,
                    f'{prefix}.derived.total_increase': total_change,
                })
                if total_change != 0:
                    state.evidence[f'{prefix}.derived.casual_share_percent'] = round(
                        casual_change / total_change * 100, 1
                    )

def compact_result(result: AnalysisResult, index: int) -> dict[str, Any]:
    return {
        'result_index': index,
        'plan': result.plan.model_dump(mode='json'),
        'rows': result.rows,
        'coverage': result.coverage.model_dump(mode='json'),
        'derived': result.derived,
    }

def model_context(question: str, inspection: DatasetInspection, state: RunState, limits: AgentLimits) -> dict[str, Any]:
    return {
        'question': question,
        'workflow_phase': state.phase,
        'dataset': inspection.model_dump(mode='json'),
        'observations': [compact_result(result, index) for index, result in enumerate(state.results)],
        'charts': [chart.model_dump(mode='json') for chart in state.charts],
        'evidence_catalog': state.evidence,
        'completion_feedback': state.completion_feedback,
        'previous_model_error': state.model_error_feedback,
        'remaining_budget': {
            'model_turns': limits.max_model_turns - state.model_turns,
            'tool_calls': limits.max_tool_calls - state.tool_calls,
            'tokens': limits.max_total_tokens - state.input_tokens - state.output_tokens,
            'report_revisions': limits.max_report_revisions - state.report_revisions,
        },
        'instruction': (
            ('Choose one next action. Begin with a minimal analysis plan. ' if not state.results else
               'Results are already in observations; do not recompute any plan listed there. Choose a different '
               'follow-up analysis, create_chart, or report. ')
            + 'After each result, continue only when '
            'a follow-up materially improves the answer. Use only evidence_catalog keys and exact values in '
            'final_report claims. List the question parts answered in answered_requirements and any remaining '
            'parts in unresolved_requirements. If completion_feedback is nonempty, revise the report to address it. '
            'Do not claim that an unperformed calculation or external action occurred.'
        ),
    }

def raw_arguments(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode='json') if isinstance(value, BaseModel) else value

def fingerprint_step(step: AgentStep) -> str:
    if step.action == AgentActionType.EXECUTE_ANALYSIS:
        payload = raw_arguments(step.analysis_plan)
    elif step.action == AgentActionType.CREATE_CHART:
        request = raw_arguments(step.chart_request)
        # Ignore cosmetic wording so equivalent visualizations share a fingerprint.
        payload = {
            'source_result_index': step.source_result_index,
            'chart_type': request.get('chart_type'),
            'x': request.get('x'),
            'y': request.get('y'),
            'series': request.get('series'),
            'filename': request.get('filename'),
        }
    else:
        payload = {'action': step.action.value}
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def call_context_version(state: RunState) -> tuple[int, int]:
    return len(state.results), len(state.charts)

def repeated_call_block_reason(
    state: RunState, fingerprint: str, limits: AgentLimits
) -> str | None:
    previous = state.call_outcomes.get(fingerprint)
    if previous is None:
        return None
    if previous == 'tool_execution_error':
        if state.transient_tool_retries[fingerprint] < limits.max_transient_tool_retries:
            state.transient_tool_retries[fingerprint] += 1
            return None
        return 'transient_tool_retry_exhausted'
    if previous == 'source_result_unavailable':
        if state.call_context_versions.get(fingerprint) != call_context_version(state):
            return None
        return 'repeated_state_dependent_call_without_new_evidence'
    if previous == 'success':
        return 'repeated_successful_call'
    return 'repeated_identical_invalid_call'

def record_call_outcome(
    state: RunState, fingerprint: str, observation: ToolObservation
) -> None:
    state.call_outcomes[fingerprint] = 'success' if observation.ok else str(observation.error_code)
    state.call_context_versions[fingerprint] = call_context_version(state)

class LiveAgentStep(AgentStep):
    """AgentStep for OpenAI structured outputs: no free-form dict alternatives.

    AgentStep accepts raw dicts so that invalid arguments reach the dispatcher and can be
    corrected in tests. OpenAI strict schemas reject open objects, so the live model must
    produce the typed AnalysisPlan / ChartRequest directly.
    """
    analysis_plan: AnalysisPlan | None = None
    chart_request: ChartRequest | None = None


class OpenAIModelAdapter:
    """OpenAI-compatible adapter. Works with api.openai.com or a gateway such as Portkey (set base_url)."""

    def __init__(
        self, model: str, input_cost_per_million: float = 0, output_cost_per_million: float = 0,
        base_url: str | None = None, api_key: str | None = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or None, base_url=base_url or None)
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def decide(self, context: dict[str, Any]) -> tuple[AgentStep, ModelUsage]:
        # Chat Completions is the API most OpenAI-compatible gateways support. The raw response is
        # used so token usage is still counted when the structured output fails validation.
        raw = self.client.chat.completions.with_raw_response.parse(
            model=self.model,
            messages=[
                {'role': 'system', 'content': (
                    'You are the planning component of a bounded Divvy data-analysis agent. Return exactly one '
                    'validated next action. Python executes all calculations. Never calculate from memory, invent '
                    'evidence, request arbitrary code, or expose private chain-of-thought. '
                    'Tools. execute_analysis runs analyze_data (metric trip_count or average_duration_minutes, '
                    'grouped by up to two dimensions, optional filters). create_chart plots an existing result by '
                    'source_result_index. Derived evidence is computed automatically only in two cases: (a) '
                    'trip_count grouped by start_month alone gives derived.largest_increase_month, '
                    'derived.largest_increase and derived.previous_month; (b) trip_count grouped by '
                    '[start_month, member_casual], filtered to exactly two months and not filtered by rider type, '
                    'gives derived.member_increase, derived.casual_increase, derived.total_increase and '
                    'derived.casual_share_percent. To explain a change by rider type, first find the month, then '
                    'run case (b) for that month and the month before it. '
                    'Rules for final_report: (1) every claim key must be copied verbatim from the keys of '
                    'evidence_catalog, for example result_1.derived.casual_increase, and every claim value must '
                    'equal the catalog value exactly; never invent short key names. (2) evidence_tools may only '
                    'contain names of tools actually used, chosen from: inspect_dataset, analyze_data, '
                    'create_visualization, validate_result. (3) If previous_model_error is nonempty, your last '
                    'output was invalid; fix that problem. Do not repeat a calculation already in observations. '
                    'Charts listed in context were created successfully; do not request an equivalent chart again. '
                    '(4) The data records trips only, not causes. If the question asks why something happened, '
                    'report what the data shows without saying it happened because of those numbers, and add a '
                    'limitation stating that the data does not establish the cause. '
                    '(5) Scope check before any analysis. If a key term in the question has several reasonable '
                    'meanings that would give different answers (for example "best" or "most valuable" with no '
                    'measure given), return action clarify and ask the user to define it; run no analysis. If '
                    'answering needs something this tool set cannot provide, return action blocked and say what is '
                    'missing; run no analysis. That covers a field the dataset lacks (see dataset.fields), a period '
                    'outside dataset.date_range, external data such as weather, events or prices, a statistic '
                    'other than the supported metrics (for example standard deviation or percentiles), and a '
                    'causal claim that would need external data. Never substitute a different metric, period or '
                    'proxy and present it as the answer. Use clarify when the user can fix the gap by giving a '
                    'definition; use blocked when the data or capability itself is absent.'
                )},
                {'role': 'user', 'content': json.dumps(context)},
            ],
            response_format=LiveAgentStep,
        )
        usage_data = (raw.http_response.json() or {}).get('usage') or {}
        input_tokens = int(usage_data.get('prompt_tokens') or 0)
        output_tokens = int(usage_data.get('completion_tokens') or 0)
        usage = ModelUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            estimated_cost_usd=(
                input_tokens * self.input_cost_per_million / 1_000_000
                + output_tokens * self.output_cost_per_million / 1_000_000
            ),
        )
        try:
            message = raw.parse().choices[0].message
            if message.parsed is None:
                raise RuntimeError(f'Model response did not contain a parsed AgentStep (refusal: {message.refusal})')
            step = AgentStep.model_validate(message.parsed.model_dump(mode='json'))
        except Exception as error:
            raise ModelOutputError(f'{type(error).__name__}: {error}', usage) from error
        return step, usage

class VisualizationToolArgs(StrictModel):
    source_result_index: int = Field(ge=0)
    request: ChartRequest

class ToolObservation(StrictModel):
    tool_name: str
    ok: bool
    result_type: str | None = None
    result: dict[str, Any] | None = None
    validated_arguments: dict[str, Any] | None = None
    error_code: Literal[
        'tool_not_allowed', 'invalid_arguments', 'source_result_unavailable',
        'tool_execution_error'
    ] | None = None
    detail: str | list[dict[str, Any]] | None = None
    recoverable: bool = False

TOOL_REGISTRY = {
    'analyze_data': AnalysisPlan,
    'create_visualization': VisualizationToolArgs,
}

def dispatch_tool(name: str, arguments: dict[str, Any], state: RunState) -> ToolObservation:
    if name not in TOOL_REGISTRY:
        return ToolObservation(
            tool_name=name, ok=False, error_code='tool_not_allowed',
            detail=f'Tool is not in the allowlist: {name}', recoverable=False,
        )

    try:
        validated = TOOL_REGISTRY[name].model_validate(arguments)
    except ValidationError as error:
        return ToolObservation(
            tool_name=name, ok=False, error_code='invalid_arguments',
            detail=error.errors(include_url=False), recoverable=True,
        )

    if name == 'analyze_data':
        try:
            result = analyze_data(validated)
            return ToolObservation(
                tool_name=name, ok=True, result_type='AnalysisResult',
                result=result.model_dump(mode='json'),
                validated_arguments=validated.model_dump(mode='json'),
            )
        except Exception as error:
            return ToolObservation(
                tool_name=name, ok=False,
                validated_arguments=validated.model_dump(mode='json'),
                error_code='tool_execution_error',
                detail=f'{type(error).__name__}: {error}', recoverable=True,
            )

    if validated.source_result_index >= len(state.results):
        return ToolObservation(
            tool_name=name, ok=False,
            validated_arguments=validated.model_dump(mode='json'),
            error_code='source_result_unavailable',
            detail=f'Result index {validated.source_result_index} is unavailable.', recoverable=True,
        )
    try:
        result = create_visualization(
            state.results[validated.source_result_index], validated.request
        )
        return ToolObservation(
            tool_name=name, ok=True, result_type='ChartResult',
            result=result.model_dump(mode='json'),
            validated_arguments=validated.model_dump(mode='json'),
        )
    except Exception as error:
        return ToolObservation(
            tool_name=name, ok=False,
            validated_arguments=validated.model_dump(mode='json'),
            error_code='tool_execution_error',
            detail=f'{type(error).__name__}: {error}', recoverable=True,
        )


class CompletionGateResult(StrictModel):
    valid: bool
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    repairable: bool = True

PROHIBITED_COMPLETION_CLAIMS = (
    'we modified the dataset', 'we contacted', 'we submitted', 'we approved',
    'the issue has been resolved',
)

DEPENDENT_FOLLOW_UP_PATTERNS = (
    r'\bthat month\b', r'\bthat weekday\b', r'\bthat hour\b',
    r'\bthat station\b', r'\bthat destination\b', r'\bthat bike type\b',
    r'\bthat rider type\b', r'\bthat difference\b', r'\bthat increase\b',
    r'\bthe peak\b', r'\bwithin that\b', r'\bending month\b',
    r'\bits trip counts\b', r'\btheir most common\b',
)

# Phrases that make a later calculation depend on the value selected by the
# first result. The binding is checked at completion so a real but unrelated
# evidence row cannot be presented as the answer to the dependent follow-up.
DEPENDENCY_DIMENSION_PATTERNS = {
    'start_month': (r'\bthat month\b', r'\bending month\b', r'\bwithin that month\b'),
    'start_weekday': (r'\bthat weekday\b', r'\bwithin that weekday\b'),
    'start_hour': (r'\bthat hour\b', r'\bwithin that hour\b'),
    'member_casual': (r'\bthat rider type\b',),
    'rideable_type': (r'\bthat bike type\b',),
    'start_station_name': (r'\bthat start station\b', r'\bthat station\b'),
    'end_station_name': (r'\bthat destination\b', r'\bthat destination station\b'),
}

AMBIGUOUS_REQUEST_PATTERNS = (
    r'\bmost important\b', r'\bbest station\b', r'\bmost valuable\b',
    r'\bbetter station\b',
)

UNSUPPORTED_REQUEST_PATTERNS = (
    r'\bage\b', r'\bage group\b', r'\bgender\b', r'\bincome\b',
    r'\bweather\b', r'\btemperature\b', r'\bprice\b', r'\brevenue\b',
    r'\bmedian\b', r'\bpercentile\b', r'\bstandard deviation\b',
    r'\bcorrelation\b', r'\bforecast\b', r'\bpredict\b', r'\bwhy\b',
)

SUPPORTED_ANALYSIS_PATTERNS = (
    r'\btrip(?:s)?\b', r'\bduration\b', r'\bstart(?:ing)? stations?\b',
    r'\bdestination station\b', r'\bweekday\b', r'\bstart hour\b',
    r'\bbike type\b', r'\brider type\b',
)


def requires_observation_driven_follow_up(question: str) -> bool:
    """Detect a result-dependent second request in the bounded Divvy question domain."""
    text = question.lower()
    return any(re.search(pattern, text) for pattern in DEPENDENT_FOLLOW_UP_PATTERNS)


def dependent_follow_up_binding_error(
    question: str, state: RunState, report: FinalReport,
) -> str | None:
    """Require later evidence to retain the entity selected by the first result.

    A binding is satisfied when a later analysis is filtered to the selected
    value, or when a later claim key identifies that value in a grouped result.
    This permits both efficient filtered follow-ups and valid two-dimensional
    breakdowns while rejecting unrelated but numerically grounded rows.
    """
    if len(state.results) < 2:
        return None
    text = question.lower()
    first = state.results[0]
    first_dimensions = {dimension.value for dimension in first.plan.group_by}
    first_filters = {
        (item.field.value, item.value)
        for item in first.plan.filters
        if item.operator == FilterOperator.EQUALS
    }
    # Do not infer a selected value from Result 0 when that result omitted an
    # explicitly named subgroup. In that situation Result 0 cannot be the
    # semantic selector for the dependent phrase (for example, a monthly
    # overall series cannot select "the month" for a casual-rider maximum).
    named_scopes = (
        ('casual', 'member_casual', 'casual'),
        ('member', 'member_casual', 'member'),
        ('electric', 'rideable_type', 'electric_bike'),
        ('classic', 'rideable_type', 'classic_bike'),
    )
    for marker, field, value in named_scopes:
        if re.search(rf'\b{marker}\b', text):
            if field not in first_dimensions and (field, value) not in first_filters:
                return None
    requested_dimensions = [
        dimension
        for dimension, patterns in DEPENDENCY_DIMENSION_PATTERNS.items()
        if dimension in first_dimensions
        and any(re.search(pattern, text) for pattern in patterns)
    ]
    if not requested_dimensions:
        return None

    minimize = bool(re.search(r'\b(?:fewest|shortest|lowest|minimum)\b', text))
    for dimension in requested_dimensions:
        if (
            dimension == 'start_month'
            and 'largest_increase_month' in first.derived
            and re.search(r'\b(?:increase|grew|growth)\b', text)
        ):
            selected = first.derived['largest_increase_month']
        else:
            candidates = [row for row in first.rows if dimension in row]
            if not candidates:
                continue
            selector = min if minimize else max
            selected = selector(candidates, key=lambda row: float(row[first.value_field]))[dimension]

        plan_is_scoped = any(
            any(
                item.field.value == dimension
                and item.operator == FilterOperator.EQUALS
                and item.value == selected
                for item in result.plan.filters
            )
            for result in state.results[1:]
        )
        evidence_token = f'{dimension}={selected}'
        claim_is_scoped = any(
            claim.key.startswith('result_')
            and not claim.key.startswith('result_0.')
            and evidence_token in claim.key
            for claim in report.claims
        )
        if not (plan_is_scoped or claim_is_scoped):
            return (
                f'The follow-up depends on {dimension}={selected}, selected by the first result. '
                'Run the later analysis with that exact filter or cite later grouped evidence whose key '
                'contains the selected value.'
            )
    return None


def initial_adaptive_plan_error(
    question: str, state: RunState, step: AgentStep,
) -> str | None:
    """Keep the first calculation minimal so a dependent follow-up uses an observation."""
    if (
        step.action == AgentActionType.EXECUTE_ANALYSIS
        and not state.results
        and requires_observation_driven_follow_up(question)
    ):
        plan = raw_arguments(step.analysis_plan)
        if len(plan.get('group_by') or []) > 1:
            return (
                'This request contains a result-dependent follow-up. The first analysis must use only the '
                'primary grouping dimension. Observe that result, then choose a separate follow-up analysis '
                'using the discovered month, weekday, hour, station, bike type, or rider type.'
            )
    if step.action == AgentActionType.EXECUTE_ANALYSIS and state.results:
        plan = raw_arguments(step.analysis_plan)
        group_by = set(plan.get('group_by') or [])
        pinned = {
            f.get('field') for f in plan.get('filters') or []
            if f.get('operator') == 'equals'
        }
        compared = group_by & {'member_casual', 'rideable_type'}
        if plan.get('limit') and compared and (group_by - compared) <= pinned:
            return (
                'A limit would drop categories from a comparison. Resend the analysis without limit so every '
                f'{" and ".join(sorted(compared))} value in the follow-up is returned.'
            )
    return None


def supported_request_terminal_error(
    question: str, inspection: DatasetInspection, action: AgentActionType,
) -> str | None:
    """Reject clarify/block decisions when the inspected snapshot can answer the request."""
    if action not in {AgentActionType.CLARIFY, AgentActionType.BLOCKED}:
        return None
    text = question.lower()
    if any(re.search(pattern, text) for pattern in AMBIGUOUS_REQUEST_PATTERNS):
        return None
    if any(re.search(pattern, text) for pattern in UNSUPPORTED_REQUEST_PATTERNS):
        return None
    years = {int(value) for value in re.findall(r'\b(?:19|20)\d{2}\b', text)}
    if years and years != {2025}:
        return None
    if not any(re.search(pattern, text) for pattern in SUPPORTED_ANALYSIS_PATTERNS):
        return None
    return (
        f'The inspected dataset covers {inspection.date_range[0]} through {inspection.date_range[1]} and '
        'supports trip_count and average_duration_minutes over the requested Divvy dimensions. The request '
        'is sufficiently specified; choose execute_analysis instead of clarify or blocked.'
    )


def unsupported_clarification_block_reason(
    question: str, inspection: DatasetInspection,
) -> str | None:
    """Return a controller-owned block explanation for an impossible clarification.

    Clarification is reserved for ambiguous supported concepts. Asking the user
    another question cannot supply a field, period, statistic, external context,
    or causal capability that the bounded workflow does not have.
    """
    text = question.lower()
    if any(re.search(pattern, text) for pattern in AMBIGUOUS_REQUEST_PATTERNS):
        return None
    if any(re.search(pattern, text) for pattern in UNSUPPORTED_REQUEST_PATTERNS):
        return (
            'This request is blocked because it requires data or an analysis capability '
            'that is not available in the inspected Divvy dataset and approved tools.'
        )
    years = {int(value) for value in re.findall(r'\b(?:19|20)\d{2}\b', text)}
    if years and years != {2025}:
        return (
            f'This request is blocked because the dataset covers only '
            f'{inspection.date_range[0]} through {inspection.date_range[1]}.'
        )
    return None

def with_exclusion_disclosure(report: FinalReport, state: RunState) -> FinalReport:
    """The controller, not the model, states material eligibility exclusions it computed itself."""
    material = [r for r in state.results if r.coverage.excluded_rate >= 0.01]
    existing = ' '.join(report.limitations).lower()
    if not material or 'exclud' in existing or 'eligibility' in existing:
        return report
    worst = max(material, key=lambda r: r.coverage.excluded_rate)
    note = (
        f'{worst.coverage.excluded_rows:,} of {worst.coverage.candidate_rows:,} rows were excluded as ineligible '
        f'({worst.coverage.eligibility_rule}).'
    )
    return report.model_copy(update={'limitations': [*report.limitations, note]})

def completion_gate(
    question: str, state: RunState, report: FinalReport,
    evidence_validation: ValidationResult,
) -> CompletionGateResult:
    used_tools = {'inspect_dataset'} | {
        event.payload.get('tool')
        for event in state.trace
        if event.event_type == 'tool_call' and event.payload.get('tool')
    }
    report_text = report.answer.lower()
    limitation_text = ' '.join(report.limitations).lower()
    material_exclusion = any(
        result.coverage.excluded_rate >= 0.01 for result in state.results
    )
    dependent_follow_up_required = requires_observation_driven_follow_up(question)
    dependency_error = dependent_follow_up_binding_error(question, state, report)
    checks = {
        'completed_status': report.status == TerminalStatus.COMPLETED,
        'analysis_evidence_exists': bool(state.results),
        'claims_match_evidence': evidence_validation.valid,
        'evidence_tools_used': bool(report.evidence_tools) and set(report.evidence_tools).issubset(used_tools | {'validate_result'}),
        'question_requirements_declared': bool(report.answered_requirements),
        'no_unresolved_requirements': not report.unresolved_requirements,
        'observation_driven_follow_up': (
            not dependent_follow_up_required or len(state.results) >= 2
        ),
        'dependent_follow_up_preserves_selection': dependency_error is None,
        'material_exclusions_disclosed': (
            not material_exclusion
            or 'exclud' in limitation_text
            or 'eligibility' in limitation_text
        ),
        'no_prohibited_completion_claims': not any(
            phrase in report_text for phrase in PROHIBITED_COMPLETION_CLAIMS
        ),
    }
    messages = {
        'completed_status': 'Final report status must be completed.',
        'analysis_evidence_exists': 'A factual report requires at least one analysis result.',
        'claims_match_evidence': 'One or more report claims are not supported by computed evidence.',
        'evidence_tools_used': 'Evidence tools must be nonempty and limited to tools actually used.',
        'question_requirements_declared': 'List the question parts answered by the report.',
        'no_unresolved_requirements': 'Resolve or explicitly stop for remaining requirements before completion.',
        'observation_driven_follow_up': (
            'This question contains a result-dependent follow-up. Run a second analysis after observing the '
            'first result before reporting.'
        ),
        'dependent_follow_up_preserves_selection': dependency_error or '',
        'material_exclusions_disclosed': 'Disclose material eligibility exclusions in limitations.',
        'no_prohibited_completion_claims': 'Remove claims that an external or unperformed action was completed.',
    }
    errors = [messages[name] for name, passed in checks.items() if not passed]
    nonrepairable = not checks['analysis_evidence_exists']
    return CompletionGateResult(
        valid=all(checks.values()), checks=checks, errors=errors,
        repairable=not nonrepairable,
    )

MODEL_ERROR_HINTS = (
    ('execute_analysis requires', ' Resend execute_analysis with analysis_plan {objective, metric, group_by, '
     'filters, sort_order, limit} and a nonempty expected_information_gain.'),
    ('requires message', ' Resend the action with a nonempty message, or choose execute_analysis or report.'),
    ('start_year', ' Omit start_year filters; the dataset is 2025 only. Never send an empty filter value.'),
    ('report requires final_report', ' Resend report with a complete final_report.'),
)

def model_error_hint(error: Exception) -> str:
    text = str(error)
    return ''.join(hint for marker, hint in MODEL_ERROR_HINTS if marker in text)

MAX_GUARD_REJECTIONS = 3

def reject_by_guard(state: RunState, limits: AgentLimits):
    """A guard rejection is controller feedback, not progress: refund the model turn it used.

    Returns a terminal reason once the guards have rejected too many times, else None.
    """
    state.guard_rejections += 1
    if state.guard_rejections > MAX_GUARD_REJECTIONS:
        return 'guard_rejection_limit'
    state.model_turns = max(0, state.model_turns - 1)
    return None

def remaining_budget(state: RunState, limits: AgentLimits) -> dict[str, int | float]:
    return {
        'model_turns': max(0, limits.max_model_turns - state.model_turns),
        'tool_calls': max(0, limits.max_tool_calls - state.tool_calls),
        'tokens': max(0, limits.max_total_tokens - state.input_tokens - state.output_tokens),
        'report_revisions': max(0, limits.max_report_revisions - state.report_revisions),
        'elapsed_seconds': round(time.perf_counter() - state.started_at, 3),
        'time_remaining_seconds': round(max(0, limits.max_elapsed_seconds - (time.perf_counter() - state.started_at)), 3),
    }

def decision_trace_payload(step: AgentStep, state: RunState, limits: AgentLimits) -> dict[str, Any]:
    payload = step.model_dump(mode='json')
    payload.update({
        'model_turn': state.model_turns,
        'remaining_budget': remaining_budget(state, limits),
    })
    return payload

def tool_observation_payload(
    observation: ToolObservation, state: RunState, limits: AgentLimits,
    elapsed_ms: float, retry_number: int, evidence_keys_created: list[str],
) -> dict[str, Any]:
    payload = observation.model_dump(mode='json')
    payload.update({
        'model_turn': state.model_turns,
        'execution_elapsed_ms': round(elapsed_ms, 3),
        'retry_number': retry_number,
        'evidence_keys_created': evidence_keys_created,
        'remaining_budget': remaining_budget(state, limits),
    })
    return payload

def finalize_run(
    state: RunState, status: TerminalStatus, stop_reason: str,
    report: FinalReport | None = None, user_message: str | None = None,
) -> AgentRunResult:
    state.stop_reason = stop_reason
    terminal_phase = status.value
    if state.phase != terminal_phase:
        transition_phase(state, terminal_phase, stop_reason)
    assert_phase_history(state.phase_history)
    return AgentRunResult(
        run_id=state.run_id, status=status, final_report=report, user_message=user_message,
        trace=state.trace, model_turns=state.model_turns, tool_calls=state.tool_calls,
        total_tokens=state.input_tokens + state.output_tokens,
        estimated_cost_usd=round(state.estimated_cost_usd, 6),
        elapsed_seconds=round(time.perf_counter() - state.started_at, 3),
        stop_reason=stop_reason, phase_history=state.phase_history,
    )

def run_agent(question: str, adapter: ModelAdapter, limits: AgentLimits | None = None) -> AgentRunResult:
    limits = limits or AgentLimits()
    state = RunState(run_id=str(uuid.uuid4()), started_at=time.perf_counter())
    sequence = 1
    state.trace.append(trace_event(state.run_id, sequence, 'request', {
        'question': question,
        'limits': limits.model_dump(mode='json'),
    }))
    sequence += 1

    inspection = inspect_dataset()
    state.tool_calls += 1
    state.trace.append(trace_event(state.run_id, sequence, 'observation', inspection))
    sequence += 1
    transition_phase(state, 'inspected', 'dataset_inspection_completed')

    while True:
        elapsed = time.perf_counter() - state.started_at
        if elapsed > limits.max_elapsed_seconds:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'time_limit')
        if state.model_turns >= limits.max_model_turns:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'model_turn_limit')
        if state.input_tokens + state.output_tokens >= limits.max_total_tokens:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'token_limit')
        if state.estimated_cost_usd >= limits.max_estimated_cost_usd:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'cost_limit')

        transition_phase(state, 'deciding', 'request_next_adaptive_action')
        context = model_context(question, inspection, state, limits)
        try:
            step, usage = adapter.decide(context)
        except Exception as error:
            failed_usage = getattr(error, 'usage', None)
            if failed_usage is not None:
                state.input_tokens += failed_usage.input_tokens
                state.output_tokens += failed_usage.output_tokens
                state.estimated_cost_usd += failed_usage.estimated_cost_usd
            state.retries += 1
            state.model_error_feedback = [f'{type(error).__name__}: {str(error)[:500]}{model_error_hint(error)}']
            state.trace.append(trace_event(state.run_id, sequence, 'observation', {
                'error_type': type(error).__name__, 'message': str(error), 'retry': state.retries,
            }))
            sequence += 1
            if state.retries > limits.max_retries:
                return finalize_run(state, TerminalStatus.FAILED, 'model_error_retry_exhausted')
            continue

        state.model_error_feedback = []
        state.model_turns += 1
        state.input_tokens += usage.input_tokens
        state.output_tokens += usage.output_tokens
        state.estimated_cost_usd += usage.estimated_cost_usd
        state.trace.append(trace_event(
            state.run_id, sequence, 'decision',
            decision_trace_payload(step, state, limits),
        ))
        sequence += 1

        if state.input_tokens + state.output_tokens > limits.max_total_tokens:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'token_limit')
        if state.estimated_cost_usd > limits.max_estimated_cost_usd:
            return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'cost_limit')

        plan_guard_error = initial_adaptive_plan_error(question, state, step)
        if plan_guard_error is not None:
            state.model_error_feedback = [plan_guard_error]
            state.trace.append(trace_event(state.run_id, sequence, 'observation', {
                'error_type': 'adaptive_plan_guard_rejected',
                'message': plan_guard_error,
                'model_turn': state.model_turns,
                'remaining_budget': remaining_budget(state, limits),
            }))
            sequence += 1
            stop = reject_by_guard(state, limits)
            if stop:
                return finalize_run(state, TerminalStatus.FAILED, stop)
            continue

        if step.action in {AgentActionType.EXECUTE_ANALYSIS, AgentActionType.CREATE_CHART}:
            if state.tool_calls >= limits.max_tool_calls:
                return finalize_run(state, TerminalStatus.BUDGET_EXHAUSTED, 'tool_call_limit')
            fingerprint = fingerprint_step(step)
            block_reason = repeated_call_block_reason(state, fingerprint, limits)
            if block_reason is not None:
                if (
                    step.action == AgentActionType.CREATE_CHART
                    and block_reason == 'repeated_successful_call'
                ):
                    state.model_error_feedback = [
                        'Equivalent chart already created successfully. Do not call the chart tool again; '
                        'use the existing chart and proceed to the final report.'
                    ]
                    state.trace.append(trace_event(state.run_id, sequence, 'observation', {
                        'error_type': 'duplicate_chart_blocked',
                        'message': state.model_error_feedback[0],
                        'model_turn': state.model_turns,
                        'fingerprint': fingerprint,
                        'remaining_budget': remaining_budget(state, limits),
                    }))
                    sequence += 1
                    continue
                return finalize_run(state, TerminalStatus.FAILED, block_reason)
            state.fingerprints[fingerprint] += 1

        if step.action == AgentActionType.EXECUTE_ANALYSIS:
            transition_phase(state, 'tool_pending', 'analysis_tool_selected')
            arguments = raw_arguments(step.analysis_plan)
            state.trace.append(trace_event(state.run_id, sequence, 'tool_call', {
                'model_turn': state.model_turns,
                'tool': 'analyze_data',
                'raw_arguments': arguments,
                'fingerprint': fingerprint,
                'reason': step.reason,
                'expected_information_gain': step.expected_information_gain,
                'retry_number': state.transient_tool_retries[fingerprint],
                'remaining_budget': remaining_budget(state, limits),
            }))
            sequence += 1
            evidence_before = set(state.evidence)
            tool_started = time.perf_counter()
            observation = dispatch_tool('analyze_data', arguments, state)
            execution_elapsed_ms = (time.perf_counter() - tool_started) * 1000
            state.tool_calls += 1
            if observation.ok:
                result = AnalysisResult.model_validate(observation.result)
                state.results.append(result)
                add_result_evidence(state, result)
            evidence_created = sorted(set(state.evidence) - evidence_before)
            record_call_outcome(state, fingerprint, observation)
            state.trace.append(trace_event(
                state.run_id, sequence, 'observation',
                tool_observation_payload(
                    observation, state, limits, execution_elapsed_ms,
                    state.transient_tool_retries[fingerprint], evidence_created,
                ),
            ))
            sequence += 1
            transition_phase(state, 'observed', 'analysis_observation_recorded')
            continue

        if step.action == AgentActionType.CREATE_CHART:
            transition_phase(state, 'tool_pending', 'visualization_tool_selected')
            arguments = {
                'source_result_index': step.source_result_index,
                'request': raw_arguments(step.chart_request),
            }
            state.trace.append(trace_event(state.run_id, sequence, 'tool_call', {
                'model_turn': state.model_turns,
                'tool': 'create_visualization',
                'raw_arguments': arguments,
                'fingerprint': fingerprint,
                'reason': step.reason,
                'expected_information_gain': step.expected_information_gain,
                'retry_number': state.transient_tool_retries[fingerprint],
                'remaining_budget': remaining_budget(state, limits),
            }))
            sequence += 1
            tool_started = time.perf_counter()
            observation = dispatch_tool('create_visualization', arguments, state)
            execution_elapsed_ms = (time.perf_counter() - tool_started) * 1000
            state.tool_calls += 1
            if observation.ok:
                state.charts.append(ChartResult.model_validate(observation.result))
            record_call_outcome(state, fingerprint, observation)
            state.trace.append(trace_event(
                state.run_id, sequence, 'observation',
                tool_observation_payload(
                    observation, state, limits, execution_elapsed_ms,
                    state.transient_tool_retries[fingerprint], [],
                ),
            ))
            sequence += 1
            transition_phase(state, 'observed', 'visualization_observation_recorded')
            continue

        if step.action == AgentActionType.REPORT:
            transition_phase(state, 'validating', 'report_submitted_for_completion_gate')
            if not state.results:
                return finalize_run(state, TerminalStatus.FAILED, 'report_without_analysis')
            step = step.model_copy(update={'final_report': with_exclusion_disclosure(step.final_report, state)})
            validation = validate_result(
                state.results[-1], step.final_report.claims, state.evidence,
                state.charts[-1] if state.charts else None,
            )
            gate = completion_gate(question, state, step.final_report, validation)
            state.trace.append(trace_event(state.run_id, sequence, 'validation', {
                'evidence_validation': validation.model_dump(mode='json'),
                'completion_gate': gate.model_dump(mode='json'),
                'model_turn': state.model_turns,
                'report_revision': state.report_revisions,
                'remaining_budget': remaining_budget(state, limits),
            }))
            sequence += 1
            if not gate.valid:
                state.completion_feedback = gate.errors
                if gate.repairable and state.report_revisions < limits.max_report_revisions:
                    state.report_revisions += 1
                    transition_phase(state, 'report_revision', 'completion_gate_requested_revision')
                    continue
                reason = 'completion_revision_exhausted' if gate.repairable else 'completion_gate_nonrepairable'
                return finalize_run(state, TerminalStatus.FAILED, reason)
            state.completion_feedback = []
            report = step.final_report
            if not report.stop_reason.strip():
                # The controller, not the model, decides why the run stopped.
                report = report.model_copy(update={'stop_reason': 'completion_gate_passed'})
            final_payload = report.model_dump(mode='json')
            final_payload.update({
                'model_turn': state.model_turns,
                'remaining_budget': remaining_budget(state, limits),
            })
            state.trace.append(trace_event(state.run_id, sequence, 'final', final_payload))
            return finalize_run(state, TerminalStatus.COMPLETED, report.stop_reason, report)

        if step.action == AgentActionType.CLARIFY:
            block_reason = unsupported_clarification_block_reason(question, inspection)
            if block_reason is not None:
                state.trace.append(trace_event(state.run_id, sequence, 'observation', {
                    'event': 'terminal_policy_normalized',
                    'from_action': AgentActionType.CLARIFY.value,
                    'to_action': AgentActionType.BLOCKED.value,
                    'reason': block_reason,
                    'model_turn': state.model_turns,
                    'remaining_budget': remaining_budget(state, limits),
                }))
                sequence += 1
                return finalize_run(
                    state, TerminalStatus.BLOCKED, 'unsupported_request_normalized_to_blocked',
                    user_message=block_reason,
                )

        terminal_guard_error = supported_request_terminal_error(
            question, inspection, step.action
        )
        if terminal_guard_error is not None:
            state.model_error_feedback = [terminal_guard_error]
            state.trace.append(trace_event(state.run_id, sequence, 'observation', {
                'error_type': 'terminal_action_guard_rejected',
                'message': terminal_guard_error,
                'model_turn': state.model_turns,
                'remaining_budget': remaining_budget(state, limits),
            }))
            sequence += 1
            stop = reject_by_guard(state, limits)
            if stop:
                return finalize_run(state, TerminalStatus.FAILED, stop)
            continue

        if step.action == AgentActionType.CLARIFY:
            return finalize_run(state, TerminalStatus.CLARIFICATION_NEEDED, 'clarification_needed', user_message=step.message)
        if step.action == AgentActionType.BLOCKED:
            return finalize_run(state, TerminalStatus.BLOCKED, 'request_outside_permissions', user_message=step.message)
        return finalize_run(state, TerminalStatus.FAILED, 'model_declared_failure', user_message=step.message)

class MockModelAdapter:
    def __init__(self, steps: list[AgentStep], input_tokens: int = 300, output_tokens: int = 100):
        self.steps = list(steps)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def decide(self, context: dict[str, Any]) -> tuple[AgentStep, ModelUsage]:
        if not self.steps:
            raise RuntimeError('Mock adapter has no remaining test step')
        return self.steps.pop(0), ModelUsage(
            input_tokens=self.input_tokens, output_tokens=self.output_tokens, estimated_cost_usd=0
        )
