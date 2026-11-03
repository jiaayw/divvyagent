"""Trusted Divvy analysis tools: inspect_dataset, analyze_data, create_visualization, validate_result."""

from __future__ import annotations

import calendar
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds
from pydantic import BaseModel

from .schemas import *

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / 'data' / 'processed' / 'divvy_2025'
PARQUET_DIR = DATASET_DIR / 'parquet'
CHART_DIR = PROJECT_ROOT / 'output' / 'charts'


@lru_cache(maxsize=1)
def get_dataset() -> ds.Dataset:
    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f'Processed dataset not found at {PARQUET_DIR}. Keep the data/processed folder next to app.py.')
    return ds.dataset(PARQUET_DIR, format='parquet', partitioning='hive')


@lru_cache(maxsize=1)
def get_quality_report() -> dict:
    return json.loads((DATASET_DIR / 'quality_report.json').read_text(encoding='utf-8'))


DIMENSION_COLUMNS = {dimension.value for dimension in Dimension}
FILTER_COLUMNS = {field.value for field in FilterField}

def eligibility_for_plan(plan: AnalysisPlan) -> tuple[str, str]:
    dimensions = {dimension.value for dimension in plan.group_by}
    if {'start_station_name', 'end_station_name'} <= dimensions:
        return 'eligible_station_pair', 'Both start and end station name and ID are present.'
    if 'start_station_name' in dimensions:
        return 'eligible_start_station', 'Start station name and ID are present.'
    if 'end_station_name' in dimensions:
        return 'eligible_end_station', 'End station name and ID are present.'
    if plan.metric == Metric.AVERAGE_DURATION_MINUTES:
        return 'eligible_typical_duration', 'Duration is greater than 0 seconds and no longer than 24 hours.'
    return 'eligible_trip_count', 'Ride ID and both timestamps are present.'

def apply_filters(frame: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    result = frame
    for item in filters:
        column = item.field.value
        if item.operator == FilterOperator.EQUALS:
            result = result[result[column] == item.value]
        elif item.operator == FilterOperator.IN:
            result = result[result[column].isin(item.value)]
        else:
            raise ValueError(f'Unsupported filter operator: {item.operator}')
    return result

def json_records(frame: pd.DataFrame) -> list[dict[str, int | float | str]]:
    records = []
    for row in frame.to_dict(orient='records'):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = ''
            elif hasattr(value, 'item'):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records

def inspect_dataset() -> DatasetInspection:
    monthly = get_quality_report()['monthly']
    date_min = min(item['min_started_at'] for item in monthly)
    date_max = max(item['max_started_at'] for item in monthly)
    return DatasetInspection(
        row_count=get_quality_report()['total_rows'],
        file_count=get_quality_report()['monthly_file_count'],
        fields=get_dataset().schema.names,
        date_range=(date_min, date_max),
        quality_counts=get_quality_report()['quality_counts'],
        supported_metrics=[metric.value for metric in Metric],
        supported_dimensions=[dimension.value for dimension in Dimension],
    )

def analyze_data(plan: AnalysisPlan) -> AnalysisResult:
    started = time.perf_counter()
    dimensions = [dimension.value for dimension in plan.group_by]
    filter_columns = [item.field.value for item in plan.filters]
    eligibility_column, eligibility_rule = eligibility_for_plan(plan)
    columns = set(dimensions + filter_columns + ['start_year', eligibility_column])
    if plan.metric == Metric.AVERAGE_DURATION_MINUTES:
        columns.add('duration_minutes')

    table = get_dataset().to_table(columns=sorted(columns))
    frame = table.to_pandas()
    frame = frame[frame['start_year'] == 2025]
    frame = apply_filters(frame, plan.filters)
    candidate_rows = len(frame)
    eligible = frame[frame[eligibility_column].fillna(False)]
    included_rows = len(eligible)

    if plan.metric == Metric.TRIP_COUNT:
        grouped = eligible.groupby(dimensions, dropna=False, observed=True).size().rename('trip_count').reset_index()
        value_field = 'trip_count'
    elif plan.metric == Metric.AVERAGE_DURATION_MINUTES:
        grouped = (
            eligible.groupby(dimensions, dropna=False, observed=True)['duration_minutes']
            .mean().rename('average_duration_minutes').reset_index()
        )
        grouped['average_duration_minutes'] = grouped['average_duration_minutes'].round(3)
        value_field = 'average_duration_minutes'
    else:
        raise ValueError(f'Unsupported metric: {plan.metric}')

    ascending = plan.sort_order == SortOrder.ASCENDING
    if plan.limit is None:
        grouped = grouped.sort_values(dimensions, ascending=ascending)
    else:
        grouped = grouped.sort_values(value_field, ascending=ascending).head(plan.limit)

    derived = {}
    if dimensions == ['start_month'] and plan.metric == Metric.TRIP_COUNT and len(grouped) > 1:
        ordered = grouped.sort_values('start_month').copy()
        ordered['change'] = ordered['trip_count'].diff()
        largest_index = ordered['change'].idxmax()
        derived = {
            'largest_increase_month': int(ordered.loc[largest_index, 'start_month']),
            'largest_increase': int(ordered.loc[largest_index, 'change']),
            'previous_month': int(ordered.loc[largest_index, 'start_month']) - 1,
        }

    excluded_rows = candidate_rows - included_rows
    coverage = Coverage(
        candidate_rows=candidate_rows, included_rows=included_rows, excluded_rows=excluded_rows,
        excluded_rate=(excluded_rows / candidate_rows if candidate_rows else 0),
        eligibility_rule=eligibility_rule,
    )
    return AnalysisResult(
        plan=plan, value_field=value_field, rows=json_records(grouped), coverage=coverage,
        derived=derived, elapsed_ms=(time.perf_counter() - started) * 1000,
    )

def create_visualization(result: AnalysisResult, request: ChartRequest) -> ChartResult:
    frame = pd.DataFrame(result.rows)
    required = {request.x.value, request.y}
    if request.series is not None:
        required.add(request.series.value)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Chart fields are absent from the result: {sorted(missing)}')
    if request.y != result.value_field:
        raise ValueError('Chart y field must match the result value field')

    plot_frame = frame.copy()
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if request.series is None:
        ordered = plot_frame.sort_values(request.x.value)
        if request.x == Dimension.START_MONTH:
            ordered[request.x.value] = ordered[request.x.value].map(lambda value: calendar.month_abbr[int(value)])
        if request.chart_type == ChartType.LINE:
            axis.plot(ordered[request.x.value], ordered[request.y], marker='o', linewidth=2)
        else:
            axis.bar(ordered[request.x.value].astype(str), ordered[request.y], color='#1F4E78')
    else:
        pivot = plot_frame.pivot(index=request.x.value, columns=request.series.value, values=request.y).sort_index()
        if request.x == Dimension.START_MONTH:
            pivot.index = [calendar.month_abbr[int(value)] for value in pivot.index]
        if request.chart_type == ChartType.LINE:
            pivot.plot(ax=axis, marker='o', linewidth=2)
            # Pandas may suppress one of two categorical tick labels on a line
            # chart. Pin every month label so short comparisons remain explicit.
            if request.x == Dimension.START_MONTH:
                axis.set_xticks(range(len(pivot.index)))
                axis.set_xticklabels(list(pivot.index))
        else:
            pivot.plot(kind='bar', ax=axis, color=['#1F4E78', '#E07A3F'])

    axis.set_title(request.title, fontsize=13, fontweight='bold')
    axis.set_xlabel(request.x_label)
    axis.set_ylabel(request.y_label)
    axis.grid(axis='y', alpha=0.25)
    if request.series is not None:
        legend_title = 'Rider type' if request.series == Dimension.MEMBER_TYPE else request.series.value.replace('_', ' ').title()
        axis.legend(title=legend_title)
    figure.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / request.filename
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)
    return ChartResult(path=str(path), chart_type=request.chart_type, plotted_rows=len(frame))

def validate_result(
    result: AnalysisResult,
    claims: list[EvidenceClaim],
    evidence: dict[str, int | float | str],
    chart: ChartResult | None = None,
) -> ValidationResult:
    errors, warnings = [], []
    checks = {
        'result_has_rows': bool(result.rows),
        'coverage_reconciles': (
            result.coverage.included_rows + result.coverage.excluded_rows
            == result.coverage.candidate_rows
        ),
        'claims_have_evidence': True,
        'chart_exists': chart is None or Path(chart.path).is_file(),
    }
    for claim in claims:
        if claim.key not in evidence:
            checks['claims_have_evidence'] = False
            errors.append(f'No evidence key for claim: {claim.key}')
            continue
        expected, actual = evidence[claim.key], claim.value
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            matches = abs(float(expected) - float(actual)) <= 1e-6
        else:
            matches = expected == actual
        if not matches:
            checks['claims_have_evidence'] = False
            errors.append(f'Claim {claim.key} does not match evidence')
    if result.coverage.excluded_rows:
        warnings.append(
            f"{result.coverage.excluded_rows:,} rows excluded by: {result.coverage.eligibility_rule}"
        )
    for name, passed in checks.items():
        if not passed and name not in {'claims_have_evidence'}:
            errors.append(f'Failed check: {name}')
    return ValidationResult(valid=all(checks.values()), checks=checks, errors=errors, warnings=warnings)

def trace_event(run_id: str, sequence: int, event_type: str, payload: BaseModel | dict) -> TraceEvent:
    data = payload.model_dump(mode='json') if isinstance(payload, BaseModel) else payload
    return TraceEvent(
        run_id=run_id, sequence=sequence, event_type=event_type,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), payload=data,
    )
