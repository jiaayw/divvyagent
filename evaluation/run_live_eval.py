"""Run the evaluation cases against the live model and score them with the project's rubric.

Usage:
    python evaluation/run_live_eval.py [--project <divvy-agent dir>] [--runs 3] [--cases J01,Y05]
        [--case-file evaluation_cases.csv] [--oracle-file oracle_values.csv]

Calls the project's real run_agent() with OpenAIModelAdapter (never the mock). Each run is saved
to <project>/evaluation/<out-name>/runs/<case>_r<n>.json with its trace and charts; existing run files
are skipped, so an interrupted evaluation can be resumed by running the same command again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--out-name', default='live', help='output folder/file prefix, e.g. live_v2 to keep results apart')
parser.add_argument('--cases', default='')
parser.add_argument('--case-file', default='evaluation_cases.csv', help='case CSV filename inside evaluation/')
parser.add_argument('--oracle-file', default='oracle_values.csv', help='oracle CSV filename inside evaluation/')
args = parser.parse_args()

PROJECT = Path(args.project).resolve()
sys.path.insert(0, str(PROJECT / 'src'))
load_dotenv(PROJECT / '.env', override=True)

from divvy_agent import (  # noqa: E402
    AgentActionType, AgentLimits, AgentStep, ModelUsage,
    OpenAIModelAdapter, run_agent,
)

EVAL = PROJECT / 'evaluation'
LIVE = EVAL / args.out_name
RUNS = LIVE / 'runs'
CHARTS = LIVE / 'charts'
RUNS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv('OPENAI_MODEL', '@30875-fall26-openai/gpt-5-nano')
BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.portkey.ai/v1')
IN_COST = float(os.getenv('OPENAI_INPUT_COST_PER_MILLION', 0.05))
OUT_COST = float(os.getenv('OPENAI_OUTPUT_COST_PER_MILLION', 0.40))
if not os.getenv('OPENAI_API_KEY'):
    sys.exit('OPENAI_API_KEY is not set in .env')

# Web-app limits, with more token headroom and a second report revision for the nano model.
LIMITS = dict(max_model_turns=5, max_tool_calls=6, max_elapsed_seconds=120,
              max_total_tokens=45000, max_report_revisions=2, max_estimated_cost_usd=0.25)

ALLOWED_TOOLS = {'analyze_data', 'create_visualization'}


class FailureInjectingAdapter:
    """Apply the case's declared deterministic failure once, then delegate normally."""

    def __init__(self, inner, injection: str):
        self.inner = inner
        self.injection = injection
        self.injected = False

    def decide(self, context):
        if not self.injected and self.injection == 'transient_model_timeout_once':
            self.injected = True
            raise TimeoutError('Injected transient model timeout for recovery evaluation')
        if not self.injected and self.injection == 'invalid_structured_arguments_once':
            self.injected = True
            return AgentStep(
                action=AgentActionType.EXECUTE_ANALYSIS,
                observation='Inject one invalid analysis request for the recovery case.',
                reason='Deterministic evaluation-harness failure injection.',
                expected_information_gain='Exercise bounded invalid-argument recovery.',
                analysis_plan={
                    'objective': 'Injected invalid analysis request',
                    'metric': 'unsupported_metric',
                    'group_by': ['start_month'],
                    'filters': [],
                },
            ), ModelUsage()
        return self.inner.decide(context)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


case_path = EVAL / args.case_file
oracle_path = EVAL / args.oracle_file
if not case_path.is_file():
    sys.exit(f'Evaluation case file not found: {case_path}')
if not oracle_path.is_file():
    sys.exit(f'Oracle file not found: {oracle_path}')

cases = read_csv(case_path)
oracle: dict[str, list[dict]] = {}
for row in read_csv(oracle_path):
    oracle.setdefault(row['case_id'], []).append(row)
if args.cases:
    wanted = set(args.cases.split(','))
    cases = [c for c in cases if c['case_id'] in wanted]


def row_matches(selector: dict, row: dict, plan: dict) -> bool:
    for key, want in selector.items():
        if key in row:
            if str(row[key]) != str(want):
                return False
            continue
        ok = False
        for flt in plan.get('filters', []):
            if flt['field'] != key:
                continue
            value = flt['value']
            ok = str(want) in {str(v) for v in value} if isinstance(value, list) else str(value) == str(want)
        if not ok:
            return False
    return True


def oracle_misses(case_id: str, results: list[dict]) -> list[str]:
    misses = []
    for item in oracle.get(case_id, []):
        selector = json.loads(item['selector'])
        expected = float(item['expected_value'])
        found = False
        for res in results:
            if res['value_field'] != item['field']:
                continue
            matching = [row for row in res['rows'] if row_matches(selector, row, res['plan'])]
            if any(abs(float(row[res['value_field']]) - expected) <= 0.0006 for row in matching):
                found = True
            # A trip-count total may be reported as a breakdown; accept the sum of the matching rows.
            if (item['field'] == 'trip_count' and matching and selector
                    and abs(sum(float(row['trip_count']) for row in matching) - expected) <= 0.0006):
                found = True
        if not found:
            misses.append(f"target {item['target']} {item['selector']} {item['field']}={expected}")
    return misses


def observation_driven_adaptation(trace: list[dict]) -> bool:
    """Require a successful observation and a reasoned decision between two analyses."""
    analysis_calls = [
        event for event in trace
        if event['event_type'] == 'tool_call'
        and event['payload'].get('tool') == 'analyze_data'
    ]
    if len(analysis_calls) < 2:
        return False
    first_sequence = analysis_calls[0]['sequence']
    second_sequence = analysis_calls[1]['sequence']
    observed = any(
        first_sequence < event['sequence'] < second_sequence
        and event['event_type'] == 'observation'
        and event['payload'].get('ok')
        and event['payload'].get('result_type') == 'AnalysisResult'
        for event in trace
    )
    continued = any(
        first_sequence < event['sequence'] < second_sequence
        and event['event_type'] == 'decision'
        and event['payload'].get('action') in {'execute_analysis', 'continue'}
        and bool(event['payload'].get('reason'))
        for event in trace
    )
    return observed and continued


def score(case: dict, run: dict) -> dict:
    trace = run['trace']
    tool_events = [e for e in trace if e['event_type'] == 'tool_call']
    observations = [e['payload'] for e in trace if e['event_type'] == 'observation']
    tool_obs = [o for o in observations if 'tool_name' in o and 'ok' in o]
    results = [o['result'] for o in tool_obs if o.get('ok') and o.get('result_type') == 'AnalysisResult']
    analysis_calls = sum(1 for e in tool_events if e['payload'].get('tool') == 'analyze_data')
    model_errors = sum(1 for o in observations if 'error_type' in o and o['error_type'] != 'duplicate_chart_blocked')
    tool_errors = sum(1 for o in tool_obs if not o.get('ok'))
    retries = model_errors + sum(e['payload'].get('retry_number', 0) for e in tool_events)
    validations = [e['payload'] for e in trace if e['event_type'] == 'validation']
    last_validation = validations[-1] if validations else None

    expected_status = case['expected_terminal_status']
    status_ok = run['status'] == expected_status
    must_calc = expected_status == 'completed'
    misses = oracle_misses(case['case_id'], results) if must_calc else []
    values_ok = must_calc and not misses and bool(results) if must_calc else True
    adapt_needed = case['adaptation_required'].strip().lower() == 'true'
    adaptation_ok = (not adapt_needed) or observation_driven_adaptation(trace)
    max_calls = int(case['maximum_analysis_calls'])

    unsupported = 0
    if last_validation and run['status'] == 'completed':
        unsupported = len(last_validation['evidence_validation'].get('errors', []))
    elif last_validation and not last_validation['completion_gate']['checks'].get('claims_match_evidence', True):
        unsupported = len(last_validation['evidence_validation'].get('errors', []))

    task_completion = status_ok and values_ok and adaptation_ok
    answer_correct = status_ok and values_ok
    tools_used = {e['payload'].get('tool') for e in tool_events}
    required_dispatched_tools = {
        name.strip() for name in case['required_tools'].split(',')
        if name.strip() in ALLOWED_TOOLS
    }
    chart_required = 'create_visualization' in required_dispatched_tools
    chart_calls = sum(1 for e in tool_events if e['payload'].get('tool') == 'create_visualization')
    chart_ok = chart_calls >= 1 if chart_required else chart_calls == 0
    tool_correct = (
        tools_used <= ALLOWED_TOOLS
        and required_dispatched_tools <= tools_used
        and ((analysis_calls >= 1) == must_calc)
        and chart_ok
    )
    efficient = analysis_calls <= max_calls and run['status'] != 'budget_exhausted'
    grounded = values_ok and unsupported == 0 and (must_calc or analysis_calls == 0)
    termination = status_ok and bool(run['stop_reason']) and analysis_calls <= max_calls
    recovery = None
    if retries or tool_errors:
        recovery = run['status'] == expected_status

    checks = [('task_completion_failure', task_completion), ('tool_call_error', tool_correct),
              ('trajectory_failure', efficient), ('grounding_failure', grounded),
              ('termination_failure', termination)]
    failures = [name for name, ok in checks if not ok]
    return dict(
        task_completion=task_completion, tool_call_correctness=tool_correct,
        trajectory_efficiency=efficient, grounding_evidence_use=grounded,
        recovery_quality='' if recovery is None else recovery, termination_quality=termination,
        answer_correct=answer_correct, values_correct=values_ok, adaptation_correct=adaptation_ok, unsupported_claims=unsupported,
        passed=not failures, failure_type=failures[0] if failures else 'none',
        oracle_misses='; '.join(misses), model_retries=model_errors, tool_errors=tool_errors,
        retries=retries, analysis_calls=analysis_calls,
    )


def run_one(case: dict, n: int) -> dict:
    path = RUNS / f"{case['case_id']}_r{n}.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    adapter = OpenAIModelAdapter(model=MODEL, input_cost_per_million=IN_COST,
                                 output_cost_per_million=OUT_COST, base_url=BASE_URL)
    injection = case.get('failure_injection', 'none')
    if injection != 'none':
        adapter = FailureInjectingAdapter(adapter, injection)
    started = time.perf_counter()
    try:
        run = run_agent(case['question'], adapter, AgentLimits(**LIMITS)).model_dump(mode='json')
        error = None
    except Exception as exc:  # the controller handles model errors itself; this is a crash
        run = dict(run_id='', status='failed', final_report=None, user_message=None, trace=[],
                   model_turns=0, tool_calls=0, total_tokens=0, estimated_cost_usd=0,
                   elapsed_seconds=round(time.perf_counter() - started, 3),
                   stop_reason='harness_exception', phase_history=[])
        error = f'{type(exc).__name__}: {exc}'
    charts = []
    for event in run['trace']:
        p = event['payload']
        if event['event_type'] == 'observation' and p.get('result_type') == 'ChartResult':
            src = Path(p['result']['path'])
            if src.exists():
                dest = CHARTS / f"{case['case_id']}_r{n}_{src.name}"
                shutil.copy(src, dest)
                charts.append(str(dest.relative_to(PROJECT)))
    record = dict(case_id=case['case_id'], run_number=n, question=case['question'], model=MODEL,
                  harness_error=error, charts=charts, **run)
    record['score'] = score(case, record)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return record


FIELDS = ['case_id', 'run_number', 'status', 'expected_status', 'passed', 'failure_type', 'task_completion',
          'tool_call_correctness', 'trajectory_efficiency', 'grounding_evidence_use', 'recovery_quality',
          'termination_quality', 'answer_correct', 'values_correct', 'adaptation_correct', 'unsupported_claims',
          'oracle_misses', 'stop_reason', 'model_turns', 'tool_calls', 'analysis_calls', 'retries',
          'total_tokens', 'estimated_cost_usd', 'elapsed_seconds', 'charts', 'harness_error']

rows = []
total = len(cases) * args.runs
done = 0
for case in cases:
    for n in range(1, args.runs + 1):
        done += 1
        record = run_one(case, n)
        s = record['score']
        print(f"[{done}/{total}] {case['case_id']} r{n}: {record['status']} "
              f"{'PASS' if s['passed'] else 'FAIL ' + s['failure_type']} "
              f"{record['total_tokens']} tok ${record['estimated_cost_usd']:.4f} {record['elapsed_seconds']:.0f}s",
              flush=True)
        rows.append({**{k: record.get(k, '') for k in FIELDS}, **s,
                     'expected_status': case['expected_terminal_status'],
                     'charts': ' | '.join(record['charts'])})
        with open(EVAL / f'{args.out_name}_case_results.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)


def rate(key, subset=None):
    vals = [r[key] for r in (subset if subset is not None else rows) if r[key] != '']
    return round(sum(1 for v in vals if v in (True, 'True')) / len(vals), 4) if vals else ''


def mean(key, subset=None):
    vals = [float(r[key]) for r in (subset if subset is not None else rows)]
    return round(sum(vals) / len(vals), 4) if vals else ''


adaptive_ids = {
    c['case_id'] for c in cases
    if c['adaptation_required'].strip().lower() == 'true'
}
adaptive = [r for r in rows if r['case_id'] in adaptive_ids]
by_case: dict[str, list] = {}
for r in rows:
    by_case.setdefault(r['case_id'], []).append(r)
failures: dict[str, int] = {}
for r in rows:
    if r['failure_type'] != 'none':
        failures[r['failure_type']] = failures.get(r['failure_type'], 0) + 1

summary = {
    'system': f'adaptive_agent_{args.out_name} ({MODEL})', 'cases': len(cases), 'runs_per_case': args.runs, 'total_runs': len(rows),
    'overall_accuracy': rate('passed'), 'answer_accuracy_ignoring_adaptation_rule': rate('answer_correct'), 'adaptive_case_accuracy': rate('passed', adaptive),
    'task_completion_rate': rate('task_completion'), 'tool_call_correctness_rate': rate('tool_call_correctness'),
    'trajectory_efficiency_rate': rate('trajectory_efficiency'), 'grounding_evidence_rate': rate('grounding_evidence_use'),
    'recovery_rate': rate('recovery_quality'), 'termination_quality_rate': rate('termination_quality'),
    'unsupported_claim_rate': round(sum(1 for r in rows if int(r['unsupported_claims']) > 0) / len(rows), 4) if rows else '',
    'cases_passing_every_run': sum(1 for v in by_case.values() if all(r['passed'] in (True, 'True') for r in v)),
    'average_tool_calls': mean('tool_calls'), 'average_analysis_calls': mean('analysis_calls'),
    'average_retries': mean('retries'), 'average_tokens': mean('total_tokens'),
    'average_cost_usd': mean('estimated_cost_usd'), 'total_cost_usd': round(sum(float(r['estimated_cost_usd']) for r in rows), 4),
    'average_latency_seconds': mean('elapsed_seconds'),
    'max_latency_seconds': max((float(r['elapsed_seconds']) for r in rows), default=''),
    'failures': sum(failures.values()),
}
with open(EVAL / f'{args.out_name}_summary.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(summary))
    w.writeheader()
    w.writerow(summary)
with open(EVAL / f'{args.out_name}_failure_taxonomy.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['system', 'failure_type', 'count'])
    for k, v in sorted(failures.items()):
        w.writerow([f'adaptive_agent_{args.out_name}', k, v])
print(json.dumps(summary, indent=2))
