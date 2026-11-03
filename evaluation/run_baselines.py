"""Evaluate the keyword-router and fixed one-calculation baselines on all 30 cases.

The baselines are intentionally non-adaptive. Each may execute at most one analysis.
The keyword router can safely classify a small allowlist of ambiguous or unsupported
requests; the fixed workflow cannot handle non-analysis requests.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EVAL = PROJECT / "evaluation"
sys.path.insert(0, str(PROJECT / "src"))

from divvy_agent.schemas import AnalysisPlan  # noqa: E402
from divvy_agent.tools import analyze_data  # noqa: E402


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


CASES = read_csv(EVAL / "eval_set.csv")
ORACLE: dict[str, list[dict]] = defaultdict(list)
for oracle_row in read_csv(EVAL / "oracle_values_eval_set.csv"):
    ORACLE[oracle_row["case_id"]].append(oracle_row)


PLAN_SPECS = {
    "J01": ("trip_count", ["start_month"], []),
    "J02": ("trip_count", ["start_month"], []),
    "J03": ("trip_count", ["start_weekday"], []),
    "J04": ("trip_count", ["start_month"], []),
    "J05": ("average_duration_minutes", ["start_month"], []),
    "J06": ("trip_count", ["start_month"], [("member_casual", "member")]),
    "Y01": ("average_duration_minutes", ["member_casual"], []),
    "Y02": ("average_duration_minutes", ["member_casual"], []),
    "Y03": ("trip_count", ["start_month"], []),
    "Y04": ("trip_count", ["member_casual"], [("start_month", 6)]),
    "Y05": ("trip_count", ["start_station_name"], []),
    "Y06": ("average_duration_minutes", ["start_weekday"], []),
    "Y08": ("trip_count", ["start_month"], [("member_casual", "member")]),
    "Y09": ("average_duration_minutes", ["start_weekday"], []),
    "Y10": ("trip_count", ["end_station_name"], [("member_casual", "member")]),
    "C02": ("trip_count", ["start_month"], [("member_casual", "casual")]),
    "C03": ("trip_count", ["end_station_name"], []),
    "C06": ("trip_count", ["start_month"], [("rideable_type", "classic_bike")]),
    "C08": ("trip_count", ["start_weekday"], []),
    "C10": ("trip_count", ["member_casual"], [("rideable_type", "electric_bike")]),
    "H05": ("average_duration_minutes", ["start_month"], [("rideable_type", "electric_bike")]),
    "H06": ("average_duration_minutes", ["start_month"], [("member_casual", "member")]),
    "H07": ("trip_count", ["start_weekday"], [("rideable_type", "classic_bike")]),
    "H08": ("trip_count", ["end_station_name"], [("member_casual", "casual")]),
    "Y11": ("trip_count", ["start_hour"], [("member_casual", "casual")]),
}


def plan_for(case_id: str) -> AnalysisPlan | None:
    spec = PLAN_SPECS.get(case_id)
    if spec is None:
        return None
    metric, group_by, filters = spec
    limit = 1 if group_by[0] in {"start_station_name", "end_station_name"} else None
    return AnalysisPlan.model_validate({
        "objective": f"Baseline first calculation for {case_id}",
        "metric": metric,
        "group_by": group_by,
        "filters": [
            {"field": field, "operator": "equals", "value": value}
            for field, value in filters
        ],
        "sort_order": "descending" if limit else "ascending",
        "limit": limit,
    })


RESULT_CACHE: dict[str, dict] = {}


def execute(plan: AnalysisPlan) -> dict:
    signature = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    if signature not in RESULT_CACHE:
        RESULT_CACHE[signature] = analyze_data(plan).model_dump(mode="json")
    return RESULT_CACHE[signature]


def row_matches(selector: dict, row: dict, plan: dict) -> bool:
    for key, wanted in selector.items():
        if key in row:
            if str(row[key]) != str(wanted):
                return False
            continue
        matched_filter = False
        for item in plan.get("filters", []):
            if item["field"] != key:
                continue
            value = item["value"]
            matched_filter = (
                str(wanted) in {str(member) for member in value}
                if isinstance(value, list)
                else str(value) == str(wanted)
            )
        if not matched_filter:
            return False
    return True


def oracle_misses(case_id: str, results: list[dict]) -> list[str]:
    misses = []
    for target in ORACLE.get(case_id, []):
        selector = json.loads(target["selector"])
        expected = float(target["expected_value"])
        found = False
        for result in results:
            if result["value_field"] != target["field"]:
                continue
            matching = [
                row for row in result["rows"]
                if row_matches(selector, row, result["plan"])
            ]
            if any(
                abs(float(row[result["value_field"]]) - expected) <= 0.0006
                for row in matching
            ):
                found = True
            if (
                target["field"] == "trip_count"
                and matching
                and selector
                and abs(sum(float(row["trip_count"]) for row in matching) - expected) <= 0.0006
            ):
                found = True
        if not found:
            misses.append(
                f"target {target['target']} {target['selector']} "
                f"{target['field']}={target['expected_value']}"
            )
    return misses


def keyword_terminal(question: str) -> tuple[str, str] | None:
    text = question.lower()
    tokens = set(re.findall(r"\b[a-z0-9]+\b", text))
    years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text)}
    if tokens.intersection({"age", "weather", "rain", "median", "pie"}) or (years and years != {2025}):
        return "blocked", "keyword_out_of_scope"
    if tokens.intersection({"important", "best"}):
        return "clarification_needed", "keyword_ambiguous"
    return None


def run_case(case: dict, system: str) -> dict:
    started = time.perf_counter()
    expected_status = case["expected_terminal_status"]
    result = None
    if system == "keyword_router":
        terminal = keyword_terminal(case["question"])
        if terminal is not None:
            status, stop_reason = terminal
        else:
            status, stop_reason = "completed", "one_keyword_route"
            plan = plan_for(case["case_id"])
            if plan is not None:
                result = execute(plan)
    else:
        if expected_status != "completed":
            status, stop_reason = "failed", "cannot_handle_non_analysis_request"
        else:
            status, stop_reason = "completed", "fixed_single_calculation"
            plan = plan_for(case["case_id"])
            if plan is not None:
                result = execute(plan)

    results = [] if result is None else [result]
    analysis_calls = len(results)
    tool_calls = 1 + analysis_calls
    misses = oracle_misses(case["case_id"], results) if expected_status == "completed" else []
    must_calculate = expected_status == "completed"
    values_correct = (bool(results) and not misses) if must_calculate else True
    adaptation_required = case["adaptation_required"].strip().lower() == "true"
    adaptation_correct = not adaptation_required
    status_correct = status == expected_status
    task_completion = status_correct and values_correct and adaptation_correct

    used_tools = {"inspect_dataset"}
    if results:
        used_tools.add("analyze_data")
    required_tools = {
        value.strip() for value in case["required_tools"].split(",") if value.strip()
    }
    prohibited_tools = {
        value.strip() for value in case["prohibited_tools"].split(",") if value.strip()
    }
    tool_call_correctness = (
        required_tools <= used_tools and used_tools.isdisjoint(prohibited_tools)
    )
    trajectory_efficiency = (
        analysis_calls <= int(case["maximum_analysis_calls"])
    )
    grounding = (
        values_correct
        and (must_calculate or (status_correct and analysis_calls == 0))
    )
    termination = (
        status_correct and bool(stop_reason)
        and analysis_calls <= int(case["maximum_analysis_calls"])
    )
    passed = all([
        task_completion, tool_call_correctness, trajectory_efficiency,
        grounding, termination,
    ])
    checks = [
        ("task_completion_failure", task_completion),
        ("tool_call_error", tool_call_correctness),
        ("trajectory_failure", trajectory_efficiency),
        ("grounding_failure", grounding),
        ("termination_failure", termination),
    ]
    failure_type = next((name for name, ok in checks if not ok), "none")
    return {
        "case_id": case["case_id"],
        "system": system,
        "status": status,
        "expected_status": expected_status,
        "passed": passed,
        "failure_type": failure_type,
        "task_completion": task_completion,
        "tool_call_correctness": tool_call_correctness,
        "trajectory_efficiency": trajectory_efficiency,
        "grounding_evidence_use": grounding,
        "recovery_quality": "",
        "termination_quality": termination,
        "values_correct": values_correct,
        "adaptation_correct": adaptation_correct,
        "unsupported_or_missing_targets": len(misses),
        "oracle_misses": "; ".join(misses),
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
        "analysis_calls": analysis_calls,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


SYSTEMS = ["keyword_router", "fixed_one_calculation"]
DETAILS = [run_case(case, system) for system in SYSTEMS for case in CASES]


def rate(rows: list[dict], key: str):
    values = [row[key] for row in rows if row[key] != ""]
    return round(sum(bool(value) for value in values) / len(values), 4) if values else ""


SUMMARIES = []
for system in SYSTEMS:
    rows = [row for row in DETAILS if row["system"] == system]
    adaptive = [
        row for row in rows
        if next(
            case for case in CASES if case["case_id"] == row["case_id"]
        )["adaptation_required"].strip().lower() == "true"
    ]
    SUMMARIES.append({
        "system": system,
        "cases": len(rows),
        "overall_accuracy": rate(rows, "passed"),
        "adaptive_case_accuracy": rate(adaptive, "passed"),
        "task_completion_rate": rate(rows, "task_completion"),
        "tool_call_correctness_rate": rate(rows, "tool_call_correctness"),
        "trajectory_efficiency_rate": rate(rows, "trajectory_efficiency"),
        "grounding_evidence_rate": rate(rows, "grounding_evidence_use"),
        "recovery_rate": "",
        "termination_quality_rate": rate(rows, "termination_quality"),
        "average_tool_calls": round(sum(row["tool_calls"] for row in rows) / len(rows), 4),
        "average_analysis_calls": round(sum(row["analysis_calls"] for row in rows) / len(rows), 4),
        "failures": sum(not row["passed"] for row in rows),
        "evaluation_mode": "offline deterministic; one run per case",
    })

FAILURES = []
for system in SYSTEMS:
    counts = Counter(
        row["failure_type"] for row in DETAILS
        if row["system"] == system and row["failure_type"] != "none"
    )
    for failure_type, count in sorted(counts.items()):
        FAILURES.append({
            "system": system,
            "failure_type": failure_type,
            "count": count,
        })

payload = {
    "details": DETAILS,
    "summaries": SUMMARIES,
    "failure_taxonomy": FAILURES,
}
(EVAL / "baseline_eval_set_structured.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(SUMMARIES, indent=2))
