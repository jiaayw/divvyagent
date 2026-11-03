"""Scripted offline demo: a MockModelAdapter that replays the primary demonstration question.

This exercises the real controller, tools, dispatcher and completion gate, but the
decisions are scripted. Results must not be presented as live-model performance.
"""

from __future__ import annotations

from .agent import AgentStep, MockModelAdapter
from .schemas import AnalysisPlan, ChartRequest, FinalReport

DEMO_QUESTION = (
    'Which month had the largest increase in Divvy trips, and was it driven mainly by '
    'members or casual riders? Create a chart comparing member and casual trip counts '
    'for the relevant months.'
)


def demo_adapter() -> MockModelAdapter:
    initial_plan = AnalysisPlan(
        objective='Count trips by start month and identify the largest consecutive increase.',
        metric='trip_count', group_by=['start_month'], sort_order='ascending',
    )
    follow_up_plan = AnalysisPlan(
        objective='Break down the May-to-June increase by rider type.',
        metric='trip_count', group_by=['start_month', 'member_casual'],
        filters=[{'field': 'start_month', 'operator': 'in', 'value': [5, 6]}],
        sort_order='ascending',
    )
    steps = [
        AgentStep(
            action='execute_analysis', observation='No calculation has been run.',
            reason='The request first requires monthly trip totals.',
            expected_information_gain='Identify the largest consecutive monthly increase.',
            analysis_plan=initial_plan,
        ),
        AgentStep(
            action='execute_analysis',
            observation='The largest increase is from May to June and totals 176,186 trips.',
            reason='The aggregate result does not identify which rider type drove the increase.',
            expected_information_gain='Measure the member and casual contributions.',
            analysis_plan=follow_up_plan,
        ),
        AgentStep(
            action='create_chart',
            observation='The rider-type breakdown answers the attribution requested.',
            reason='A grouped bar chart clearly compares May and June counts by rider type.',
            chart_request=ChartRequest(
                chart_type='bar', x='start_month', y='trip_count', series='member_casual',
                title='Divvy Trip Increase by Rider Type', x_label='Start month', y_label='Trips',
                filename='divvy_agent_may_june_breakdown.png',
            ),
            source_result_index=1,
        ),
        AgentStep(
            action='report', observation='The follow-up and chart are complete.',
            reason='The available evidence fully answers the request.',
            final_report=FinalReport(
                status='completed',
                answer=(
                    'Trips increased most from May to June, rising by 176,186. Casual riders '
                    'contributed 109,236 trips, or 62.0% of the increase, compared with 66,950 '
                    'from members. The increase was mainly driven by casual riders.'
                ),
                claims=[
                    {'key': 'result_1.derived.total_increase', 'value': 176186, 'unit': 'trips'},
                    {'key': 'result_1.derived.casual_increase', 'value': 109236, 'unit': 'trips'},
                    {'key': 'result_1.derived.member_increase', 'value': 66950, 'unit': 'trips'},
                    {'key': 'result_1.derived.casual_share_percent', 'value': 62.0, 'unit': 'percent'},
                ],
                evidence_tools=['inspect_dataset', 'analyze_data', 'create_visualization', 'validate_result'],
                answered_requirements=[
                    'Identify the largest monthly increase.',
                    'Identify whether members or casual riders drove the increase.',
                ],
                unresolved_requirements=[],
                limitations=['Month is assigned from started_at.', 'The data does not establish why demand changed.'],
                stop_reason='The initial result and justified follow-up answer the request, and evidence validation passed.',
            ),
        ),
    ]
    return MockModelAdapter(steps)
