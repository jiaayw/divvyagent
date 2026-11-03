"""
Divvy Adaptive Agent web interface.

Run with `python app.py` and open http://127.0.0.1:5000.
"""

import json
import os
import sys
import threading
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend: charts are saved to files, never shown

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
ENV_FILE = PROJECT_ROOT / '.env'
CHART_DIR = PROJECT_ROOT / 'output' / 'charts'
RUN_DIR = PROJECT_ROOT / 'output' / 'runs'

# The project's .env must win over keys exported in the shell (e.g. a personal OpenAI key in ~/.bashrc)
load_dotenv(ENV_FILE, override=True)

from divvy_agent import AgentLimits, AnalysisRequest, OpenAIModelAdapter, run_agent  # noqa: E402
from divvy_agent.demo import DEMO_QUESTION, demo_adapter  # noqa: E402

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

# matplotlib's pyplot is not thread-safe, so only one agent run executes at a time.
RUN_LOCK = threading.Lock()

# Course gateway (Portkey). Clear the base URL in the UI to use api.openai.com directly.
DEFAULT_MODEL = '@30875-fall26-openai/gpt-5-nano'
DEFAULT_BASE_URL = 'https://api.portkey.ai/v1'
DEFAULT_INPUT_COST = 0.05   # USD per 1M tokens, gpt-5-nano list price; editable in the UI
DEFAULT_OUTPUT_COST = 0.40


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@app.route('/')
def index():
    return render_template('index.html', demo_question=DEMO_QUESTION)


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'api_key_set': bool(os.getenv('OPENAI_API_KEY')),
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'base_url': os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL),
        'input_cost': env_float('OPENAI_INPUT_COST_PER_MILLION', DEFAULT_INPUT_COST),
        'output_cost': env_float('OPENAI_OUTPUT_COST_PER_MILLION', DEFAULT_OUTPUT_COST),
    })


def clean_env_value(value) -> str:
    text = str(value).strip()
    if '\n' in text or '\r' in text:
        raise ValueError('Values must be single-line')
    return text


@app.route('/api/config', methods=['POST'])
def set_config():
    """Save only the settings that were provided to .env and to the running process."""
    data = request.get_json(silent=True) or {}
    updates = {}
    try:
        if data.get('api_key'):
            updates['OPENAI_API_KEY'] = clean_env_value(data['api_key'])
        if data.get('model'):
            updates['OPENAI_MODEL'] = clean_env_value(data['model'])
        if data.get('base_url') is not None:  # an empty string selects api.openai.com
            updates['OPENAI_BASE_URL'] = clean_env_value(data['base_url'])
        for field, key in (('input_cost', 'OPENAI_INPUT_COST_PER_MILLION'),
                           ('output_cost', 'OPENAI_OUTPUT_COST_PER_MILLION')):
            if data.get(field) not in (None, ''):
                updates[key] = str(float(data[field]))
    except ValueError as error:
        return jsonify({'error': f'Invalid configuration value: {error}'}), 400

    lines = ENV_FILE.read_text(encoding='utf-8').splitlines() if ENV_FILE.exists() else []
    for key, value in updates.items():
        for i, line in enumerate(lines):
            if line.startswith(f'{key}='):
                lines[i] = f'{key}={value}'
                break
        else:
            lines.append(f'{key}={value}')
        os.environ[key] = value
    ENV_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return jsonify({'success': True, 'updated': sorted(updates)})


def build_limits(data: dict) -> AgentLimits:
    return AgentLimits(
        max_model_turns=int(data.get('max_turns', 5)),
        max_tool_calls=int(data.get('max_tool_calls', 6)),
        max_elapsed_seconds=float(data.get('max_seconds', 120)),
        max_total_tokens=int(data.get('max_tokens', 45000)),
        max_report_revisions=2,
        max_estimated_cost_usd=float(data.get('max_cost', 0.25)),
    )


@app.route('/api/run', methods=['POST'])
def run_agent_endpoint():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'live')
    question = str(data.get('question', '')).strip()
    if mode == 'mock' and not question:
        question = DEMO_QUESTION

    try:
        AnalysisRequest(question=question)
        limits = build_limits(data)
    except (ValidationError, ValueError) as error:
        return jsonify({'error': f'Invalid request: {error}'}), 400

    if mode == 'live':
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({'error': 'No API key is configured. Save your key (OpenAI or the course Portkey key) or use the offline demo.'}), 400
        adapter_factory = lambda: OpenAIModelAdapter(
            model=os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
            input_cost_per_million=env_float('OPENAI_INPUT_COST_PER_MILLION', DEFAULT_INPUT_COST),
            output_cost_per_million=env_float('OPENAI_OUTPUT_COST_PER_MILLION', DEFAULT_OUTPUT_COST),
            base_url=os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL),
        )
    elif mode == 'mock':
        adapter_factory = demo_adapter
    else:
        return jsonify({'error': f'Unknown mode: {mode}'}), 400

    if not RUN_LOCK.acquire(blocking=False):
        return jsonify({'error': 'Another run is in progress. Try again when it finishes.'}), 409
    try:
        result = run_agent(question, adapter_factory(), limits)
    except Exception as error:
        return jsonify({'error': str(error), 'type': type(error).__name__}), 500
    finally:
        RUN_LOCK.release()

    payload = result.model_dump(mode='json')
    payload['mode'] = mode
    payload['model'] = os.getenv('OPENAI_MODEL', DEFAULT_MODEL) if mode == 'live' else 'scripted mock'
    payload['charts'] = [
        Path(event['payload']['result']['path']).name
        for event in payload['trace']
        if event['event_type'] == 'observation'
        and event['payload'].get('result_type') == 'ChartResult'
    ]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{payload['run_id']}.json"
    payload['artifact_file'] = artifact_name
    (RUN_DIR / artifact_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return jsonify(payload)


@app.route('/api/charts/<path:filename>')
def serve_chart(filename):
    if not filename.endswith('.png'):
        abort(404)
    return send_from_directory(CHART_DIR, filename, mimetype='image/png')


if __name__ == '__main__':
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.getenv('PORT', '5000'))  # set PORT if 5000 is taken (macOS AirPlay uses it)
    print(f'Starting Divvy Adaptive Agent web UI at http://127.0.0.1:{port}')
    print(f'API key configured: {bool(os.getenv("OPENAI_API_KEY"))}')
    print(f'Model: {os.getenv("OPENAI_MODEL", DEFAULT_MODEL)}')
    print(f'Base URL: {os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL) or "api.openai.com"}')
    app.run(debug=False, host='127.0.0.1', port=port)
