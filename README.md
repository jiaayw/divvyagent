# Divvy Adaptive Data Analysis Agent

**Group members:** Yilun Ai (ya7654), Jiayu Wang (jw628632), and Chengkang Li (cl54722)

This project implements a reliable, adaptive workflow for answering natural-language questions about Chicago Divvy trips. The agent inspects the dataset, uses GPT-5 Nano to select the next action, executes approved Python analysis or visualization tools, validates computed evidence, and produces a concise report.

The package includes the complete 2025 Divvy dataset: 12 monthly files containing 5,552,994 trips. All records are included; no sampling was applied. Model requests use the course API gateway, while calculations run locally on the CPU. A GPU is not required.

## Main Features

- Bounded ReAct-style analysis loop
- Automatic dataset inspection
- Approved computation and visualization tools
- Tool allowlist and Pydantic argument validation
- Evidence-grounded reports and numerical claim validation
- Budgets, retries, legal state transitions, and structured traces
- Live GPT-5 Nano mode and an offline scripted demonstration

## Requirements

- Python 3.11 or newer
- A course Portkey API key for live model runs
- A modern web browser

## macOS Setup and Use

1. Check the installed Python version:

   ```bash
   python3 --version
   ```

   If the version is earlier than 3.11, install a current version from [python.org](https://www.python.org/downloads/) and reopen Terminal.

2. In Terminal, move to the extracted project folder and start the application:

   ```bash
   cd ~/Downloads/divvy-agent
   chmod +x start.sh
   PORT=5055 ./start.sh
   ```

   The first launch installs the required packages and may take several minutes. Port 5055 is recommended because macOS AirPlay may use port 5000.

3. Open [http://127.0.0.1:5055](http://127.0.0.1:5055) in a browser.

4. To use the live model, enter the course API key in the configuration panel and select **Save configuration**. Enter a question and select **Run with live model**.

5. To view the predefined scripted example without an API key, select **Run offline demo**. The offline demonstration verifies the interface and controller flow but does not represent live-model accuracy.

6. Press `Control+C` in Terminal to stop the server.

## Windows Setup and Use

1. Install Python 3.11 or newer and select **Add Python to PATH** during installation.
2. Extract the project and double-click `start.bat`.
3. Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

If port 5000 is unavailable, run the following commands in Command Prompt from the project directory:

```bat
set PORT=5055
start.bat
```

## Configuration

The startup scripts create `.env` from `.env.example` if it does not already exist. Configuration can be entered through the web interface or added directly to `.env`:

```text
OPENAI_API_KEY=your_course_key
OPENAI_BASE_URL=https://api.portkey.ai/v1
OPENAI_MODEL=@30875-fall26-openai/gpt-5-nano
```

Never commit, share, or submit the `.env` file. The submission package should contain only `.env.example`.

## Dataset Setup

The dataset is not stored in the GitHub repository because of its size.

- **Frozen raw and processed project data:** [Google Drive download](https://drive.google.com/file/d/1nNLil6heBD0p16rnb79Y4gxv-GrmKpv-/view?usp=drive_link)
- **Official raw-data source:** [Divvy System Data](https://divvybikes.com/system-data)
- **Data license:** [Divvy Data License Agreement](https://divvybikes.com/data-license-agreement)

The frozen dataset covers January through December 2025 and contains 5,552,994 trip records with no sampling. Station comparisons exclude trips that do not contain the required station name and station ID.

After downloading and extracting the project data, place it in the project directory using this structure:

```text
divvy-agent/
└── data/
    └── processed/
        └── divvy_2025/
            ├── parquet/
            ├── dataset_inventory.json
            ├── manifest.json
            ├── monthly_inventory.csv
            └── quality_report.json
```

The application requires the processed `data/processed/divvy_2025/` directory. The raw monthly files are provided only for reproducibility.

## Evaluation

The final evaluation set contains 30 human-verified cases: 20 adaptive cases, five non-adaptive cases, and five edge cases. It covers multi-step analysis, single-step analysis, charts, ambiguous requests, unsupported requests, and recovery from injected failures. Expected answers were independently reviewed and calculated from the frozen dataset.

Each case was run once with GPT-5 Nano, producing 30 live runs. Runs were scored with six deterministic metrics: task completion, tool-call correctness, trajectory efficiency, grounding and evidence use, recovery quality, and termination quality. Human review was used for failure analysis; no LLM-as-a-judge was used.

The adaptive agent passed 24 of 30 runs (80.0%) and 17 of 20 adaptive cases (85.0%). Its unsupported-claim rate was 0%. The offline reference baselines achieved 26.7% for the keyword router and 10.0% for the fixed one-calculation workflow. Because the baselines are deterministic offline references, they are not directly equivalent to the live-model evaluation.

To repeat the live evaluation after configuring `.env`, run from the project root:

```bash
python evaluation/run_live_eval.py \
  --runs 1 \
  --out-name reproduced_eval \
  --case-file eval_set.csv \
  --oracle-file oracle_values_eval_set.csv
```

To reproduce the offline baseline comparison, run:

```bash
python evaluation/run_baselines.py \
  --case-file eval_set.csv \
  --oracle-file oracle_values_eval_set.csv \
  --out-name reproduced_baselines
```

## Important Files

- `app.py`: Flask web interface and application entry point
- `src/divvy_agent/`: agent controller, schemas, model adapter, dispatcher, and tools
- `data/processed/divvy_2025/`: frozen 2025 Divvy dataset
- `evaluation/eval_set.csv`: 30 final evaluation questions and case requirements
- `evaluation/oracle_values_eval_set.csv`: expected values computed from the frozen dataset
- `evaluation/live_eval_set_r2/`: structured traces and charts from all 30 live runs
- `evaluation/demo_adaptive_trace.json`: complete machine-readable trace from the recorded UI demonstration
- `evaluation/representative_adaptive_trace.pdf`: visual J04 adaptive-trace summary; the complete machine-readable trace remains in the live-run folder
- `evaluation/live_eval_set_r2_case_results.csv`: case-level live results
- `evaluation/live_eval_set_r2_summary.csv`: aggregate live metrics and resource use
- `evaluation/live_eval_set_r2_failure_taxonomy.csv`: final live-run failure categories
- `evaluation/baseline_eval_set_case_results.csv`: case-level baseline results
- `evaluation/baseline_eval_set_summary.csv`: aggregate baseline comparison
- `evaluation/baseline_eval_set_failure_taxonomy.csv`: baseline failure categories
- `evaluation/grader_rubric.csv`: deterministic metric definitions

## Troubleshooting

- **Python 3.11 or newer is required:** Install a newer Python version and reopen the terminal.
- **Address already in use:** Change the port, for example from 5055 to 5056.
- **Invalid API key:** Re-enter the course API key in the web interface or update `OPENAI_API_KEY` in `.env`.
- **Another run is in progress:** Wait for the current run to finish. The application permits one run at a time because chart generation is not thread-safe.

Live runs normally take approximately 40–60 seconds. Actual time and API cost depend on the selected model, tool calls, retries, and response length.
