# Towards the "LLM Contract"? Implications from Simulated Negotiations between Large Language Models

This repository contains the support material for the paper Towards the ``LLM Contract''? Implications from Simulated Negotiations between Large Language Models.<br />

main_simulation.py - contains the script with main simulation and system prompts<br />
analyze_results.py - contains main data analysis script<br />
stat.ipynb - contains simple data transformation for inferential statistical analysis (conducted later in JASP)<br />

## Setup

```
python3 -m venv environment
source environment/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the API keys required by the models you use (e.g.
`OPENAI_API_KEY`, `GEMINI_API_KEY`), consumed via `litellm`/`python-dotenv`.

## Usage

```
python main_simulation.py <path_to_config.json> <output_subdirectory>
```

Example:

```
python main_simulation.py config/gemini_small.json simulation4
```

Config files (see `config/`) specify `n_runs`, `max_rounds`, and the
model/temperature for each of the three agents (`buyer`, `seller`, `judge`).

Results are written to `results/<output_subdirectory>/`:
- `results.csv` — one row per run (rounds completed, outcome, final price, token usage, cost)
- `judge_verdicts.csv` — per-round judge labels/justifications for each of the 8 requirements
- `price_trajectories.csv` — buyer/seller offers per round
- `classifications/crun_NNN.txt` — full transcript + judge verdicts for each run
- `transcription/run_NNN.txt` — conversation only, without judge verdicts

