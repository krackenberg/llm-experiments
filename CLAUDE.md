# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run a single test:
```bash
pytest tests/test_eval_runner.py::test_metrics_basic
```

Lint:
```bash
ruff check src tests
```

Run the full eval pipeline (uses mock client if `ANTHROPIC_API_KEY` is not set):
```bash
python examples/support_ticket_triage/run_example.py
python examples/support_ticket_triage/run_example.py --canary          # 3-case quick check
python examples/support_ticket_triage/run_example.py --cases t001 t005 # specific cases
python examples/support_ticket_triage/run_example.py --model haiku     # different model
python examples/support_ticket_triage/run_example.py --no-judge --no-perspectives
```

Run a lower-level eval via the framework module (requires AWS credentials for Bedrock models):
```bash
python -m src.eval.eval_runner --model sonnet --prompt triage_v001.txt
```

## Architecture

This is a **repeatable LLM experimentation framework** — a template for building, evaluating, and productionizing LLM-based tasks. The code follows a 9-step playbook: define task/eval → pick model → version prompts → automate evals → handle model volatility → multi-perspective analysis → fine-tune decision → production packaging.

### Key data flow

```
gold_set.jsonl / canary_set.jsonl
    → eval_runner (loads cases, calls run_inference)
        → run_inference (loads prompt template, substitutes {input_data}, calls LLM client)
            → AnthropicClient | BedrockClient | MockClient
        ← raw JSON string
    → metrics.py (hard eval: JSON validity, field accuracy)
    → llm_judge.py (LLM-as-judge: quality scoring via second model call)
    → multi-perspective analysis (same ticket through N role-based prompts)
    → synthesis (final model synthesizes all perspectives into unified assessment)
    → eval_results_*.json + traces/*.json
```

### LLM clients (`src/llm/`)

Three interchangeable clients sharing the same `.chat(messages) -> str` interface:
- **`AnthropicClient`** — calls Anthropic Messages API directly; requires `ANTHROPIC_API_KEY`
- **`BedrockClient`** — calls AWS Bedrock; requires AWS credentials and `region` in config
- **`MockClient`** — deterministic keyword-based responses for local/CI use; no API key needed

`run_inference.py` selects the client by reading `src/config/models.yml` and checking the `provider` field. `examples/support_ticket_triage/run_example.py` auto-detects `ANTHROPIC_API_KEY` and falls back to mock.

### Tracing

`src/llm/tracing.py` provides a `@trace_call` decorator. Any decorated function writes a JSON trace to `traces/<uuid>.json` including args, result (truncated to 2000 chars), duration, and error. Apply it to inference functions.

### Prompt versioning

Prompts live in `src/prompts/` as plain text files with `{input_data}` as the substitution placeholder. Every prompt change requires: running the gold set, comparing metrics, and adding an entry to `src/prompts/prompt_changelog.md`.

### Eval structure

- **Hard eval** — JSON validity + schema field checks + expected-value comparison against gold set labels
- **LLM-as-judge** — a second model call scores quality dimensions (specificity, correctness, completeness) on a 1–5 scale
- **Multi-perspective** — the same input is passed through N role-specific prompts (e.g., Customer Success Manager, Engineering Lead, Security Analyst), then a synthesis prompt produces a unified assessment

### Config files

- `src/config/models.yml` — model aliases → provider + model ID + region
- `src/config/model_pins.yml` — pinned production model IDs
- `src/config/perspectives.yml` — role definitions for multi-perspective analysis
- `src/config/prompt_graph.yml` — multi-step prompt chains (summarize → classify)

### Working example

`examples/support_ticket_triage/` is the canonical working demo. It is self-contained: it imports from `src/` but manages its own gold set, canary set, and eval results. This is the reference for how to wire a new task into the framework.
