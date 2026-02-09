# Working Example: Support Ticket Triage

This is a fully functional demo of the LLM Experiment Framework applied to a
**support ticket triage** task — classifying incoming tickets by category,
priority, and routing team.

## What It Demonstrates

| Framework Section | What runs |
|---|---|
| §1 Task & Eval | `task_spec.md`, `gold_set.jsonl` (20 cases), `canary_set.jsonl` (3 cases) |
| §2 Model Sweep | Configurable via `--model` flag against `models.yml` |
| §3 Prompt Structure | `triage_v001.txt` — structured prompt with clear schema |
| §4 Tracing | Every LLM call logged to `traces/` as JSON |
| §5 Automated Eval | Hard checks (JSON, schema, accuracy) + LLM-as-judge scoring |
| §6 Canary Checks | `--canary` mode runs must-pass/must-fail sanity set |
| §7 Multi-Perspective | 3 analyst roles (CS, Eng, Security) + synthesis on one case |
| §8 Fine-Tune Decision | N/A (prompt-only in this example) |
| §9 Production Packaging | See `src/services/api_service.py` in the main repo |

## Quick Start

```bash
# No API key needed — runs with mock client for demo
cd examples/support_ticket_triage
python run_example.py

# With real Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
python run_example.py

# Quick canary check
python run_example.py --canary

# Specific cases only
python run_example.py --cases t001 t005 t013

# Skip expensive phases
python run_example.py --no-judge --no-perspectives

# Try a different model
python run_example.py --model haiku
```

## Output

- **Console**: Per-case results, judge scores, perspective analysis, summary metrics
- **`eval_results_<model>_<id>.json`**: Full machine-readable results
- **`../../traces/`**: Individual trace files for every LLM call

## Gold Set Design

20 tickets following the 70/30 split from the framework:

- **14 normal cases**: clear billing, bug, feature request, access, inquiry tickets
- **6 edge/corner cases**: ambiguous priority (t012), security overlap (t013),
  cosmetic "bug" (t016), enterprise onboarding (t017), pre-sales (t015),
  payment edge case (t019)

## Extending This Example

1. **New prompt version**: Copy `triage_v001.txt` → `triage_v002.txt`, edit,
   run `python run_example.py --prompt triage_v002.txt`, compare metrics.
2. **More test cases**: Append to `gold_set.jsonl` (keep the format).
3. **Different task entirely**: Use this as a template — swap the prompt,
   gold set, and schema validation logic.
