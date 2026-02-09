# LLM Experiment Framework

This repo is a template for repeatable LLM experiments and productionization.

## Quickstart

1. Define your task:
   - Edit `src/prompts/task_spec.md`
   - Create `src/data/gold_set.jsonl`

2. Wire a prompt:
   - Edit `src/prompts/prompt_v001.txt` (keep `{input_data}` placeholder)

3. Configure models:
   - Edit `src/config/models.yml`
   - Pin production models in `src/config/model_pins.yml`

4. Run eval:
   ```bash
   python -m src.eval.eval_runner --model claude-3.7-sonic --prompt prompt_v001.txt
   ```

5. Inspect:
   - Metrics in `eval_results_*.json`
   - Traces in `traces/`

6. Deploy:
   - Use `src/services/api_service.py` for an HTTP API
   - Or call `rf_qaqc_task` inside your pipeline

## Conventions

- Prompts and configs are versioned in Git.
- Every change to prompts requires:
  - `gold_set.jsonl` run
  - Metrics comparison
  - Entry in `prompts/prompt_changelog.md`
- Canary set is run on a schedule to detect model/provider changes.

## Framework Overview

This framework follows a 9-step playbook for LLM experimentation:

1. **Define Task & Eval** — Task spec, eval spec, and gold set before any coding.
2. **Start With Strongest Model** — Run the same prompt across a model zoo, pick the best, then optimize cost.
3. **Structure Prompt & Context** — Treat the LLM as a constraint satisfaction engine. Modularize prompts.
4. **Tracing & Prompt Versioning** — Log every request; version prompts like code.
5. **Automate Evaluation Loops** — One-command eval with hard checks and LLM-as-judge.
6. **Model & Provider Volatility** — Pin model IDs, run canary checks, treat upgrades like library upgrades.
7. **Multiple Perspectives** — Run different analyst roles on the same data, then synthesize.
8. **Fine-Tune Decision** — Exhaust prompt/context/RAG improvements before fine-tuning.
9. **Package for Production** — REST API, pipeline integration, monitoring.

## Repo Layout

```
llm-experiments/
  README.md
  pyproject.toml
  src/
    __init__.py
    config/
      models.yml
      model_pins.yml
      perspectives.yml
      prompt_graph.yml
    prompts/
      task_spec.md
      prompt_v001.txt
      judge_prompt.txt
      synthesis_prompt.txt
      prompt_changelog.md
    data/
      gold_set.jsonl
      canary_set.jsonl
      kb/                # PDFs, docs, etc. for RAG/eval
    llm/
      bedrock_client.py
      run_inference.py
      tracing.py
    eval/
      eval_spec.md
      eval_runner.py
      metrics.py
      llm_judge.py
    services/
      api_service.py
      pipeline_task.py
  tests/
    test_eval_runner.py
    test_prompts.py
  infra/
    airflow_dag.py
    docker/
      Dockerfile
      docker-compose.yml
    ci/
      github_actions.yml
```
