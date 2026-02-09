#!/usr/bin/env python3
"""
run_example.py — End-to-end working demo of the LLM Experiment Framework.

Exercises the full loop:
  1. Load config + gold set
  2. Run inference for every test case
  3. Hard evals  (JSON validity, schema checks, field-level accuracy)
  4. LLM-as-judge eval  (quality scoring via a second model call)
  5. Multi-perspective analysis on a single ticket
  6. Synthesis of perspectives
  7. Tracing  (every call logged to traces/)
  8. Summary report with metrics + failure analysis

If ANTHROPIC_API_KEY is set → uses the real Anthropic API.
Otherwise → uses a deterministic mock client so the full pipeline runs locally.

Usage:
    python run_example.py                       # full gold set (mock or real)
    python run_example.py --canary              # canary-only quick check
    python run_example.py --cases t001 t005     # specific case IDs
    python run_example.py --model haiku         # different model from models.yml
    python run_example.py --no-judge            # skip LLM-as-judge phase
    python run_example.py --no-perspectives     # skip multi-perspective phase

    # Force real API even if mock would be used:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_example.py
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, field, asdict

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
EXAMPLE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))

from src.llm.anthropic_client import AnthropicClient, AnthropicConfig
from src.llm.mock_client import MockClient, MockConfig
from src.llm.tracing import trace_call, TRACE_DIR

# ── Constants ────────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"billing", "bug", "feature_request", "account_access", "general_inquiry"}
VALID_PRIORITIES = {"P1_critical", "P2_high", "P3_medium", "P4_low"}
VALID_TEAMS = {"billing_ops", "engineering", "product", "security", "support_general"}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    json_valid: bool = False
    schema_valid: bool = False
    category_correct: bool = False
    priority_correct: bool = False
    team_correct: bool = False
    parsed_output: dict = field(default_factory=dict)
    raw_output: str = ""
    error: str | None = None
    latency_sec: float = 0.0
    judge_scores: dict = field(default_factory=dict)


@dataclass
class EvalSummary:
    model: str
    prompt_version: str
    timestamp: str
    total_cases: int
    json_valid_rate: float
    schema_valid_rate: float
    category_accuracy: float
    priority_accuracy: float
    team_accuracy: float
    all_correct_rate: float
    avg_latency_sec: float
    judge_avg_specificity: float | None = None
    judge_avg_correctness: float | None = None
    judge_avg_completeness: float | None = None
    using_mock: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_prompt(name: str) -> str:
    return (SRC / "prompts" / name).read_text()


def make_client(model_name: str) -> tuple:
    """Return (client, using_mock)."""
    import yaml
    cfg = yaml.safe_load((SRC / "config" / "models.yml").read_text())
    m = cfg["models"][model_name]

    if m["provider"] == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient(AnthropicConfig(model_id=m["model_id"])), False
    else:
        if m["provider"] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            pass  # will print notice in main
        return MockClient(MockConfig(model_id=f"mock-{model_name}")), True


def validate_schema(parsed: dict) -> bool:
    """Hard check: required fields present with valid enum values."""
    return (
        isinstance(parsed, dict)
        and parsed.get("category") in VALID_CATEGORIES
        and parsed.get("priority") in VALID_PRIORITIES
        and parsed.get("suggested_team") in VALID_TEAMS
        and isinstance(parsed.get("reasoning"), str)
        and len(parsed["reasoning"]) > 0
    )


def clean_json_output(raw: str) -> str:
    """Strip markdown fences the model sometimes wraps around JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


# ── Core inference ───────────────────────────────────────────────────────────

@trace_call
def run_triage(client, input_data: str, prompt_template: str) -> str:
    full_prompt = prompt_template.replace("{input_data}", input_data)
    return client.chat([{"role": "user", "content": full_prompt}])


def eval_single_case(client, case: dict, prompt_template: str) -> CaseResult:
    result = CaseResult(case_id=case["id"])
    start = time.time()

    try:
        raw = run_triage(client, case["input_data"], prompt_template)
        result.raw_output = raw
        result.latency_sec = time.time() - start

        parsed = json.loads(clean_json_output(raw))
        result.json_valid = True
        result.parsed_output = parsed

        result.schema_valid = validate_schema(parsed)
        result.category_correct = parsed.get("category") == case["expected_category"]
        result.priority_correct = parsed.get("priority") == case["expected_priority"]
        result.team_correct = parsed.get("suggested_team") == case["expected_team"]

    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
        result.latency_sec = time.time() - start
    except Exception as e:
        result.error = str(e)
        result.latency_sec = time.time() - start

    return result


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator of support ticket triage quality.

ORIGINAL TICKET:
{ticket}

MODEL'S TRIAGE OUTPUT:
{output}

Rate the triage on these dimensions (1 = terrible, 5 = excellent):
1. specificity: Does the reasoning reference concrete details from the ticket?
2. correctness: Are the category, priority, and team assignments logical and defensible?
3. completeness: Does the reasoning cover the key factors (severity, customer impact, tier)?

Return ONLY valid JSON — no markdown, no commentary:
{{"specificity": <int 1-5>, "correctness": <int 1-5>, "completeness": <int 1-5>}}"""


@trace_call
def judge_triage(judge_client, ticket: str, model_output: str) -> dict:
    prompt = JUDGE_PROMPT.format(ticket=ticket, output=model_output)
    raw = judge_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return json.loads(clean_json_output(raw))


# ── Multi-perspective analysis (Framework Section 7) ─────────────────────────

PERSPECTIVES = [
    {
        "id": "customer_success",
        "role": "Customer Success Manager",
        "prompt": (
            "As a Customer Success Manager, analyze this support ticket. "
            "Focus on: customer sentiment, churn risk, relationship impact, and whether "
            "this needs proactive outreach beyond just fixing the issue.\n\n"
            "TICKET:\n{input_data}\n\n"
            'Return ONLY JSON:\n'
            '{"sentiment": "positive"|"neutral"|"frustrated"|"angry", '
            '"churn_risk": "low"|"medium"|"high", "needs_outreach": true|false, '
            '"cs_notes": "string"}'
        ),
    },
    {
        "id": "engineering_lead",
        "role": "Engineering Lead",
        "prompt": (
            "As an Engineering Lead, analyze this support ticket for technical impact. "
            "Focus on: likely root cause category, blast radius, whether this could be a "
            "regression, and estimated complexity to fix.\n\n"
            "TICKET:\n{input_data}\n\n"
            'Return ONLY JSON:\n'
            '{"likely_cause": "string", "blast_radius": "single_user"|"subset"|"all_users", '
            '"possible_regression": true|false, "fix_complexity": "trivial"|"moderate"|"complex", '
            '"eng_notes": "string"}'
        ),
    },
    {
        "id": "security_analyst",
        "role": "Security Analyst",
        "prompt": (
            "As a Security Analyst, analyze this support ticket. "
            "Focus on: whether there are any security implications, data exposure risk, "
            "potential for account compromise, and whether security team should be looped in.\n\n"
            "TICKET:\n{input_data}\n\n"
            'Return ONLY JSON:\n'
            '{"security_relevant": true|false, "data_exposure_risk": "none"|"low"|"medium"|"high", '
            '"account_compromise_risk": "none"|"low"|"medium"|"high", "security_notes": "string"}'
        ),
    },
]

SYNTHESIS_PROMPT = (
    "You are a senior support operations manager synthesizing multiple analyst "
    "perspectives on a support ticket.\n\n"
    "ORIGINAL TICKET:\n{ticket}\n\n"
    "ANALYST PERSPECTIVES:\n{perspectives}\n\n"
    "Synthesize these into a unified assessment. Identify where analysts agree, "
    "where they disagree, and what the recommended action plan should be.\n\n"
    'Return ONLY JSON:\n'
    '{{"common_findings": ["string"], "disagreements": ["string"], '
    '"unified_priority": "P1_critical"|"P2_high"|"P3_medium"|"P4_low", '
    '"action_plan": ["string"], "escalation_needed": true|false}}'
)


@trace_call
def run_perspective(client, perspective: dict, input_data: str) -> dict:
    prompt = perspective["prompt"].replace("{input_data}", input_data)
    raw = client.chat([{"role": "user", "content": prompt}])
    return {"perspective_id": perspective["id"], "analysis": json.loads(clean_json_output(raw))}


@trace_call
def run_synthesis(client, ticket: str, perspective_results: list[dict]) -> dict:
    perspectives_str = json.dumps(perspective_results, indent=2)
    prompt = SYNTHESIS_PROMPT.format(ticket=ticket, perspectives=perspectives_str)
    raw = client.chat([{"role": "user", "content": prompt}])
    return json.loads(clean_json_output(raw))


# ── Main runner ──────────────────────────────────────────────────────────────

def run_eval(
    model_name: str,
    prompt_version: str,
    gold_path: Path,
    case_ids: list[str] | None = None,
    run_judge: bool = True,
    run_perspectives: bool = True,
) -> dict:
    """Run the full eval loop and return results dict."""

    client, using_mock = make_client(model_name)
    prompt_template = load_prompt(prompt_version)
    gold = load_jsonl(gold_path)

    if case_ids:
        gold = [c for c in gold if c["id"] in case_ids]

    mode_label = "MOCK (no API key)" if using_mock else "LIVE API"

    print(f"\n{'='*72}")
    print(f"  LLM Experiment Framework — Working Example")
    print(f"  Model: {model_name}  |  Prompt: {prompt_version}  |  Mode: {mode_label}")
    print(f"  Gold set: {gold_path.name} ({len(gold)} cases)  |  Judge: {'yes' if run_judge else 'no'}")
    print(f"{'='*72}")

    if using_mock:
        print(f"\n  ℹ  No ANTHROPIC_API_KEY found. Running with deterministic mock client.")
        print(f"     Set the key to use real Anthropic API calls.\n")

    # ── Phase 1: Inference + hard eval ───────────────────────────────────
    print("Phase 1: Inference + Hard Eval")
    print("-" * 50)
    results: list[CaseResult] = []
    for i, case in enumerate(gold):
        r = eval_single_case(client, case, prompt_template)
        results.append(r)

        all_ok = r.category_correct and r.priority_correct and r.team_correct
        mark = "✓" if all_ok else "✗"
        cat_m = "✓" if r.category_correct else "✗"
        pri_m = "✓" if r.priority_correct else "✗"
        tm_m = "✓" if r.team_correct else "✗"

        print(
            f"  [{i+1:2d}/{len(gold)}] {r.case_id:15s}  "
            f"cat:{cat_m} pri:{pri_m} team:{tm_m}  "
            f"({r.latency_sec:.1f}s)  {mark}"
        )
        if r.error:
            print(f"          ERROR: {r.error}")

    # ── Phase 2: LLM-as-judge ────────────────────────────────────────────
    if run_judge:
        print(f"\nPhase 2: LLM-as-Judge")
        print("-" * 50)
        for i, (case, r) in enumerate(zip(gold, results)):
            if not r.json_valid:
                print(f"  [{i+1:2d}/{len(gold)}] {r.case_id:15s}  SKIP (invalid JSON)")
                continue
            try:
                scores = judge_triage(client, case["input_data"], r.raw_output)
                r.judge_scores = scores
                print(
                    f"  [{i+1:2d}/{len(gold)}] {r.case_id:15s}  "
                    f"spec={scores.get('specificity','?')}  "
                    f"corr={scores.get('correctness','?')}  "
                    f"comp={scores.get('completeness','?')}"
                )
            except Exception as e:
                print(f"  [{i+1:2d}/{len(gold)}] {r.case_id:15s}  JUDGE ERROR: {e}")

    # ── Phase 3: Multi-perspective ───────────────────────────────────────
    perspective_demo = None
    if run_perspectives and gold:
        demo_case = next(
            (c for c in gold if c.get("expected_priority") == "P1_critical"),
            gold[0],
        )
        print(f"\nPhase 3: Multi-Perspective Analysis (case {demo_case['id']})")
        print("-" * 50)

        persp_results = []
        for p in PERSPECTIVES:
            try:
                analysis = run_perspective(client, p, demo_case["input_data"])
                persp_results.append(analysis)
                snippet = json.dumps(analysis["analysis"], indent=None)[:80]
                print(f"  ✓ {p['id']:20s} → {snippet}...")
            except Exception as e:
                print(f"  ✗ {p['id']:20s} → ERROR: {e}")

        if persp_results:
            print(f"\n  Synthesizing {len(persp_results)} perspectives...")
            try:
                synthesis = run_synthesis(client, demo_case["input_data"], persp_results)
                perspective_demo = {
                    "case_id": demo_case["id"],
                    "perspectives": persp_results,
                    "synthesis": synthesis,
                }
                print(f"  ✓ Unified priority:   {synthesis.get('unified_priority', '?')}")
                print(f"  ✓ Escalation needed:  {synthesis.get('escalation_needed', '?')}")
                if synthesis.get("action_plan"):
                    print(f"  ✓ Action plan:")
                    for step in synthesis["action_plan"][:4]:
                        print(f"      → {step}")
            except Exception as e:
                print(f"  ✗ Synthesis error: {e}")

    # ── Compute metrics ──────────────────────────────────────────────────
    n = len(results)

    def safe_rate(pred): return sum(pred(r) for r in results) / n if n else 0

    summary = EvalSummary(
        model=model_name,
        prompt_version=prompt_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_cases=n,
        json_valid_rate=safe_rate(lambda r: r.json_valid),
        schema_valid_rate=safe_rate(lambda r: r.schema_valid),
        category_accuracy=safe_rate(lambda r: r.category_correct),
        priority_accuracy=safe_rate(lambda r: r.priority_correct),
        team_accuracy=safe_rate(lambda r: r.team_correct),
        all_correct_rate=safe_rate(
            lambda r: r.category_correct and r.priority_correct and r.team_correct
        ),
        avg_latency_sec=sum(r.latency_sec for r in results) / n if n else 0,
        using_mock=using_mock,
    )

    judged = [r for r in results if r.judge_scores]
    if judged:
        summary.judge_avg_specificity = sum(r.judge_scores.get("specificity", 0) for r in judged) / len(judged)
        summary.judge_avg_correctness = sum(r.judge_scores.get("correctness", 0) for r in judged) / len(judged)
        summary.judge_avg_completeness = sum(r.judge_scores.get("completeness", 0) for r in judged) / len(judged)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY {'(MOCK)' if using_mock else '(LIVE)'}")
    print(f"{'='*72}")
    print(f"  Model:              {summary.model}")
    print(f"  Prompt:             {summary.prompt_version}")
    print(f"  Total cases:        {summary.total_cases}")
    print(f"  ─────────────────────────────────")
    print(f"  JSON valid rate:    {summary.json_valid_rate:6.0%}")
    print(f"  Schema valid rate:  {summary.schema_valid_rate:6.0%}")
    print(f"  ─────────────────────────────────")
    print(f"  Category accuracy:  {summary.category_accuracy:6.0%}")
    print(f"  Priority accuracy:  {summary.priority_accuracy:6.0%}")
    print(f"  Team accuracy:      {summary.team_accuracy:6.0%}")
    print(f"  All-correct rate:   {summary.all_correct_rate:6.0%}")
    print(f"  ─────────────────────────────────")
    print(f"  Avg latency:        {summary.avg_latency_sec:.2f}s")
    if summary.judge_avg_specificity is not None:
        print(f"  ─────────────────────────────────")
        print(f"  Judge specificity:  {summary.judge_avg_specificity:.1f} / 5")
        print(f"  Judge correctness:  {summary.judge_avg_correctness:.1f} / 5")
        print(f"  Judge completeness: {summary.judge_avg_completeness:.1f} / 5")
    print(f"{'='*72}")

    # ── Failure analysis ─────────────────────────────────────────────────
    failures = [
        r for r in results
        if not (r.category_correct and r.priority_correct and r.team_correct)
    ]
    if failures:
        print(f"\n  FAILURE ANALYSIS ({len(failures)} cases)")
        print(f"  {'─'*50}")
        for r in failures:
            expected = next(c for c in gold if c["id"] == r.case_id)
            print(f"\n  {r.case_id}:")
            if r.error:
                print(f"    Error: {r.error}")
            else:
                diffs = []
                if not r.category_correct:
                    diffs.append(f"category: {r.parsed_output.get('category')} (expected {expected['expected_category']})")
                if not r.priority_correct:
                    diffs.append(f"priority: {r.parsed_output.get('priority')} (expected {expected['expected_priority']})")
                if not r.team_correct:
                    diffs.append(f"team: {r.parsed_output.get('suggested_team')} (expected {expected['expected_team']})")
                for d in diffs:
                    print(f"    ✗ {d}")
                if r.parsed_output.get("reasoning"):
                    print(f"    reasoning: \"{r.parsed_output['reasoning'][:120]}\"")
    else:
        print(f"\n  ✓ All {n} cases passed!")

    # ── Write results to file ────────────────────────────────────────────
    run_id = str(uuid4())[:8]
    out = {
        "run_id": run_id,
        "summary": asdict(summary),
        "case_results": [asdict(r) for r in results],
    }
    if perspective_demo:
        out["perspective_demo"] = perspective_demo

    out_path = EXAMPLE_DIR / f"eval_results_{model_name}_{run_id}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    trace_count = len(list(TRACE_DIR.glob("*.json")))
    print(f"\n  Output files:")
    print(f"    Results: {out_path.name}")
    print(f"    Traces:  {TRACE_DIR}/ ({trace_count} files)")
    print()

    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Support Ticket Triage — LLM Experiment Framework Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_example.py                    # full gold set
  python run_example.py --canary           # quick canary check (3 cases)
  python run_example.py --cases t001 t005  # specific cases only
  python run_example.py --model haiku      # try a different model
  python run_example.py --no-judge --no-perspectives  # inference + hard eval only
        """,
    )
    parser.add_argument("--model", default="sonnet", help="Model name from models.yml (default: sonnet)")
    parser.add_argument("--prompt", default="triage_v001.txt", help="Prompt file name")
    parser.add_argument("--canary", action="store_true", help="Run canary set only (3 cases)")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to run")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge phase")
    parser.add_argument("--no-perspectives", action="store_true", help="Skip multi-perspective phase")

    args = parser.parse_args()

    gold_path = EXAMPLE_DIR / ("canary_set.jsonl" if args.canary else "gold_set.jsonl")

    run_eval(
        model_name=args.model,
        prompt_version=args.prompt,
        gold_path=gold_path,
        case_ids=args.cases,
        run_judge=not args.no_judge,
        run_perspectives=not args.no_perspectives,
    )


if __name__ == "__main__":
    main()
