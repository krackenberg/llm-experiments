import json
from pathlib import Path
import argparse
from typing import Dict, Any

from ..llm.run_inference import run_rf_qaqc
from .metrics import compute_qaqc_metrics

ROOT = Path(__file__).parents[2]


def load_gold_set() -> list[Dict[str, Any]]:
    path = ROOT / "src" / "data" / "gold_set.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def eval_model(model_name: str, prompt_version: str):
    gold = load_gold_set()
    records = []
    for case in gold:
        input_data = case["input_data"]
        expected_status = case["expected_status"]
        raw = run_rf_qaqc(model_name, input_data, prompt_version)
        try:
            parsed = json.loads(raw)
            json_valid = True
        except Exception:
            parsed = {}
            json_valid = False

        status = parsed.get("status")
        status_correct = json_valid and (status == expected_status)

        records.append({
            "id": case["id"],
            "json_valid": json_valid,
            "status_correct": status_correct,
            "raw_output": raw[:1000],
        })

    metrics = compute_qaqc_metrics(records)
    out_path = ROOT / f"eval_results_{model_name}_{prompt_version}.json"
    out_path.write_text(json.dumps({
        "metrics": metrics.__dict__,
        "records": records,
    }, indent=2))
    print(f"Metrics: {metrics}")
    print(f"Details written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="prompt_v001.txt")
    args = parser.parse_args()
    eval_model(args.model, args.prompt)
