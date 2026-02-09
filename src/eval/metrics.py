from dataclasses import dataclass
from typing import List


@dataclass
class QAEvalResult:
    total: int
    status_accuracy: float
    json_valid_rate: float


def compute_qaqc_metrics(records: List[dict]) -> QAEvalResult:
    total = len(records)
    json_valid = sum(r["json_valid"] for r in records)
    status_correct = sum(r["status_correct"] for r in records)
    return QAEvalResult(
        total=total,
        status_accuracy=status_correct / total if total else 0.0,
        json_valid_rate=json_valid / total if total else 0.0,
    )
