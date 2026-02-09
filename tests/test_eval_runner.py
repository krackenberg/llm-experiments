from src.eval.metrics import compute_qaqc_metrics


def test_metrics_basic():
    records = [
        {"json_valid": True, "status_correct": True},
        {"json_valid": False, "status_correct": False},
    ]
    m = compute_qaqc_metrics(records)
    assert m.total == 2
    assert m.json_valid_rate == 0.5
    assert m.status_accuracy == 0.5
