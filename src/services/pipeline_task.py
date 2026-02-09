import json
from ..llm.run_inference import run_rf_qaqc


def rf_qaqc_task(input_data: str, model_name="claude-3.7-sonic", prompt_version="prompt_v001.txt"):
    raw = run_rf_qaqc(model_name, input_data, prompt_version)
    parsed = json.loads(raw)
    # write to DB, S3, or pass downstream in Airflow XCom
    return parsed
