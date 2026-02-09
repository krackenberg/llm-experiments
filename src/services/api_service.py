from fastapi import FastAPI
from pydantic import BaseModel
import json

from ..llm.run_inference import run_rf_qaqc

app = FastAPI()


class QAQCRequest(BaseModel):
    input_data: str
    model_name: str = "claude-3.7-sonic"
    prompt_version: str = "prompt_v001.txt"


class QAQCResponse(BaseModel):
    status: str
    reasons: list[str]
    suggested_followup: list[str]
    raw: str


@app.post("/rf/qaqc", response_model=QAQCResponse)
def rf_qaqc(req: QAQCRequest):
    raw = run_rf_qaqc(req.model_name, req.input_data, req.prompt_version)
    data = json.loads(raw)
    return QAQCResponse(
        status=data["status"],
        reasons=data.get("reasons", []),
        suggested_followup=data.get("suggested_followup", []),
        raw=raw,
    )
