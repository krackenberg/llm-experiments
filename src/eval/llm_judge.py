import json
from pathlib import Path
from ..llm.run_inference import make_client, load_prompt

JUDGE_PROMPT_FILE = "judge_prompt.txt"


def judge_explanation(model_name: str, case_input: str, model_output: str) -> dict:
    client = make_client(model_name)
    prompt_template = load_prompt(JUDGE_PROMPT_FILE)
    full_prompt = prompt_template.format(
        case_input=case_input,
        model_output=model_output,
    )
    messages = [{"role": "user", "content": full_prompt}]
    raw = client.chat(messages)
    return json.loads(raw)  # expects JSON with scores
