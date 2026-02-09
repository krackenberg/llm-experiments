import yaml
from pathlib import Path
from .bedrock_client import BedrockClient, BedrockConfig
from .tracing import trace_call


def load_models_config():
    cfg_path = Path(__file__).parents[1] / "config" / "models.yml"
    return yaml.safe_load(cfg_path.read_text())


def make_client(model_name: str):
    cfg = load_models_config()
    m = cfg["models"][model_name]
    if m["provider"] != "bedrock":
        raise NotImplementedError("Only Bedrock shown here")
    return BedrockClient(BedrockConfig(model_id=m["model_id"], region=m["region"]))


def load_prompt(version="prompt_v001.txt"):
    p = Path(__file__).parents[1] / "prompts" / version
    return p.read_text()


@trace_call
def run_rf_qaqc(model_name: str, input_data: str, prompt_version="prompt_v001.txt"):
    client = make_client(model_name)
    prompt_template = load_prompt(prompt_version)
    full_prompt = prompt_template.replace("{input_data}", input_data)

    messages = [
        {"role": "user", "content": full_prompt}
    ]

    raw = client.chat(messages)
    return raw  # caller validates/loads JSON
