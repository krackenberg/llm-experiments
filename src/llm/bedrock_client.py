import boto3
import json
from dataclasses import dataclass


@dataclass
class BedrockConfig:
    model_id: str
    region: str


class BedrockClient:
    def __init__(self, cfg: BedrockConfig):
        self.cfg = cfg
        self.client = boto3.client("bedrock-runtime", region_name=cfg.region)

    def chat(self, messages, max_tokens=1024, temperature=0.2):
        body = {
            "modelId": self.cfg.model_id,
            "input": {
                "messages": messages
            },
            "generationConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        }
        resp = self.client.invoke_model(
            modelId=self.cfg.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        out = json.loads(resp["body"].read())
        # Adapt to Anthropic-in-Bedrock format as needed
        return out["output"]["message"]["content"][0]["text"]
