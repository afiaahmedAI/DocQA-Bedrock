import json
import os

import boto3

_region = os.environ.get("AWS_REGION", "us-east-1")
_embed_model_id = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
_chat_model_id = os.environ.get("CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

_client = boto3.client("bedrock-runtime", region_name=_region)


def embed_text(text: str) -> list[float]:
    response = _client.invoke_model(
        modelId=_embed_model_id,
        body=json.dumps({"inputText": text}),
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def generate_answer(prompt: str) -> str:
    response = _client.converse(
        modelId=_chat_model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]
