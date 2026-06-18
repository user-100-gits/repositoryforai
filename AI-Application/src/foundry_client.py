import json
from typing import Any, Dict, List
import requests
from .config import settings


class FoundryClient:
    def __init__(self):
        if not settings.foundry_api_key:
            raise ValueError("FOUNDRY_API_KEY is required")
        if not settings.foundry_embedding_url or not settings.foundry_llm_url:
            raise ValueError("FOUNDRY_EMBEDDING_URL and FOUNDRY_LLM_URL are required")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": settings.foundry_embedding_model,
            "input": texts,
        }
        response = requests.post(settings.foundry_embedding_url, headers=settings.auth_headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "data" in data:
            return [item["embedding"] for item in data["data"]]
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        raise ValueError(f"Unexpected embedding response: {json.dumps(data)}")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        message_payload = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": settings.foundry_llm_model,
            "messages": message_payload,
            "temperature": 0.0,
        }
        response = requests.post(settings.foundry_llm_url, headers=settings.auth_headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            if "text" in choice:
                return choice["text"]
        raise ValueError(f"Unexpected generation response: {json.dumps(data)}")
