"""Minimal Anthropic Messages client: one forced tool call in, one validated dict out."""
from __future__ import annotations

import base64
import os

import httpx

from .registry import ToolFailed, TransientError

API_URL = "https://api.anthropic.com/v1/messages"


class Claude:
    def __init__(self, model: str | None = None, api_key: str | None = None, transport=None):
        self.model = model or os.environ.get("CUTROOM_LLM_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.transport = transport  # tests inject httpx.MockTransport here

    def tool_call(self, system: str, content, name: str, schema: dict, max_tokens: int = 1500) -> dict:
        """Ask for exactly one call of tool `name` and return its input."""
        if not self.api_key:
            raise ToolFailed("ANTHROPIC_API_KEY is not set")
        body = {
            "model": self.model, "max_tokens": max_tokens, "system": system,
            "tools": [{"name": name, "description": f"Submit the {name} result.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": name},
            "messages": [{"role": "user", "content": content}],
        }
        try:
            with httpx.Client(timeout=90, transport=self.transport) as http:
                r = http.post(API_URL, json=body, headers={
                    "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        except httpx.TransportError as e:
            raise TransientError(f"llm: {e}") from e
        if r.status_code in (408, 429) or r.status_code >= 500:
            raise TransientError(f"llm: HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ToolFailed(f"llm: HTTP {r.status_code}: {r.text[:200]}")
        for block in r.json().get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == name:
                return block["input"]
        raise ToolFailed("llm: response had no tool call")


def image_block(jpeg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(jpeg).decode()}}
