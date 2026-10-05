"""Typed tool registry with permission scopes and retry on transient failure."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ValidationError


class TransientError(Exception):
    """Worth retrying: timeouts, 429s, 5xx from a provider."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


class ToolFailed(Exception):
    """A tool gave up after its retries, or failed in a non-retryable way."""


class PermissionRequired(Exception):
    def __init__(self, tool: str, scope: str):
        super().__init__(f"tool '{tool}' needs scope '{scope}'")
        self.tool, self.scope = tool, scope


@dataclass
class CallInfo:
    attempt: int  # 0 on the first try, 1 on the first retry, ...


@dataclass
class Tool:
    name: str
    input_model: type[BaseModel]
    fn: Callable[[BaseModel, CallInfo], dict]
    scopes: tuple[str, ...] = ()
    description: str = ""


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    backoff_s: float = 0.0  # base for exponential backoff; 0 keeps tests and mocks fast
    max_wait_s: float = 60.0

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def describe(self) -> list[dict]:
        """Tool manifest in the shape an LLM tool-use API expects."""
        return [{"name": t.name, "description": t.description, "scopes": list(t.scopes),
                 "input_schema": t.input_model.model_json_schema()} for t in self.tools.values()]

    def call(self, name: str, args: dict, granted: set[str], retries: int = 0,
             on_retry: Callable[[int, str], None] | None = None) -> dict:
        if name not in self.tools:
            raise ToolFailed(f"unknown tool '{name}'")
        tool = self.tools[name]
        for scope in tool.scopes:
            if scope not in granted:
                raise PermissionRequired(name, scope)
        try:
            inp = tool.input_model(**args)
        except ValidationError as e:
            raise ToolFailed(f"invalid arguments for '{name}': {e.errors()[0]['msg']}") from e
        for attempt in range(retries + 1):
            try:
                return tool.fn(inp, CallInfo(attempt))
            except TransientError as e:
                if attempt == retries:
                    raise ToolFailed(f"{e} (gave up after {attempt + 1} tries)") from e
                if on_retry:
                    on_retry(attempt, str(e))
                # wait at least as long as the provider asked; hammering a rate limit only extends it
                time.sleep(min(self.max_wait_s, max(self.backoff_s * 2 ** attempt, e.retry_after or 0)))
        raise AssertionError("unreachable")
