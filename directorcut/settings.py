"""Environment settings.

Settings are read as DIRECTORCUT_<NAME>. The project was previously called Cutroom, so
CUTROOM_<NAME> is still accepted: a deployment configured under the old names keeps
working, and can be moved to the new names one setting at a time. An empty value
counts as unset.
"""
from __future__ import annotations

import logging
import os

PREFIX, LEGACY_PREFIX = "DIRECTORCUT_", "CUTROOM_"
log = logging.getLogger("directorcut")


def env(name: str, default: str = "") -> str:
    return os.environ.get(PREFIX + name) or os.environ.get(LEGACY_PREFIX + name) or default


def env_float(name: str, default: float) -> float:
    raw = env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        log.warning("%s%s=%r is not a number; using %s", PREFIX, name, raw, default)
        return default


def parse_model(item: str, default_price: float) -> tuple[str, float] | None:
    """'owner/name=price' -> (model, price). A missing or unreadable price falls back to
    the default instead of stopping the service: the price only feeds cost tracking."""
    model, _, price = item.strip().partition("=")
    if not model:
        return None
    try:
        return model, float(price) if price.strip() else default_price
    except ValueError:
        log.warning("price %r for model %s is not a number; using %s", price, model, default_price)
        return model, default_price
