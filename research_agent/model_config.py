"""Shared Gemini model settings with quota-aware retries."""

from __future__ import annotations

import os

from google.genai import types

RETRY_ATTEMPTS = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "5"))
RETRY_INITIAL_DELAY = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "2.0"))

# Mitigates Vertex AI 429 RESOURCE_EXHAUSTED on burst multi-agent calls.
GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=RETRY_INITIAL_DELAY,
            attempts=RETRY_ATTEMPTS,
        ),
    ),
)
