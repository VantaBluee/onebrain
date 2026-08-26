#!/usr/bin/env python3
"""Manual M1 checklist: an UNMODIFIED OpenAI SDK pointed at OneBrain.

Usage:
    onebrain run tinystories-260k          # or any model
    pip install openai
    python scripts/check_openai_sdk.py [base_url] [model]

Defaults: base_url http://127.0.0.1:11435/v1, model taken from /v1/models.
This script exists for the milestone's Definition of Done ("an unmodified
OpenAI-SDK script ... streams tokens"); CI covers the same wire behavior in
Rust via `cargo xtask e2e`.
"""
import sys

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11435/v1"
client = OpenAI(base_url=base_url, api_key="unused-on-localhost")

model = sys.argv[2] if len(sys.argv) > 2 else client.models.list().data[0].id
print(f"model: {model}")

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Once upon a time"}],
    stream=True,
    max_tokens=32,
)

pieces = 0
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    pieces += 1 if delta else 0
    print(delta, end="", flush=True)
print()
assert pieces > 0, "expected at least one streamed content chunk"
print(f"OK: streamed {pieces} chunks through the OpenAI SDK")
