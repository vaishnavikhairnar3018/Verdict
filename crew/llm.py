import time
from typing import TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .config import FALLBACK_MODEL, MODEL

T = TypeVar("T", bound=BaseModel)

_client = None


def get_client() -> genai.Client:
    """Create the Gemini client on first use, so importing this module needs no API key (tests)."""
    global _client
    if _client is None:
        # Reads GEMINI_API_KEY from the environment. Give up on any request after 60 s.
        _client = genai.Client(http_options=types.HttpOptions(timeout=60_000))
    return _client

usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "models": []}


def reset_usage():
    usage.update(calls=0, input_tokens=0, output_tokens=0, models=[])

RETRY_WAITS = [5, 15, 30]          # seconds to wait before each retry
RETRYABLE = {429, 500, 503, 504}   # rate limited / server overloaded: waiting can fix these
COOLDOWN = 10 * 60                 # skip a model that keeps failing for 10 minutes

broken_until: dict[str, float] = {}   # model -> time when we may try it again (circuit breaker)


def generate(prompt: str, config: types.GenerateContentConfig):
    """Call Gemini, retrying temporary errors, then switching to the fallback model.

    Returns (response, name of the model that actually answered).
    """
    for model in (MODEL, FALLBACK_MODEL):
        if time.time() < broken_until.get(model, 0):
            continue                        # circuit open: this model failed recently, skip it
        for wait in RETRY_WAITS + [None]:
            try:
                return get_client().models.generate_content(model=model, contents=prompt, config=config), model
            except (errors.APIError, httpx.TransportError) as e:
                # TransportError = timeout or dropped connection, also temporary.
                code = getattr(e, "code", "network")
                if isinstance(e, errors.APIError) and code not in RETRYABLE:
                    raise                   # a real bug (bad key, bad request): don't hide it
                if wait is None:
                    break                   # out of retries for this model
                print(f"    ({model} busy: {code}, retrying in {wait}s)")
                time.sleep(wait)
        broken_until[model] = time.time() + COOLDOWN
        print(f"    ({model} unavailable, skipping it for {COOLDOWN // 60} minutes)")
    broken_until.clear()   # give both models a fresh chance next time
    raise RuntimeError("Gemini is overloaded or out of free quota right now. "
                       "Please try again in a few minutes.")


def ask(system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
    """Send one prompt to Gemini and get back a validated `schema` object."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",            # reply must be JSON...
        response_json_schema=schema.model_json_schema(),  # ...in exactly this shape
        thinking_config=types.ThinkingConfig(thinking_level=effort),  # how hard it thinks
    )

    response, model = generate(prompt, config)

    meta = response.usage_metadata
    usage["calls"] += 1
    if model not in usage["models"]:
        usage["models"].append(model)   # so the dashboard shows which model really answered
    usage["input_tokens"] += (meta.prompt_token_count or 0) if meta else 0
    usage["output_tokens"] += (meta.candidates_token_count or 0) if meta else 0

    if not response.text:
        raise RuntimeError("Gemini returned no text (the reply may have been blocked).")
    return schema.model_validate_json(response.text)
