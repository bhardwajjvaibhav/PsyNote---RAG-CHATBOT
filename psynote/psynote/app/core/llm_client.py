"""
core/llm_client.py

Isolates the generation call (OpenRouter, Section 6: gpt-4o-mini or
claude-3.5-sonnet) behind one function: retries, timeouts, and
model-fallback all live here so rag_pipeline.py and the eventual
chat-history/prompt-building code (Phase 7+ in Section 8) just call
generate(messages) and get back text or a clear final failure -- they
never need to know an HTTP call happened at all.

Design note -- injectable transport: same dependency-injection shape as
reranker.py's `scorer` param. The real transport makes an HTTP call to
OpenRouter; tests inject a fake transport so retry/backoff/fallback
logic can be verified deterministically and fast, without real network
access or a live API key. Production usage just doesn't pass `transport`
and gets the real HTTP path.

Design note -- .env loading: OPENROUTER_API_KEY is read from the
process environment (see _default_transport below), and this module
loads a project-root .env file into that environment on import, using
python-dotenv, resolved by THIS FILE's own location rather than the
current working directory -- so `python -m core.rag_pipeline` from
app/, `uvicorn api.routes:app` from app/, and a one-off `python -c`
from anywhere all find the same .env file (project_root/.env, one
level above app/) without needing python-dotenv's default cwd-search
behavior to happen to guess right. Safe to have python-dotenv missing
entirely (e.g. a deployment that sets real env vars directly) -- import
failure there is swallowed, not fatal.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
    # llm_client.py -> core/ -> app/ -> project root -> .env
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # python-dotenv not installed -- fine if env vars are set some other way

import os
import time

DEFAULT_MODELS = ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]  # Section 6 fallback order, OpenRouter-prefixed
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    """Base class for all llm_client errors."""


class LLMTransientError(LLMError):
    """Retryable: timeout, 5xx, or 429 rate limit."""


class LLMClientError(LLMError):
    """Not retryable: 4xx other than 429 -- the request itself is broken."""


class LLMAllProvidersExhaustedError(LLMError):
    """Every model in the fallback chain failed after its retry budget."""

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        summary = "; ".join(f"{a['model']}: {a['error']}" for a in attempts)
        super().__init__(f"All providers exhausted. Attempts: {summary}")


def _default_transport(model: str, messages: list[dict], timeout: float) -> str:
    """
    Real HTTP call to OpenRouter. Requires OPENROUTER_API_KEY in the
    environment. Raises LLMTransientError / LLMClientError per the
    classification above so generate()'s retry logic can branch on it.
    """
    import requests  # imported lazily so this module has no hard
                      # dependency on `requests` for callers that only
                      # ever use an injected transport (e.g. tests)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMClientError("OPENROUTER_API_KEY is not set")

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages},
            timeout=timeout,
        )
    except requests.exceptions.Timeout as e:
        raise LLMTransientError(f"timeout after {timeout}s") from e
    except requests.exceptions.ConnectionError as e:
        raise LLMTransientError(f"connection error: {e}") from e

    if response.status_code == 429 or response.status_code >= 500:
        raise LLMTransientError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise LLMClientError(f"HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate(
    messages: list[dict],
    models: list[str] | None = None,
    max_retries_per_model: int = 3,
    timeout_seconds: float = 30.0,
    backoff_base_seconds: float = 1.0,
    transport=None,
) -> dict:
    """
    Generate a completion, retrying transient failures and falling back
    across `models` in order (default: Section 6's gpt-4o-mini then
    claude-3.5-sonnet) when a model's retry budget is exhausted or it
    hits a non-retryable client error.

    Returns {"text": ..., "model": which model actually succeeded,
    "attempts": full attempt log across all models/retries} on success.

    Raises LLMAllProvidersExhaustedError if every model in the chain
    fails. The exception carries the full attempt log for the caller to
    surface or log to security/audit_log.py (Phase 9).
    """
    models = models or DEFAULT_MODELS
    transport_fn = transport or _default_transport
    attempts: list[dict] = []

    for model in models:
        for retry_num in range(max_retries_per_model):
            try:
                text = transport_fn(model, messages, timeout_seconds)
                return {"text": text, "model": model, "attempts": attempts}
            except LLMClientError as e:
                # Not retryable -- this request is broken for this model.
                # Log it once and move straight to the next model.
                attempts.append({"model": model, "retry": retry_num, "error": str(e), "retryable": False})
                break
            except LLMTransientError as e:
                attempts.append({"model": model, "retry": retry_num, "error": str(e), "retryable": True})
                is_last_retry_for_this_model = retry_num == max_retries_per_model - 1
                if not is_last_retry_for_this_model:
                    delay = backoff_base_seconds * (2 ** retry_num)
                    time.sleep(delay)
                # else: retries exhausted for this model, fall through to next model

    raise LLMAllProvidersExhaustedError(attempts)


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python llm_client.py
# Uses injected fake transports so this never needs network access or a
# real OPENROUTER_API_KEY.

if __name__ == "__main__":
    import time as time_module

    # Speed up the self-test: real backoff isn't the thing under test here.
    time_module.sleep = lambda seconds: None

    messages = [{"role": "user", "content": "Summarize this patient's mood trend."}]
    PRIMARY, FALLBACK = DEFAULT_MODELS  # OpenRouter-prefixed, e.g. "openai/gpt-4o-mini"

    print("=== success on first try ===")
    def always_succeeds(model, msgs, timeout):
        return f"[{model}] ok response"
    result = generate(messages, transport=always_succeeds)
    print(result)
    assert result["model"] == PRIMARY
    assert result["attempts"] == []

    print("\n=== transient error retries then succeeds on same model ===")
    call_count = {"n": 0}
    def fails_twice_then_succeeds(model, msgs, timeout):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise LLMTransientError("simulated 503")
        return "recovered response"
    result = generate(messages, max_retries_per_model=5, transport=fails_twice_then_succeeds)
    print(result["text"], "| attempts:", len(result["attempts"]))
    assert result["text"] == "recovered response"
    assert result["model"] == PRIMARY  # same model, just retried
    assert len(result["attempts"]) == 2
    assert all(a["retryable"] for a in result["attempts"])

    print("\n=== transient errors exhaust retries on primary, falls back to secondary ===")
    def primary_always_fails_secondary_succeeds(model, msgs, timeout):
        if model == PRIMARY:
            raise LLMTransientError("simulated persistent 503")
        return f"[{model}] fallback response"
    result = generate(messages, max_retries_per_model=2, transport=primary_always_fails_secondary_succeeds)
    print(result["text"], "| model:", result["model"], "| attempts:", len(result["attempts"]))
    assert result["model"] == FALLBACK
    assert len(result["attempts"]) == 2  # both retries on the primary, then fallback succeeded
    assert all(a["model"] == PRIMARY for a in result["attempts"])

    print("\n=== client error (non-retryable) skips straight to next model, no retries burned ===")
    def primary_client_error(model, msgs, timeout):
        if model == PRIMARY:
            raise LLMClientError("simulated 400 bad request")
        return f"[{model}] recovered"
    result = generate(messages, max_retries_per_model=5, transport=primary_client_error)
    print(result["text"], "| attempts:", len(result["attempts"]))
    assert len(result["attempts"]) == 1  # NOT 5 -- client error doesn't retry
    assert result["attempts"][0]["retryable"] is False

    print("\n=== all providers exhausted raises with full attempt log ===")
    def always_fails(model, msgs, timeout):
        raise LLMTransientError(f"simulated persistent failure for {model}")
    try:
        generate(messages, max_retries_per_model=2, transport=always_fails)
        print("FAILED: should have raised LLMAllProvidersExhaustedError")
    except LLMAllProvidersExhaustedError as e:
        print(f"OK, raised with {len(e.attempts)} logged attempts")
        assert len(e.attempts) == 2 * len(DEFAULT_MODELS)  # 2 retries x 2 models

    print("\n=== missing API key on real transport raises LLMClientError, not a crash ===")
    os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        _default_transport(PRIMARY, messages, timeout=5.0)
        print("FAILED: should have raised LLMClientError")
    except LLMClientError as e:
        print(f"OK, raised: {e}")

    print("\nSelf-test passed.")