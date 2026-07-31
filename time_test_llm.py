"""
Standalone LLM benchmark (fast + streamed input).

Assumes the prompt is arriving as a *stream* (e.g. live STT partials) instead
of a single finished string. The response is generated with latency-tuned
Ollama options and a warm-up pass so the first token comes back as fast as
possible.

Run:
    python time_test_llm.py

This file only *reads* the existing modules/config and talks to Ollama
directly for the fast path. It does not modify or affect the main app.
"""

import time

import ollama

from language import detect_dominant_language
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, LLM_MODEL, LLM_MAX_TOKENS


# Latency-tuned decode options. Narrow sampling + capped context = fewer
# tokens to score per step = faster first token and higher throughput.
FAST_OPTIONS = {
    "num_predict": LLM_MAX_TOKENS,
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
}

# Keep the model resident in memory between turns so we never pay reload cost.
KEEP_ALIVE = "10m"


def _system_instruction(language):
    name = SUPPORTED_LANGUAGES.get(language, "English")
    if language == "hi":
        return (
            "The user's conversation language is Hindi. "
            "Always respond ONLY in proper Hindi using Devanagari script. "
            "Keep answers direct and to the point."
        )
    return (
        f"Current conversation language is {name}. "
        f"Always answer ONLY in {name}. Keep answers direct and to the point."
    )


def warm_up(model):
    """Preload the model weights so the first real turn isn't cold."""
    t0 = time.perf_counter()
    try:
        for _ in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={"num_predict": 1, "temperature": 0.0},
        ):
            pass
    except Exception as e:
        print(f"(warm-up skipped: {e})")
        return
    print(f"Warm-up: {time.perf_counter() - t0:.2f}s")


def stream_input(text, delay=0.03):
    """
    Simulate the prompt arriving as a stream of words (like live STT partials).
    Yields the growing text; the final yield is the complete prompt.
    """
    words = text.split()
    acc = []
    for w in words:
        acc.append(w)
        time.sleep(delay)
        yield " ".join(acc)


def generate(model, prompt, language):
    """Stream a response from Ollama with the fast options; report timing."""
    messages = [
        {"role": "system", "content": _system_instruction(language)},
        {"role": "user", "content": prompt},
    ]

    print("\nTarz: ", end="", flush=True)

    t0 = time.perf_counter()
    first_token_time = None
    chunks = 0
    chars = 0

    for part in ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        keep_alive=KEEP_ALIVE,
        options=FAST_OPTIONS,
    ):
        token = part["message"]["content"]
        if not token:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        chunks += 1
        chars += len(token)
        print(token, end="", flush=True)

    total = time.perf_counter() - t0

    print("\n\n--- Timings ---")
    print(f"Language            : {language}")
    print(f"Time to first token : {(first_token_time or 0.0):6.2f}s")
    print(f"Total generation    : {total:6.2f}s")
    print(f"Chunks / chars      : {chunks} / {chars}")
    if total > 0:
        print(
            f"Throughput          : {chunks / total:6.2f} chunks/s "
            f"({chars / total:6.1f} chars/s)"
        )
    print()


def run():
    print("=== LLM BENCHMARK (fast, streamed input) ===")
    print(f"Model: {LLM_MODEL}")

    warm_up(LLM_MODEL)
    print()

    previous_language = DEFAULT_LANGUAGE

    while True:
        try:
            prompt = input("You (blank to quit): ").strip()
        except EOFError:
            break

        if not prompt:
            break

        # ---- Consume the prompt as a stream (mimics live STT) ----
        in_start = time.perf_counter()
        streamed = ""
        print("  (input streaming) ", end="", flush=True)
        for streamed in stream_input(prompt):
            print(".", end="", flush=True)
        input_time = time.perf_counter() - in_start
        print(f" [input ready in {input_time:.2f}s]")

        language = detect_dominant_language(
            streamed, previous_language=previous_language
        )
        previous_language = language

        generate(LLM_MODEL, streamed, language)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nBye.")
