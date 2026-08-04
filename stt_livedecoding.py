"""
Fast STT benchmark (matches Tarz production pipeline) — LIVE DECODING VERSION.

Key change from the original: faster-whisper's `model.transcribe()` returns a
LAZY generator of segments. The original code did:

    text = "".join(s.text for s in segments).strip()

which silently blocks until every segment is decoded before printing anything.
This version iterates the generator directly and prints each segment the
moment it's ready, so the transcript appears "live" while decoding is still
in progress — same total decode time, much lower *perceived* latency.

Measures:
- Model load
- Warm-up
- Recording time
- Time-to-first-segment (TTFS) -- real latency-to-first-output
- Per-segment decode timing
- Final transcription latency (total)
- Real-time factor
"""

import time
import numpy as np

from microphone import Microphone
from stt import STT
from config import (
    SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURES,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_HINDI_PROMPT,
    WHISPER_TELUGU_PROMPT,
    WHISPER_MALAYALAM_PROMPT,
    WHISPER_ARABIC_PROMPT,
    WHISPER_HINDI_HOTWORDS,
    WHISPER_TELUGU_HOTWORDS,
    WHISPER_MALAYALAM_HOTWORDS,
    WHISPER_ARABIC_HOTWORDS,
    WHISPER_HINDI_PREFIX,
    WHISPER_TELUGU_PREFIX,
    WHISPER_MALAYALAM_PREFIX,
    WHISPER_ARABIC_PREFIX,
    STT_ALLOWED_LANGUAGES,
)

# If you want max speed over max accuracy, try beam_size=1 (greedy decoding)
# instead of WHISPER_BEAM_SIZE. Worth A/B testing on your real traffic —
# greedy is meaningfully faster but can be less accurate on noisy audio.
LIVE_BEAM_SIZE = WHISPER_BEAM_SIZE

# Only re-run the Hindi second pass if the first pass's confidence in "hi"
# is below this threshold. Previously the code ALWAYS re-decoded when
# language == "hi", even at high confidence — that's a wasted full decode.
HINDI_CONFIDENCE_REDECODE_THRESHOLD = 0.80


def warm_up(stt):
    t0 = time.perf_counter()

    try:
        segments, _ = stt.model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            without_timestamps=True,
        )
        list(segments)

    except Exception as e:
        print(f"Warm-up skipped: {e}")
        return

    print(f"Warm-up           : {time.perf_counter()-t0:.2f}s")


_GARBAGE_PATTERNS = {"???", "...", "[ Silence ]", "[BLANK_AUDIO]", "(inaudible)"}

def _is_hallucination(text: str) -> bool:
    """Catch repetition loops, known garbage tokens, and all-punctuation outputs."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in _GARBAGE_PATTERNS:
        return True
    # all non-alphanumeric / non-script characters → garbage
    if all(not c.isalpha() for c in stripped):
        return True
    if len(stripped) < 6:
        return False
    # repeated n-gram covers >60 % of the text
    for n in (2, 3, 4):
        chunk = stripped[:n]
        repeated = chunk * (len(stripped) // n)
        if stripped.startswith(repeated) and len(repeated) / len(stripped) > 0.6:
            return True
    return False


_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "ar": (0x0600, 0x06FF),  # Arabic
}


def _wrong_script(text: str, language: str) -> bool:
    """True when the transcript contains no characters from the expected script."""
    r = _SCRIPT_RANGES.get(language)
    if r is None or not text:
        return False
    return not any(r[0] <= ord(c) <= r[1] for c in text)


def _is_prompt_echo(text: str) -> bool:
    """True when Whisper parrotted the initial prompt instead of transcribing audio."""
    for prompt in (WHISPER_HINDI_PROMPT, WHISPER_TELUGU_PROMPT,
                   WHISPER_MALAYALAM_PROMPT, WHISPER_ARABIC_PROMPT):
        if prompt and any(word in text for word in prompt.split(",")):
            return True
    return False


def live_decode(stt, audio, language=None, label="Pass"):
    """
    Streams segments as faster-whisper decodes them, printing each one
    the instant it's ready instead of waiting for the whole utterance.

    Returns: (full_text, info, timings_dict)
    """

    # prefix hard-forces the correct script; hotwords provide additional vocab bias.
    hotwords = None
    prefix = None
    if language == "hi":
        hotwords = WHISPER_HINDI_HOTWORDS
        prefix = WHISPER_HINDI_PREFIX
    elif language == "te":
        hotwords = WHISPER_TELUGU_HOTWORDS
        prefix = WHISPER_TELUGU_PREFIX
    elif language == "ml":
        hotwords = WHISPER_MALAYALAM_HOTWORDS
        prefix = WHISPER_MALAYALAM_PREFIX
    elif language == "ar":
        hotwords = WHISPER_ARABIC_HOTWORDS
        prefix = WHISPER_ARABIC_PREFIX

    decode_start = time.perf_counter()

    segments, info = stt.model.transcribe(
        audio,
        beam_size=LIVE_BEAM_SIZE,
        language=language,
        task="transcribe",
        vad_filter=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        prefix=prefix,
        hotwords=hotwords,
        temperature=WHISPER_TEMPERATURES,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )

    # NOTE: `info` (language, language_probability) is available immediately —
    # faster-whisper runs language ID before returning, the generator only
    # lazily defers the actual segment decoding.

    chunks = []
    first_segment_time = None
    segment_times = []
    prev_t = decode_start

    print(f"\n[{label}] streaming transcript (lang={info.language}, "
          f"conf={info.language_probability:.2f}):")
    print("  ", end="", flush=True)

    for seg in segments:
        now = time.perf_counter()

        if first_segment_time is None:
            first_segment_time = now - decode_start

        segment_times.append(now - prev_t)
        prev_t = now

        chunks.append(seg.text)
        print(seg.text, end="", flush=True)

    print()  # newline after streamed transcript

    total_decode_time = time.perf_counter() - decode_start
    text = "".join(chunks).strip()

    # Remove the injected prefix so it doesn't bleed into the final transcript.
    if prefix and text.startswith(prefix):
        text = text[len(prefix):].strip()

    timings = {
        "ttfs": first_segment_time if first_segment_time is not None else total_decode_time,
        "segment_times": segment_times,
        "total_decode_time": total_decode_time,
    }

    return text, info, timings


def run():

    print("\n========== STT BENCHMARK (LIVE DECODING) ==========\n")

    load_start = time.perf_counter()

    mic = Microphone()
    stt = STT()

    print(f"Model load        : {time.perf_counter()-load_start:.2f}s")

    warm_up(stt)

    while True:

        try:
            input("\nPress ENTER and speak (Ctrl+C to quit)... ")

        except EOFError:
            break

        print("Listening...\n")

        record_start = time.perf_counter()

        audio = mic.listen()

        record_time = time.perf_counter() - record_start

        if len(audio) == 0:
            print("No audio captured.")
            continue

        audio_seconds = len(audio) / SAMPLE_RATE

        text, info, timings = live_decode(stt, audio, label="Pass 1")

        second_pass = 0.0
        second_pass_ttfs = 0.0

        needs_hindi_redecode = (
            getattr(info, "language", None) == "hi"
            and (info.language_probability < HINDI_CONFIDENCE_REDECODE_THRESHOLD
                 or _wrong_script(text, "hi"))
        )
        needs_telugu_redecode = (
            getattr(info, "language", None) == "te"
            and (info.language_probability < HINDI_CONFIDENCE_REDECODE_THRESHOLD
                 or _wrong_script(text, "te"))
        )
        needs_tamil_redecode = (
            getattr(info, "language", None) == "ml"
            and (info.language_probability < HINDI_CONFIDENCE_REDECODE_THRESHOLD
                 or _wrong_script(text, "ml"))
        )
        needs_arabic_redecode = (
            getattr(info, "language", None) == "ar"
            and (info.language_probability < HINDI_CONFIDENCE_REDECODE_THRESHOLD
                 or _wrong_script(text, "ar"))
        )
        # Whisper guessed a language outside the allowed set (e.g. ja).
        needs_fallback = getattr(info, "language", None) not in STT_ALLOWED_LANGUAGES

        if needs_hindi_redecode:

            hi_text, hi_info, hi_timings = live_decode(
                stt,
                audio,
                language="hi",
                label="Pass 2 (Hindi re-decode)",
            )

            second_pass = hi_timings["total_decode_time"]
            second_pass_ttfs = hi_timings["ttfs"]

            if (hi_text
                    and not _is_hallucination(hi_text)
                    and not _is_prompt_echo(hi_text)
                    and not _wrong_script(hi_text, "hi")):
                text = hi_text
                info = hi_info
            else:
                print("  [Pass 2 rejected — keeping Pass 1 result]")

        elif needs_telugu_redecode:

            te_text, te_info, te_timings = live_decode(
                stt,
                audio,
                language="te",
                label="Pass 2 (Telugu re-decode)",
            )

            second_pass = te_timings["total_decode_time"]
            second_pass_ttfs = te_timings["ttfs"]

            if (te_text
                    and not _is_hallucination(te_text)
                    and not _is_prompt_echo(te_text)
                    and not _wrong_script(te_text, "te")):
                text = te_text
                info = te_info
            else:
                print("  [Pass 2 rejected — keeping Pass 1 result]")

        elif needs_tamil_redecode:

            ta_text, ta_info, ta_timings = live_decode(
                stt,
                audio,
                language="ml",
                label="Pass 2 (Malayalam re-decode)",
            )

            second_pass = ta_timings["total_decode_time"]
            second_pass_ttfs = ta_timings["ttfs"]

            if (ta_text
                    and not _is_hallucination(ta_text)
                    and not _is_prompt_echo(ta_text)
                    and not _wrong_script(ta_text, "ml")):
                text = ta_text
                info = ta_info
            else:
                print("  [Pass 2 rejected — keeping Pass 1 result]")

        elif needs_arabic_redecode:

            ar_text, ar_info, ar_timings = live_decode(
                stt,
                audio,
                language="ar",
                label="Pass 2 (Arabic re-decode)",
            )

            second_pass = ar_timings["total_decode_time"]
            second_pass_ttfs = ar_timings["ttfs"]

            if (ar_text
                    and not _is_hallucination(ar_text)
                    and not _is_prompt_echo(ar_text)
                    and not _wrong_script(ar_text, "ar")):
                text = ar_text
                info = ar_info
            else:
                print("  [Pass 2 rejected — keeping Pass 1 result]")

        elif needs_fallback:

            print(f"  [Detected '{info.language}' — not in allowed set; trying hi/te/ta/ar fallback]")
            best_text, best_info, best_timings = None, None, None

            for lang in ("hi", "te", "ml", "ar"):
                fb_text, fb_info, fb_timings = live_decode(
                    stt, audio,
                    language=lang,
                    label=f"Fallback ({lang})",
                )
                second_pass += fb_timings["total_decode_time"]
                if second_pass_ttfs == 0.0:
                    second_pass_ttfs = fb_timings["ttfs"]

                if (fb_text
                        and not _is_hallucination(fb_text)
                        and not _is_prompt_echo(fb_text)
                        and not _wrong_script(fb_text, lang)):
                    if (best_info is None
                            or fb_info.language_probability > best_info.language_probability):
                        best_text, best_info, best_timings = fb_text, fb_info, fb_timings

            if best_text:
                text = best_text
                info = best_info
            else:
                print("  [All fallbacks rejected — keeping Pass 1 result]")

        decode_time = timings["total_decode_time"] + second_pass

        print("----------------------------------")
        print(f"Transcript        : {text}")
        print(f"Language          : {info.language}")
        print(f"Confidence        : {info.language_probability:.2f}")
        print("----------------------------------")
        print(f"Recording                    : {record_time:.2f}s")
        print(f"Time to first segment (TTFS) : {timings['ttfs']:.2f}s")
        print(f"Pass 1 decode time           : {timings['total_decode_time']:.2f}s")
        print(f"  segments in pass 1         : {len(timings['segment_times'])}")

        if second_pass:
            lang_label = "Telugu" if needs_telugu_redecode else "Hindi"
            print(f"{lang_label} 2nd pass TTFS               : {second_pass_ttfs:.2f}s")
            print(f"{lang_label} 2nd pass decode time        : {second_pass:.2f}s")

        print(f"Total time to generate transcript     : {decode_time:.2f}s")
        print(f"Audio length                  : {audio_seconds:.2f}s")
        print(f"Real-time Factor              : {decode_time/audio_seconds:.2f}x")

        if audio_seconds:
            speed = audio_seconds / decode_time
            print(f"Speed                         : {speed:.2f}x realtime")

        print("----------------------------------")


if __name__ == "__main__":

    try:
        run()

    except KeyboardInterrupt:
        print("\nBye.")