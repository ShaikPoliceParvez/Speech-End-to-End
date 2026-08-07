from faster_whisper import WhisperModel
import numpy as np
import time
import re

# IndicConformer import is optional; missing package falls back to Whisper-only.
try:
    from indic_asr_onnx import IndicTranscriber as _IndicBase
    from indic_stt import StreamingIndicTranscriber as _IndicStreamer
    _INDIC_PKG_OK = True
except ImportError:
    _INDIC_PKG_OK = False

# Phrases Whisper commonly hallucinates on near-silent / ambient-noise input.
# Matching is case-insensitive against stripped, punctuation-stripped text.
_HALLUCINATION_PHRASES = frozenset({
    "thank you", "thanks", "thank you for watching",
    "thank you for watching this video", "please subscribe",
    "like and subscribe", "subtitles by", "subtitle by",
    "you", "bye", "okay", "ok", "um", "uh", "ah", "hmm",
    # Common non-English hallucinations
    "धन्यवाद", "शुक्रिया", "ధన్యవాదాలు", "നന്ദി", "شكرا",
})

from config import (
    WHISPER_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURES,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_HINDI_PROMPT,
    WHISPER_TELUGU_PROMPT,
    WHISPER_MALAYALAM_PROMPT,
    WHISPER_ARABIC_PROMPT,
    WHISPER_LANGUAGE_CONFIDENCE_HIGH,
    STT_RETRY_ON_LOW_CONFIDENCE,
    STT_ALLOWED_LANGUAGES,
    STT_INDIC_ASR_ENABLED,
    STT_INDIC_LANGUAGES,
    DEFAULT_LANGUAGE,
    HINDI_ROMAN_CORE_WORDS,
    TELUGU_ROMAN_CORE_WORDS,
    MALAYALAM_ROMAN_CORE_WORDS,
    ARABIC_ROMAN_CORE_WORDS,
    ENGLISH_CORE_WORDS,
    TECHNICAL_BORROWED_WORDS,
)


class STT:
    def __init__(self, verbose=False):
        self.verbose = verbose

        print("Loading Whisper...")
        model_start = time.perf_counter()

        self.model = WhisperModel(
            WHISPER_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        self.model_startup_metrics = {
            "model": WHISPER_SIZE,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE,
            "model_startup_ms": round((time.perf_counter() - model_start) * 1000, 2),
        }

        print("Whisper Ready")

        # ── IndicConformer (optional; Whisper remains the fallback) ───
        self._indic: _IndicStreamer | None = None
        if STT_INDIC_ASR_ENABLED and _INDIC_PKG_OK:
            _langs = sorted(STT_INDIC_LANGUAGES & set(STT_ALLOWED_LANGUAGES))
            if _langs:
                try:
                    print(f"Loading IndicConformer ({', '.join(_langs)})...")
                    _t0 = time.perf_counter()
                    _base = _IndicBase()
                    self._indic = _IndicStreamer(_base, languages=_langs)
                    self.model_startup_metrics["indic_ms"] = round(
                        (time.perf_counter() - _t0) * 1000, 2
                    )
                    print(f"IndicConformer Ready ({', '.join(_langs)})")
                except Exception as _e:
                    print(f"[STT] IndicConformer unavailable: {_e} — using Whisper only")

    def _transcribe_indic(self, audio: np.ndarray, language: str) -> "dict | None":
        """
        Attempt IndicConformer transcription for a known Indic language.
        Returns a result dict on success, None in all failure cases so the
        caller falls back to Whisper (covers both exceptions and language
        switches where IndicConformer produces no output).
        """
        try:
            raw = self._indic.transcribe(audio, language=language)
            text = (raw.get("partial") or "").strip()
            # Empty output on non-silent audio means language switch or unclear
            # speech; return None so Whisper auto-detect handles it.
            if not text:
                return None
            detected_script = self.detect_script(text)
            print(f"[DEBUG STT] IndicConformer({language}) returned: '{text}' | Script: {detected_script}")
            return {
                "text": text,
                "language": language,
                "confidence": 0.95,           # IndicConformer gives no per-utterance score
                "script": detected_script,
                "first_segment_ms": raw.get("encoder_ms"),
                "first_partial_transcript": text,
                "latency_ms": raw.get("total_ms", 0.0),
            }
        except Exception as exc:
            print(f"[STT] IndicConformer error ({exc}) \u2014 falling back to Whisper")
            return None

    def _decode(self, audio, language, final):
        initial_prompt = {
            "hi": WHISPER_HINDI_PROMPT,
            "te": WHISPER_TELUGU_PROMPT,
            "ml": WHISPER_MALAYALAM_PROMPT,
            "ar": WHISPER_ARABIC_PROMPT,
        }.get(language)
        return self.model.transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE if final else 1,
            language=language,
            task="transcribe",
            vad_filter=final,
            without_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            temperature=WHISPER_TEMPERATURES if final else 0.0,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        )

    def _retry_supported_languages(self, audio, final, candidates=None):
        """
        Retry only with supported languages and select the cleanest
        transcription, not merely the largest language probability.
        """
        best_segments = None
        best_info = None
        best_score = -1
        best_first_segment_ms = None

        retry_start = time.perf_counter()
        for language in (candidates or STT_ALLOWED_LANGUAGES):
            try:
                segments, info = self._decode(audio, language, final)
                segments = list(segments)
                text = "".join(segment.text for segment in segments).strip()
                score = self._calculate_score(text, language, info, segments)

                if score > best_score:
                    best_score = score
                    best_segments = segments
                    best_info = info
                    best_first_segment_ms = round(
                        (time.perf_counter() - retry_start) * 1000, 2
                    )
            except Exception:
                continue

        return best_segments, best_info, best_first_segment_ms

    @staticmethod
    def _script_matches_language(language, script):
        """Allow scripts normally produced by each supported language."""
        expected_scripts = {
            "en": {"latin", "unknown", "mixed"},
            "hi": {"devanagari", "latin", "mixed", "unknown"},
            "te": {"telugu", "latin", "mixed", "unknown"},
            "ml": {"malayalam", "latin", "mixed", "unknown"},
            "ar": {"arabic", "mixed", "unknown"},
        }
        return script in expected_scripts.get(language, set())

    @staticmethod
    def _roman_language(text):
        """Identify decisive Roman-script language vocabulary across all supported languages."""
        tokens = [
            token.lower()
            for token in re.findall(r"[a-zA-Z]+", text)
            if token.lower() not in TECHNICAL_BORROWED_WORDS
        ]
        if not tokens:
            return None

        scores = {
            "en": sum(token in ENGLISH_CORE_WORDS for token in tokens),
            "hi": sum(token in HINDI_ROMAN_CORE_WORDS for token in tokens),
            "te": sum(token in TELUGU_ROMAN_CORE_WORDS for token in tokens),
            "ml": sum(token in MALAYALAM_ROMAN_CORE_WORDS for token in tokens),
            "ar": sum(token in ARABIC_ROMAN_CORE_WORDS for token in tokens),
        }
        language = max(scores, key=scores.get)
        hits = scores[language]
        if hits and list(scores.values()).count(hits) == 1 and (hits >= 2 or hits / len(tokens) >= 0.5):
            return language
        return None

    def _transcript_matches_language(self, text, language):
        """Verify language ID against native script and decisive Roman words."""
        script = self.detect_script(text)
        if not self._script_matches_language(language, script):
            return False

        roman_language = self._roman_language(text) if script == "latin" else None
        return roman_language is None or roman_language == language

    def _calculate_score(self, text, language, info, segments):
        """Score an already-decoded candidate; it never changes its transcript."""
        score = 3.0 * getattr(info, "language_probability", 0.0)
        script = self.detect_script(text)

        # Native script or decisive Roman vocabulary agreeing with the forced
        # decode is stronger evidence than Whisper's language ID alone.
        score += 1.5 if self._transcript_matches_language(text, language) else -1.5
        roman_language = self._roman_language(text) if script == "latin" else None
        if roman_language == language:
            score += 2.0
        elif roman_language is not None:
            score -= 2.0

        # Faster Whisper exposes per-segment quality metrics. Higher average
        # log probability and a normal compression ratio favor clean output.
        log_probabilities = [
            value for value in (getattr(segment, "avg_logprob", None) for segment in segments)
            if value is not None
        ]
        if log_probabilities:
            score += max(-1.0, min(0.0, sum(log_probabilities) / len(log_probabilities)))

        compression_ratios = [
            value for value in (getattr(segment, "compression_ratio", None) for segment in segments)
            if value is not None
        ]
        if compression_ratios:
            average_ratio = sum(compression_ratios) / len(compression_ratios)
            score += 0.5 if average_ratio <= WHISPER_COMPRESSION_RATIO_THRESHOLD else -1.0

        if self._is_suspicious_text(text):
            score -= 3.0
        return score

    @staticmethod
    def _is_suspicious_text(text):
        """Detect empty or repeated-token output that is not useful speech."""
        tokens = text.lower().split()
        return not text or (len(tokens) >= 3 and len(set(tokens)) == 1)

    def transcribe(
        self,
        audio: np.ndarray,
        language=None,
        final=True,
    ):
        """
        Parameters
        ----------
        audio : float32 numpy array
            Mono PCM at SAMPLE_RATE.

        Returns
        -------
        dict
        {
            text,
            language,
            confidence,
            script
        }
        """

        if len(audio) == 0:
            return self.empty_result()

        # Reject near-silent audio before invoking Whisper; it hallucinates
        # common phrases ("Thank you", etc.) on ambient-noise recordings.
        # Threshold lowered from 0.02 to 0.005 to support quieter microphones
        if float(np.max(np.abs(audio))) < 0.005:
            return self.empty_result()

        # -- Language detection strategy -----------------------------------------------
        # Phase 1: Fast probe to detect language (always run, even with hints)
        detected_language = None
        probe_confidence = 0.0
        try:
            # Fast Whisper probe: ~20ms, provides language + confidence
            detected_language, probe_confidence, _ = self.model.detect_language(audio)
            if detected_language not in STT_ALLOWED_LANGUAGES:
                detected_language = None
        except Exception:
            pass

        # Phase 2: IndicConformer fast path (Indic only, high-confidence or matching hint)
        # IMPORTANT: NEVER force a language hint from prior turn on IndicConformer.
        # Why? Language switches (Hindi → Telugu mid-conversation) are NOT detectable by hints alone.
        # Strategy: Always run Whisper's fast detect_language() probe first to catch switches.
        # If Whisper is confident about an Indic language AND it matches the hint, use IndicConformer.
        # If no match, let Whisper auto-detect (handles language switches correctly).
        effective_indic_lang = None
        if self._indic is not None and detected_language in STT_INDIC_LANGUAGES:
            # Whisper probe detected an Indic language; check if we should use IndicConformer
            if probe_confidence >= WHISPER_LANGUAGE_CONFIDENCE_HIGH:
                # High confidence: use IndicConformer with detected language
                effective_indic_lang = detected_language
            elif language is not None and language == detected_language and probe_confidence >= 0.60:
                # Hint matches probe + medium confidence: use IndicConformer for speed
                effective_indic_lang = language

        if effective_indic_lang is not None:
            result = self._transcribe_indic(audio, effective_indic_lang)
            if result is not None:
                return result
            # IndicConformer gave no output (exception or language truly switched).
            # Fall through to Whisper with the detected language (not None).
            language = detected_language  # Use probe result, not None

        # -- Whisper transcription with detected language -----------------------------------------------
        # Use the detected language from probe instead of language=None for more reliable auto-detect
        # If probe detected nothing, None is passed and Whisper does its internal auto-detect
        transcription_language = language if language is not None else detected_language

        transcription_start = time.perf_counter()
        segments, info = self._decode(audio, transcription_language, final)

        parts = []
        first_segment_ms = None
        for segment in segments:
            if first_segment_ms is None:
                first_segment_ms = round((time.perf_counter() - transcription_start) * 1000, 2)
            parts.append(segment.text)

        text = "".join(parts).strip()

        # Drop known Whisper hallucination phrases immediately — no retry helps.
        if text.lower().rstrip(".,!?…") in _HALLUCINATION_PHRASES:
            return self.empty_result()

        detected_language = getattr(info, "language", None)
        script = self.detect_script(text)
        roman_language = self._roman_language(text) if script == "latin" else None
        script_mismatch = not self._transcript_matches_language(text, detected_language)

        # Fast path: keep first-pass output when it is already usable, because
        # a second decode is expensive and is the main source of latency for
        # multilingual turns.
        first_pass_usable = (
            detected_language in STT_ALLOWED_LANGUAGES
            and bool(text)
            and not self._is_suspicious_text(text)
            and self._transcript_matches_language(text, detected_language)
            and (script != "latin" or roman_language in (None, detected_language))
        )

        # Retry only for strong failure signals.
        # Also retry when a forced language hint produced unusable output
        # (e.g. user switches language mid-conversation).
        hint_failed = language is not None and not first_pass_usable
        needs_retry = (
            (language is None or hint_failed)
            and not first_pass_usable
            and (
                hint_failed
                or (
                    STT_RETRY_ON_LOW_CONFIDENCE
                    and getattr(info, "language_probability", 0.0) < WHISPER_LANGUAGE_CONFIDENCE_HIGH
                )
                or
                detected_language not in STT_ALLOWED_LANGUAGES
                or script_mismatch
                or self._is_suspicious_text(text)
                or (
                    detected_language in ("hi", "te")
                    and script == "unknown"
                )
            )
        )

        if needs_retry:
            # Narrow candidates by script so we do 1 decode instead of all 5.
            if script == "devanagari":
                candidates = ("en", "hi")  # Devanagari is unambiguously Hindi/Urdu
            elif roman_language in STT_ALLOWED_LANGUAGES:
                # Roman-script evidence is a strong disambiguator for Hinglish,
                # Roman Telugu/Malayalam, and Arabizi.
                if roman_language == "hi":
                    candidates = ("hi", "en")
                else:
                    candidates = (roman_language, "en")
            elif detected_language == "hi":
                candidates = ("en", "hi")  # Hinglish (Latin) is the only real Hindi/English confusion
            elif script == "telugu":
                candidates = ("te", "en")  # include en for Telglish code-switching
            elif script == "malayalam":
                candidates = ("ml",)
            elif script == "arabic":
                candidates = ("ar",)
            elif detected_language in STT_ALLOWED_LANGUAGES:
                candidates = (detected_language,)
            else:
                candidates = STT_ALLOWED_LANGUAGES
            retry_segments, retry_info, retry_first_segment_ms = self._retry_supported_languages(
                audio, final, candidates
            )

            if retry_segments:
                segments = retry_segments
                info = retry_info
                parts = [segment.text for segment in segments]
                text = "".join(parts).strip()
                # The original first_segment_ms belonged to the discarded
                # decode; replace it with the timing of the retry that
                # actually produced the returned transcript.
                first_segment_ms = retry_first_segment_ms

        transcription_latency_ms = round((time.perf_counter() - transcription_start) * 1000, 2)

        detected_language = getattr(info, "language", DEFAULT_LANGUAGE)
        if detected_language not in STT_ALLOWED_LANGUAGES:
            detected_language = DEFAULT_LANGUAGE
        probability = getattr(info, "language_probability", 0.0)

        result = {
            "text": text,
            "language": detected_language,
            "confidence": probability,
            "script": self.detect_script(text),
            "first_segment_ms": first_segment_ms,
            "first_partial_transcript": parts[0].strip() if parts else None,
            "latency_ms": transcription_latency_ms,
        }

        if self.verbose:
            print(result)

        return result

    @staticmethod
    def detect_script(text):
        has_dev = any("\u0900" <= c <= "\u097F" for c in text)
        has_telugu = any("\u0C00" <= c <= "\u0C7F" for c in text)
        has_malayalam = any("\u0D00" <= c <= "\u0D7F" for c in text)
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
        has_lat = any(c.isascii() and c.isalpha() for c in text)

        # Multiple scripts are retained as "mixed" so verification does not
        # mistake an English borrowed word for a failed Telugu/Hindi transcript.
        if sum((has_dev, has_telugu, has_malayalam, has_arabic, has_lat)) > 1:
            return "mixed"

        if has_telugu:
            return "telugu"

        if has_malayalam:
            return "malayalam"

        if has_dev:
            return "devanagari"

        if has_arabic:
            return "arabic"

        if has_lat:
            return "latin"

        return "unknown"

    @staticmethod
    def empty_result():
        return {
            "text": "",
            "language": None,
            "confidence": 0.0,
            "script": "unknown",
            "first_segment_ms": None,
            "first_partial_transcript": None,
            "latency_ms": 0.0,
        }
