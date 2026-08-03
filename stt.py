from faster_whisper import WhisperModel
import numpy as np
import time
import re

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
    WHISPER_LANGUAGE_CONFIDENCE_HIGH,
    STT_ALLOWED_LANGUAGES,
    DEFAULT_LANGUAGE,
    HINDI_ROMAN_CORE_WORDS,
    TELUGU_ROMAN_CORE_WORDS,
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

    def _decode(self, audio, language, final):
        initial_prompt = {
            "hi": WHISPER_HINDI_PROMPT,
            "te": WHISPER_TELUGU_PROMPT,
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

    def _retry_supported_languages(self, audio, final):
        """
        Retry only with supported languages and select the cleanest
        transcription, not merely the largest language probability.
        """
        best_segments = None
        best_info = None
        best_score = -1
        best_first_segment_ms = None

        retry_start = time.perf_counter()
        for language in STT_ALLOWED_LANGUAGES:
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
        }
        return script in expected_scripts.get(language, set())

    @staticmethod
    def _roman_language(text):
        """Identify decisive Roman Telugu, Hindi, or English vocabulary."""
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

        transcription_start = time.perf_counter()
        segments, info = self._decode(audio, language, final)

        parts = []
        first_segment_ms = None
        for segment in segments:
            if first_segment_ms is None:
                first_segment_ms = round((time.perf_counter() - transcription_start) * 1000, 2)
            parts.append(segment.text)

        text = "".join(parts).strip()

        detected_language = getattr(info, "language", None)
        script = self.detect_script(text)
        script_mismatch = not self._transcript_matches_language(text, detected_language)

        # A second decode is expensive, so it is reserved for low-confidence,
        # unsupported, script-mismatched, Arabic/Persian, or empty results.
        # Hindi and Telugu both get the "unknown script" check: a hi/te
        # detection that produced no native/Latin script at all (e.g. only
        # digits or punctuation) is just as broken for either language.
        needs_retry = (
            language is None
            and (
                getattr(info, "language_probability", 0.0) < WHISPER_LANGUAGE_CONFIDENCE_HIGH
                or
                detected_language not in STT_ALLOWED_LANGUAGES
                or script_mismatch
                or script == "arabic"
                or self._is_suspicious_text(text)
                or (
                    detected_language in ("hi", "te")
                    and script == "unknown"
                )
            )
        )

        if needs_retry:
            retry_segments, retry_info, retry_first_segment_ms = self._retry_supported_languages(
                audio, final
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
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
        has_lat = any(c.isascii() and c.isalpha() for c in text)

        # Multiple scripts are retained as "mixed" so verification does not
        # mistake an English borrowed word for a failed Telugu/Hindi transcript.
        if sum((has_dev, has_telugu, has_arabic, has_lat)) > 1:
            return "mixed"

        if has_telugu:
            return "telugu"

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