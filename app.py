import config
import msvcrt
import threading
from microphone import Microphone
from stt import STT
from router import Router
from llm import LLM, sentence_stream
from tts_router import TTSRouter
from camera import Camera
from language import detect_dominant_language, normalize_text, detect_script
from tracing import LangfuseTracer


class Tarz:

    def __init__(self):

        print("Starting Tarz...\n")

        self.tracing = LangfuseTracer()
        self.mic = Microphone()
        self.stt = STT()
        self.router = Router()
        self.camera = Camera()
        self.llm = LLM(model=config.LLM_MODEL, tracer=self.tracing)
        self.tts = TTSRouter(on_event=self._on_event)
        self.tracing.set_model_startup_metrics(
            stt=self.stt.model_startup_metrics,
            llm=self.llm.measure_model_startup(),
            tts=self.tts.model_startup_metrics,
        )

    def _on_event(self, name, data):
        if config.DEBUG:
            print(f"[EVENT] {name}: {data}")

    @staticmethod
    def _return_to_menu_requested(text):
        command = text.strip().lower().strip(".,!?;:。")
        return command in {
            "/menu", "0", "menu", "back", "go back", "back to menu",
            "menu par jao", "menu par wapas jao", "मेनू पर जाओ", "मेनू पर वापस जाओ",
            "మెనూకి వెళ్ళు", "మెనూకి వెళ్లు", "మెనూకు వెళ్ళు",
        }

    def _watch_for_barge_in(self, stop_event):
        # Ignore Enter left in the console input buffer before speech begins.
        while msvcrt.kbhit():
            msvcrt.getwch()

        while not stop_event.wait(0.05):
            if not msvcrt.kbhit():
                continue

            key = msvcrt.getwch()
            if self.tts.is_speaking() and key in ("\r", "\n"):
                self.tts.stop()
                print("\nTarz: (stopped - listening for your next question)")
                return

    def process(
        self,
        text,
        stt_language_hint=None,
        stt_language_confidence=None,
        turn=None,
        input_mode=None,
        request_start=None,
        pipeline_metrics=None,
    ):

        if not text.strip():
            return

        input_mode = input_mode or ("voice" if stt_language_hint else "text")
        request_start = request_start or self.tracing.now()
        owns_turn = turn is None
        if owns_turn:
            with self.tracing.turn_attributes():
                with self.tracing.start_turn(text, None, "CHAT", input_mode) as root_turn:
                    self._process(
                        text,
                        stt_language_hint,
                        stt_language_confidence,
                        root_turn,
                        input_mode,
                        request_start,
                        pipeline_metrics,
                    )
        else:
            self._process(
                text,
                stt_language_hint,
                stt_language_confidence,
                turn,
                input_mode,
                request_start,
                pipeline_metrics,
            )

        if owns_turn:
            self.tracing.flush()

    def _process(self, text, stt_language_hint, stt_language_confidence, turn, input_mode, request_start, pipeline_metrics):

        print(f"\nYou: {text}")

        # ---- Language + script detection, Hinglish normalization ----
        with self.tracing.start_step(
            "Language",
            input={
                "message": text,
                "stt_language_hint": stt_language_hint,
                "stt_language_confidence": stt_language_confidence,
            },
        ) as classification:
            script = detect_script(text)
            language = detect_dominant_language(
                text,
                stt_hint=stt_language_hint,
                stt_confidence=stt_language_confidence,
                previous_language=self.llm.memory.get_language(),
            )
            normalized_text = normalize_text(text, language)
            self.llm.memory.set_input_script(script)
            self.tracing.update_step(
                classification,
                output={
                    "language": language,
                    "script": script,
                    "normalized_message": normalized_text,
                },
            )

        if normalized_text != text:
            print(f"Normalized: {normalized_text}")

        language_name = config.SUPPORTED_LANGUAGES.get(language, language)
        print(f"Language: {language_name} | Script: {script}")

        # ---- Intent routing ----
        router_start = self.tracing.now()
        with self.tracing.start_step(
            "Router",
            input={"message": normalized_text},
        ) as routing:
            route = self.router.route(normalized_text)
            intent = route["intent"]
            self.tracing.update_step(
                routing,
                output={"selected_tool": intent, "confidence": route["confidence"]},
                metadata={
                    "reason": route["reason"],
                    "latency_ms": self.tracing.elapsed_ms(router_start),
                },
            )

        self.tracing.update_turn(turn, normalized_text, language, intent)

        with self.tracing.start_step(
            "Memory",
            metadata={"type": "conversation-history", "rag_enabled": False},
        ) as memory:
            history = self.llm.memory.messages()
            self.tracing.update_step(
                memory,
                output={"message_count": len(history), "retrieved_chunks": 0},
            )

        image = None
        vision_requested = intent in ("VISION", "OCR")
        if vision_requested and self.llm.supports_vision():
            with self.tracing.start_step(
                "Camera",
                metadata={"image_included_in_trace": False},
                as_type="tool",
            ) as capture:
                print("Opening camera...")
                image, image_size = self.camera.capture()
                self.tracing.update_step(
                    capture,
                    output={"captured": True, "image_size": image_size},
                )
        elif vision_requested:
            print("Camera analysis is unavailable with the configured text-only model.")

        # ---- LLM streaming -> sentence buffering -> TTS streaming ----
        self.tts.start_turn()
        self.tts.set_language(language)
        barge_in_stop = threading.Event()
        barge_in_listener = None
        print("Press ENTER to interrupt Tarz and ask your next question.")
        barge_in_listener = threading.Thread(
            target=self._watch_for_barge_in,
            args=(barge_in_stop,),
            daemon=True,
        )
        barge_in_listener.start()

        token_stream = self.llm.stream(
            prompt=normalized_text,
            image=image,
            vision_requested=vision_requested,
            language=language,
        )

        sentences = sentence_stream(
            token_stream,
            min_chars=config.TTS_MIN_CHARS,
            min_words=config.TTS_MIN_WORDS,
            max_chars=config.TTS_MAX_CHARS,
            max_words=config.TTS_MAX_WORDS,
            first_sentence_immediately=config.TTS_FIRST_SENTENCE_IMMEDIATELY,
            should_stop=self.tts.is_interrupted,
        )

        full_response = []

        def relay(stream):
            for sentence in stream:
                full_response.append(sentence)
                print(f"Tarz: {sentence}")
                yield sentence

        tts = self.tracing.start_manual_step(
            turn,
            "TTS",
            metadata={"language": language, "engine": self.tts.backend_name, "voice": config.VOICE},
        )
        playback = self.tracing.start_manual_step(
            turn,
            "Playback",
            metadata={"engine": "sounddevice"},
        )
        try:
            self.tts.speak_stream(relay(sentences))
            self.tts.wait_until_idle()
        finally:
            barge_in_stop.set()
            if barge_in_listener is not None:
                barge_in_listener.join(timeout=0.1)
        tts_metrics = self.tts.get_turn_metrics()
        self.tracing.update_step(
            tts,
            output={
                "first_audio_latency_ms": tts_metrics["first_audio_latency_ms"],
                "synthesis_duration_ms": tts_metrics["synthesis_duration_ms"],
                "audio_duration_ms": round(tts_metrics["audio_duration_ms"], 2),
                "chunk_count": tts_metrics["chunk_count"],
            },
            metadata={"sample_rate": tts_metrics["sample_rate"], "speed": config.TTS_SPEED},
        )
        self.tracing.update_step(
            playback,
            output={
                "playback_duration_ms": tts_metrics["playback_duration_ms"],
                "queue_delay_ms": tts_metrics["queue_delay_ms"],
                "audio_duration_ms": round(tts_metrics["audio_duration_ms"], 2),
            },
        )
        if tts is not None:
            tts.end()
        if playback is not None:
            playback.end()

        if not full_response:
            print("Tarz: (no response)")

        if turn is not None:
            turn.update(output={"response": " ".join(full_response)})
            total_latency_ms = self.tracing.elapsed_ms(request_start)
            timing = {
                "total_interaction_latency_ms": total_latency_ms,
                "router_latency_ms": self.tracing.elapsed_ms(router_start),
                "llm": self.llm.last_metrics,
                "tts": {
                    "synthesis_duration_ms": tts_metrics["synthesis_duration_ms"],
                    "playback_duration_ms": tts_metrics["playback_duration_ms"],
                },
            }
            timing.update(pipeline_metrics or {})
            for name, duration in {
                "router_percent": timing["router_latency_ms"],
                "llm_percent": self.llm.last_metrics.get("total_latency_ms", 0),
                "tts_percent": tts_metrics["synthesis_duration_ms"] or 0,
                "playback_percent": tts_metrics["playback_duration_ms"] or 0,
                "stt_percent": timing.get("stt_latency_ms", 0),
            }.items():
                timing[name] = round(duration / max(total_latency_ms, 0.001) * 100, 2)
            self.tracing.update_turn_metrics(turn, timing)

    def run_voice(self):

        print("Say 'back to menu' to choose a different mode.")

        while True:

            request_start = self.tracing.now()
            with self.tracing.turn_attributes():
                with self.tracing.start_turn("", None, "VOICE", "voice") as turn:
                    with self.tracing.start_step(
                        "VAD",
                        metadata={"audio_included_in_trace": False},
                    ) as recording:
                        audio, vad_metrics = self.mic.listen(return_metrics=True)
                        audio_seconds = len(audio) / config.SAMPLE_RATE
                        self.tracing.update_step(
                            recording,
                            output={
                                "audio_duration_seconds": round(audio_seconds, 3),
                                **vad_metrics,
                            },
                        )

                    with self.tracing.start_step(
                        "STT",
                        metadata={
                            "engine": "faster-whisper",
                            "model": config.WHISPER_SIZE,
                            "beam_size": config.WHISPER_BEAM_SIZE,
                            "compute_type": config.WHISPER_COMPUTE,
                            "audio_duration_seconds": round(audio_seconds, 3),
                            "audio_included_in_trace": False,
                            "partial_transcript_supported": False,
                        },
                    ) as transcription:
                        result = self.stt.transcribe(audio)
                        self.tracing.update_step(
                            transcription,
                            output={
                                "text": result["text"],
                                "language": result["language"],
                                "confidence": result["confidence"],
                                "first_segment_ms": result["first_segment_ms"],
                                "transcript_length": len(result["text"]),
                                "latency_ms": result["latency_ms"],
                            },
                        )

                    if self._return_to_menu_requested(result["text"]):
                        print("Returning to mode selection...")
                        self.tracing.record_event(turn, "Conversation Ended", {"returned_to_menu": True})
                        self.tracing.flush()
                        return

                    self.process(
                        result["text"],
                        stt_language_hint=result["language"],
                        stt_language_confidence=result["confidence"],
                        turn=turn,
                        input_mode="voice",
                        request_start=request_start,
                        pipeline_metrics={
                            "vad": vad_metrics,
                            "stt_latency_ms": result["latency_ms"],
                        },
                    )

                    self.tracing.record_event(
                        turn,
                        "Conversation Ended",
                        {"response_completed": True},
                    )


            self.tracing.flush()

    def run_text(self):

        while True:

            text = input("\nYou (/menu to change mode): ")

            if self._return_to_menu_requested(text):
                print("Returning to mode selection...")
                return

            self.process(text)


if __name__ == "__main__":

    tarz = Tarz()

    while True:
        mode = input(
            "\nChoose mode\n"
            "1. Voice\n"
            "2. Text\n"
            "0. Exit\n\n"
            "Choice: "
        ).strip()

        if mode == "1":
            tarz.run_voice()
        elif mode == "2":
            tarz.run_text()
        elif mode == "0":
            break
        else:
            print("Please choose 1, 2, or 0.")


