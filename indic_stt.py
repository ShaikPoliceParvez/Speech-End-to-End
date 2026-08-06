"""
StreamingIndicTranscriber
=========================
Low-latency streaming ASR built on top of indic_asr_onnx.IndicTranscriber.

Key design decisions (based on ONNX model inspection):
- Encoder has NO state I/O  →  each audio chunk is encoded independently.
- RNNT decoder HAS explicit LSTM state I/O (states.1 / onnx::Slice_3 in,
  states / 162 out)  →  decoder runs incrementally across chunks, one token
  at a time, carrying h/c forward instead of re-processing the full token
  history on every emission (which is what the original library does).
- vocab.json and language_masks.json are cached once at construction.
- The language-specific joint_post_net session is owned here so switching
  the base transcriber's language does not corrupt our session pointer.
- No WAV files. No disk I/O after init.

Pipeline per feed() call:
    np.ndarray (audio chunk, 16 kHz float32)
        │
        ▼
    _preprocess()          ←  mel-spectrogram + log + CMVN (per-chunk)
        │
        ▼
    _encode()              ←  encoder ONNX + joint_enc projection  [1,T',640]
        │
        ▼
    _decode_enc_output()   ←  greedy RNNT, one encoder frame at a time
        │                     LSTM state (h, c) preserved across chunks
        ▼
    partial transcript
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch

from indic_asr_onnx import IndicTranscriber


class StreamingIndicTranscriber:
    """
    Streaming RNNT transcriber over IndicConformer.

    Multilingual usage::

        base = IndicTranscriber()
        # Pre-load adapters for all languages you need
        s = StreamingIndicTranscriber(base, languages=["te", "hi", "ml"])

        # Switch language between utterances
        s.set_language("hi")
        info = s.feed(audio_chunk)

        # Or pass language per-call
        info = s.feed(audio_chunk, language="te")

    Single-language usage (backwards-compatible)::

        s = StreamingIndicTranscriber(base, language="te")
    """

    BLANK_ID: int = 256
    SAMPLE_RATE: int = 16_000

    def __init__(
        self,
        base: IndicTranscriber,
        language: Optional[str] = None,
        languages: Optional[list] = None,
        chunk_ms: int = 480,
    ) -> None:
        """
        Parameters
        ----------
        base        : Already-initialised IndicTranscriber.
        language    : Single ISO code for backwards-compatibility ("te", "hi", ...).
        languages   : List of ISO codes to pre-load (e.g. ["te", "hi", "ml"]).
                      Pass this when you want multilingual support.
        chunk_ms    : Audio chunk size in milliseconds.
        """
        self._base = base
        self._chunk_samples = int(chunk_ms * self.SAMPLE_RATE / 1000)

        # Resolve which languages to load
        _langs = languages or ([language] if language else [])
        if not _langs:
            raise ValueError("Provide language= or languages=")

        # Load shared RNNT sessions once (encoder + rnnt + joint_enc/pred/pre_net)
        base._load_rnnt_models(_langs[0])

        # ── Load vocab and adapters for every requested language ───────
        _vocab_path = os.path.join(base.model_dir, "config", "vocab.json")
        with open(_vocab_path, "r", encoding="utf-8") as f:
            _all_vocab = json.load(f)

        self._vocabs: dict = {}
        self._post_nets: dict = {}
        for lang in _langs:
            self._vocabs[lang] = _all_vocab[lang]
            _post_path = os.path.join(
                base.model_dir, "onnx",
                f"adapters/joint_post_net_{lang}_quantized_int8.onnx",
            )
            _sess = ort.InferenceSession(_post_path, providers=base.providers)
            self._post_nets[lang] = (_sess, _sess.get_inputs()[0].name)

        # Set initial active language
        self._lang: str = _langs[0]
        self._post_net, self._post_in = self._post_nets[self._lang]
        self._vocab: list = self._vocabs[self._lang]

        # ── Cache input names to avoid per-call .get_inputs() ─────────
        _enc_ins = base.encoder_sess.get_inputs()
        self._enc_audio = _enc_ins[0].name
        self._enc_len: Optional[str] = _enc_ins[1].name if len(_enc_ins) > 1 else None
        self._joint_enc_in = base.joint_enc_sess.get_inputs()[0].name
        self._joint_pred_in = base.joint_pred_sess.get_inputs()[0].name
        self._pre_net_in = base.joint_pre_net_sess.get_inputs()[0].name

        # ── Runtime state ──────────────────────────────────────────────
        self._buffer = np.empty(0, dtype=np.float32)
        self._chunk_count = 0
        self._reset_decoder()

    # ──────────────────────────────────────────────────────────────────
    # Internal state helpers
    # ──────────────────────────────────────────────────────────────────

    def _reset_decoder(self) -> None:
        # RNNT LSTM states: h [2,1,640], c [2,1,640]
        self._h = np.zeros((2, 1, 640), dtype=np.float32)
        self._c = np.zeros((2, 1, 640), dtype=np.float32)
        # Token history (starts with BLANK sentinel, matching original library)
        self._tokens: list[int] = [self.BLANK_ID]
        # Predictor embedding for current last token; None = needs init
        self._pred_current: Optional[np.ndarray] = None
        self._text = ""

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state.  Call between utterances."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._chunk_count = 0
        self._reset_decoder()

    def set_language(self, language: str) -> None:
        """
        Switch the active language.  The language must have been listed in
        languages= at construction time.
        Resets decoder state (new language = new utterance context).
        """
        if language not in self._post_nets:
            raise ValueError(
                f"Language '{language}' not loaded.  "
                f"Available: {list(self._post_nets.keys())}"
            )
        if language != self._lang:
            self._lang = language
            self._post_net, self._post_in = self._post_nets[language]
            self._vocab = self._vocabs[language]
            self._reset_decoder()

    @property
    def language(self) -> str:
        return self._lang
        self._buffer = np.empty(0, dtype=np.float32)
        self._chunk_count = 0
        self._reset_decoder()

    def feed(self, audio_chunk: np.ndarray, language: Optional[str] = None) -> dict:
        """
        Feed a raw float32 audio chunk (any length, 16 kHz).

        language : optional per-call override.  If provided, switches the
                   active language before transcribing (must have been loaded
                   at construction time).
        """
        if language is not None:
            self.set_language(language)
        t0 = time.perf_counter()
        self._chunk_count += 1
        self._buffer = np.concatenate([self._buffer, audio_chunk])
        input_duration_ms = len(audio_chunk) / self.SAMPLE_RATE * 1000

        if len(self._buffer) < self._chunk_samples:
            return {
                "chunk": self._chunk_count,
                "partial": self._text,
                "chunk_ms": round(input_duration_ms, 1),
                "prep_ms": 0.0,
                "encoder_ms": 0.0,
                "decoder_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "rtf": 0.0,
                "buffering": True,
            }

        chunk = self._buffer[: self._chunk_samples]
        self._buffer = self._buffer[self._chunk_samples :]
        chunk_ms = self._chunk_samples / self.SAMPLE_RATE * 1000

        t1 = time.perf_counter()
        features = self._preprocess(chunk)
        prep_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        enc_out = self._encode(features)
        enc_ms = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        self._decode_enc_output(enc_out)
        dec_ms = (time.perf_counter() - t3) * 1000

        self._text = self._detokenize()
        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "chunk": self._chunk_count,
            "partial": self._text,
            "chunk_ms": round(chunk_ms, 1),
            "prep_ms": round(prep_ms, 1),
            "encoder_ms": round(enc_ms, 1),
            "decoder_ms": round(dec_ms, 1),
            "total_ms": round(total_ms, 1),
            "rtf": round(total_ms / chunk_ms, 3),
            "buffering": False,
        }

    def finalize(self) -> dict:
        """
        Flush any remaining buffered audio and return the final result.
        Safe to call even if buffer is empty.
        """
        if len(self._buffer) >= 160:   # at least one mel frame (160 samples)
            # Zero-pad to a multiple of hop_length so mel is well-defined
            pad = (-len(self._buffer)) % self._chunk_samples
            padded = np.concatenate([self._buffer, np.zeros(pad, dtype=np.float32)])
            self._buffer = np.empty(0, dtype=np.float32)
            return self.feed(padded)
        return {
            "chunk": self._chunk_count,
            "partial": self._text,
            "chunk_ms": 0.0,
            "prep_ms": 0.0,
            "encoder_ms": 0.0,
            "decoder_ms": 0.0,
            "total_ms": 0.0,
            "rtf": 0.0,
            "buffering": False,
        }

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> dict:
        """
        Transcribe a complete utterance in one shot (no chunking).
        This is the primary integration entry-point for the STT router.

        Returns the same feed() dict shape for consistency.
        """
        if language is not None:
            self.set_language(language)
        self.reset()
        old = self._chunk_samples
        self._chunk_samples = max(len(audio), 1)  # treat the whole audio as one chunk
        try:
            return self.feed(audio)
        finally:
            self._chunk_samples = old  # restore so streaming use still works

    def get_partial(self) -> str:
        return self._text

    # ──────────────────────────────────────────────────────────────────
    # Preprocessing  (identical math to base._preprocess_audio)
    # ──────────────────────────────────────────────────────────────────

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """float32 numpy [T] → mel features float32 [80, T_frames]."""
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self._base.device)
        mel = self._base.mel_transform(wav)          # [1, 80, T_frames]
        mel = torch.log(mel + 1e-9)
        mean = mel.mean(dim=2, keepdim=True)
        std = mel.std(dim=2, keepdim=True) + 1e-5
        mel = (mel - mean) / std
        return mel.squeeze(0).cpu().numpy().astype(np.float32)  # [80, T_frames]

    # ──────────────────────────────────────────────────────────────────
    # Encoder
    # ──────────────────────────────────────────────────────────────────

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """
        [80, T_frames] → joint-enc projected output [1, T_enc, 640].

        Runs: encoder (1024-dim) → joint_enc (640-dim projection).
        """
        feats = features[np.newaxis]                              # [1, 80, T]
        length = np.array([features.shape[1]], dtype=np.int64)
        enc_in: dict = {self._enc_audio: feats}
        if self._enc_len:
            enc_in[self._enc_len] = length
        enc_out = self._base.encoder_sess.run(None, enc_in)[0]   # [1, 1024, T']

        enc_t = enc_out.transpose(0, 2, 1)                       # [1, T', 1024]
        return self._base.joint_enc_sess.run(
            None, {self._joint_enc_in: enc_t}
        )[0]                                                       # [1, T', 640]

    # ──────────────────────────────────────────────────────────────────
    # RNNT decoder — incremental with state carry-forward
    # ──────────────────────────────────────────────────────────────────

    def _decoder_step(self, token_id: int) -> None:
        """
        Advance the RNNT prediction network by exactly one token.

        Passes the saved LSTM state (h, c) and updates it in-place.
        This is O(1) regardless of history length, unlike the original
        library which re-processes the full token sequence each time.

        The equivalence to full-sequence processing follows from LSTM
        determinism: run([t0,t1,...,tk], h0=0) ≡ run([tk], h0=h_k-1).
        """
        inp = np.array([[token_id]], dtype=np.int32)              # [1, 1]
        out = self._base.rnnt_sess.run(None, {
            "targets":        inp,
            "target_length":  np.array([1], dtype=np.int32),
            "states.1":       self._h,
            "onnx::Slice_3":  self._c,
        })
        # out[0]: outputs [1, 640, 1]
        # out[2]: updated hidden state h [2, 1, 640]
        # out[3]: updated cell  state c [2, 1, 640]
        last_emb = out[0].transpose(0, 2, 1)[:, -1:, :]          # [1, 1, 640]
        self._h = out[2]
        self._c = out[3]
        self._pred_current = self._base.joint_pred_sess.run(
            None, {self._joint_pred_in: last_emb}
        )[0]                                                       # [1, 1, 640]

    def _decode_enc_output(self, enc_output: np.ndarray) -> None:
        """
        Greedy RNNT decode of one chunk's encoder output.
        Emits tokens into self._tokens, updates LSTM state in-place.
        """
        # Bootstrap predictor on first call after reset
        if self._pred_current is None:
            self._decoder_step(self._tokens[-1])    # seed with BLANK

        T = enc_output.shape[1]
        t = 0

        while t < T:
            enc_frame = enc_output[:, t : t + 1, :]               # [1, 1, 640]
            joint = enc_frame + self._pred_current                 # [1, 1, 640]

            pre = self._base.joint_pre_net_sess.run(
                None, {self._pre_net_in: joint}
            )[0]
            logits = self._post_net.run(
                None, {self._post_in: pre}
            )[0]                                                    # [1, 1, 257]

            k = int(np.argmax(logits[0, 0, :]))

            if k == self.BLANK_ID:
                t += 1
            else:
                self._tokens.append(k)
                self._decoder_step(k)

    # ──────────────────────────────────────────────────────────────────
    # Detokenization  (identical to original library)
    # ──────────────────────────────────────────────────────────────────

    def _detokenize(self) -> str:
        out, prev = [], None
        for idx in self._tokens[1:]:                              # skip leading BLANK
            if idx != prev and idx != self.BLANK_ID and idx < len(self._vocab):
                out.append(self._vocab[idx])
            prev = idx
        return "".join(out).replace("▁", " ").strip()
