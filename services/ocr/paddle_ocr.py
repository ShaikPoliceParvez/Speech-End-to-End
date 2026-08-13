"""Thread-safe, startup-initialized PaddleOCR service."""

from dataclasses import dataclass
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OCRResult:
    text: str
    source: str
    pages: int = 1


class PaddleOCRService:
    """Own one warmed PaddleOCR engine for the lifetime of the application."""

    _instance = None
    _instance_lock = Lock()

    def __init__(self, lang: str = "en"):
        if getattr(self, "_created", False):
            return
        self.lang = lang
        self._engine = None
        self._inference_lock = Lock()
        self._created = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def initialized(self):
        return self._engine is not None

    def initialize(self):
        """Create and warm the single OCR engine exactly once."""
        if self._engine is not None:
            return self

        with self._instance_lock:
            if self._engine is not None:
                return self

            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("FLAGS_log_level", "2")
            from paddleocr import PaddleOCR

            # Paddle/PaddleX writes progress to both streams during model setup.
            # Exceptions still propagate normally after the redirection ends.
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                previous_level = logging.root.manager.disable
                logging.disable(logging.INFO)
                try:
                    self._engine = PaddleOCR(
                        lang=self.lang,
                        device="cpu",
                        enable_mkldnn=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                    self._engine.predict(np.zeros((64, 64, 3), dtype=np.uint8))
                finally:
                    logging.disable(previous_level)
        return self

    @staticmethod
    def _as_dict(value: Any):
        if isinstance(value, dict):
            return value
        raw = getattr(value, "json", None)
        if callable(raw):
            raw = raw()
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def _read_result(cls, result: Any) -> list[str]:
        """Extract text from PaddleOCR 3.x results without coupling to one result class."""
        texts = []
        results = result if isinstance(result, (list, tuple)) else [result]
        for item in results:
            data = cls._as_dict(item)
            if not data:
                continue
            payload = data.get("res", data)
            for key in ("rec_texts", "texts"):
                values = payload.get(key)
                if isinstance(values, list):
                    texts.extend(str(value).strip() for value in values if str(value).strip())
                    break
        return texts

    def read_image(self, image_path: str | Path | bytes) -> OCRResult:
        """Read one image using the already-initialized engine."""
        self.initialize()
        source_name = str(image_path) if not isinstance(image_path, bytes) else "<memory>"
        if isinstance(image_path, bytes):
            prediction_source = np.asarray(Image.open(BytesIO(image_path)).convert("RGB"))
        else:
            prediction_source = str(image_path) if isinstance(image_path, Path) else image_path
        with self._inference_lock:
            result = self._engine.predict(prediction_source)
        return OCRResult(text="\n".join(self._read_result(result)), source=source_name)

    def read_pdf(self, pdf_path: str | Path) -> OCRResult:
        """Render and read every PDF page using the same OCR engine."""
        import pymupdf

        source = Path(pdf_path).expanduser().resolve()
        output_dir = Path(tempfile.mkdtemp(prefix="tarz_ocr_pdf_"))
        chunks = []
        try:
            with pymupdf.open(source) as document:
                for page_number, page in enumerate(document, start=1):
                    page_path = output_dir / f"page_{page_number}.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False).save(page_path)
                    text = self.read_image(page_path).text.strip()
                    if text:
                        chunks.append(f"[Page {page_number}]\n{text}")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
        return OCRResult(text="\n\n".join(chunks), source=str(source), pages=len(chunks) or 1)

    def extract(self, source: str | Path | bytes, lang=None) -> OCRResult:
        """Backward-compatible alias for existing callers."""
        return self.read_image(source)

    def shutdown(self):
        with self._instance_lock, self._inference_lock:
            self._engine = None
