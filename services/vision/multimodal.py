"""Coordinate image/PDF preparation and OCR without owning LLM or TTS logic."""

from dataclasses import dataclass
from pathlib import Path

from services.ocr import OCRResult, PaddleOCRService
from services.vision.image_service import SUPPORTED_IMAGE_EXTENSIONS, prepare_image
from services.vision.pdf_service import PDFService


@dataclass(frozen=True)
class MultimodalInput:
    image: object
    ocr_text: str = ""
    source: str = ""
    pages: int = 1


class VisionService:
    def __init__(self, ocr_service=None, pdf_service=None):
        self.ocr = ocr_service or PaddleOCRService()
        self.pdf = pdf_service or PDFService()

    def prepare(self, source: str | Path, use_ocr: bool = False, ocr_lang: str = "en") -> MultimodalInput:
        if isinstance(source, bytes):
            image = prepare_image(source)
            result = self.ocr.extract(source, lang=ocr_lang) if use_ocr else OCRResult("", "<camera>")
            return MultimodalInput(image=image, ocr_text=result.text, source="<camera>")

        path = Path(source).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self.pdf.render(path)
            ocr_text = self.ocr.read_pdf(path).text if use_ocr else ""
            return MultimodalInput(
                image=prepare_image(pages[0]),
                ocr_text=ocr_text,
                source=str(path),
                pages=len(pages),
            )
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported media type: {suffix or path.name}")
        image = prepare_image(path)
        result = self.ocr.extract(path, lang=ocr_lang) if use_ocr else OCRResult("", str(path))
        return MultimodalInput(image=image, ocr_text=result.text, source=str(path))

    def prepare_many(
        self,
        sources: list[str | Path],
        use_ocr: bool = False,
        ocr_lang: str = "en",
    ) -> list[MultimodalInput]:
        return [self.prepare(source, use_ocr=use_ocr, ocr_lang=ocr_lang) for source in sources]

    def _ocr_pages(self, pages: list[Path], ocr_lang: str) -> str:
        chunks = []
        for index, page in enumerate(pages, start=1):
            text = self.ocr.extract(page, lang=ocr_lang).text.strip()
            if text:
                chunks.append(f"[Page {index}]\n{text}")
        return "\n\n".join(chunks)