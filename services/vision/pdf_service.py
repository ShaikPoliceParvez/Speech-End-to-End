"""Render PDF pages to temporary images for OCR and Gemma Vision."""

from pathlib import Path
import tempfile


class PDFService:
    def render(self, source: str | Path) -> list[Path]:
        import pymupdf

        pdf_path = Path(source).expanduser().resolve()
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported PDF source: {pdf_path}")

        output_dir = Path(tempfile.mkdtemp(prefix="tarz_pdf_"))
        pages = []
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)
                page_path = output_dir / f"page_{page_number + 1}.png"
                pixmap.save(page_path)
                pages.append(page_path)
        return pages