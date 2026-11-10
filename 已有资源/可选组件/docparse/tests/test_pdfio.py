import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from docparse import pdfio


class TestPdfProbe(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _native_pdf(self) -> Path:
        path = self.dir / "native.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Quarterly Report 2026", fontsize=20)
        page.insert_text((72, 110), "Revenue grew 18.4% to CNY 1,234,567.", fontsize=11)
        doc.save(path)
        doc.close()
        return path

    def _scanned_pdf(self) -> Path:
        image = Image.new("RGB", (600, 800), "white")
        ImageDraw.Draw(image).text((50, 50), "SCAN", fill="black")
        image_path = self.dir / "scan.png"
        image.save(image_path)
        path = self.dir / "scanned.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(image_path))
        doc.save(path)
        doc.close()
        return path

    def test_native_page_is_detected(self):
        probe = pdfio.probe_pdf(self._native_pdf())[0]
        self.assertEqual(probe.kind, "native")
        self.assertGreater(probe.chars, 10)

    def test_scanned_page_is_detected(self):
        probe = pdfio.probe_pdf(self._scanned_pdf())[0]
        self.assertEqual(probe.kind, "scanned")

    def test_native_extraction_keeps_text_and_bbox(self):
        path = self._native_pdf()
        with pymupdf.open(path) as doc:
            blocks = pdfio.extract_native(doc[0], 1)
        self.assertTrue(blocks)
        self.assertTrue(all(b.source == "native_text" for b in blocks))
        self.assertTrue(all(len(b.bbox) == 4 for b in blocks))
        self.assertTrue(any(b.type == "title" for b in blocks))
        self.assertIn("1,234,567", " ".join(b.content for b in blocks))

    def test_render(self):
        with pymupdf.open(self._native_pdf()) as doc:
            image = pdfio.render_page(doc[0], 150)
        self.assertGreater(image.width, 100)
        self.assertGreater(image.height, 100)


if __name__ == "__main__":
    unittest.main()
