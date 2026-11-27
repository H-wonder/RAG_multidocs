import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_native_bbox_on_cropped_rotated_page_with_table(self):
        path = self.dir / "rotated.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page(width=450, height=570)
            page.set_cropbox(pymupdf.Rect(22, 28, 422, 528))
            page.insert_text((20, 45), "Materials analysis and optical calibration.", fontsize=11)
            page.insert_text((20, 75), "Visible margin text", fontsize=10)
            page.insert_text((399, 75), " ", fontsize=10)
            for y in (110, 134, 158, 182, 206):
                page.draw_line((25, y), (365, y))
            for x in (25, 205, 365):
                page.draw_line((x, 110), (x, 206))
            # The two right columns share the first header cell.
            page.draw_line((285, 134), (285, 206))
            for x, y, text in [
                (34, 128, "Module"), (214, 128, "Credit allocation"),
                (34, 151, "Biomedical optics"), (214, 151, "Core"), (294, 151, "6"),
                (34, 176, "Signal analysis"), (214, 176, "Elective"), (294, 176, "3"),
                (34, 198, "Precision mechanics"), (214, 198, "Core"), (294, 198, "4"),
            ]:
                page.insert_text((x, y), text, fontsize=10)
            page.insert_text((388, 240), "MM", fontsize=8)
            page.set_rotation(90)
            doc.save(path)
        with pymupdf.open(path) as doc:
            page = doc[0]
            by_text = {b.content: b for b in pdfio.extract_native(page, 1)}
            self.assertEqual(page.rotation, 90)
            self.assertEqual(page.rect, pymupdf.Rect(0, 0, 500, 400))
        self.assertEqual(by_text["Materials analysis and optical calibration."].page, 1)
        margin = by_text["Visible margin text"]
        self.assertEqual(margin.warnings[0]["code"], "native_bbox_from_text_spans")
        self.assertGreater(margin.warnings[0]["raw_bbox_pt"][2], 400)
        self.assertEqual(margin.bbox, [844.0, 50.0, 871.5, 254.2])
        edge = by_text["MM"]
        self.assertEqual(edge.bbox[-1], 1000.0)
        self.assertEqual(edge.warnings[0]["code"], "native_bbox_clipped")
        table = next(b for b in by_text.values() if b.type == "table")
        self.assertEqual(table.bbox, [588.0, 62.5, 780.0, 912.5])
        self.assertIn(
            "<td>Biomedicaloptics</td><td>Core</td><td>6</td>",
            table.content.replace(" ", ""),
        )
        self.assertEqual(table.warnings, [])
        for block in by_text.values():
            x0, y0, x1, y1 = block.bbox
            self.assertTrue(0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000)

    def test_unrecoverable_boxes_retain_content_with_explicit_warning(self):
        bounds = pymupdf.Rect(0, 0, 400, 500)
        for raw in [None, (15, 20, 435, 40), (-4, 5, -2, 10),
                    (30, 40, 10, 50), (0, 0, float("nan"), 0)]:
            with self.subTest(raw=raw):
                span = {"text": "A searchable passage.", "size": 10, "bbox": raw}
                block = {"type": 0, "bbox": raw, "lines": [{"spans": [span]}]}
                page = SimpleNamespace(
                    rect=bounds, rotation_matrix=pymupdf.Matrix(1, 1),
                    derotation_matrix=pymupdf.Matrix(1, 1),
                    get_text=lambda mode: {"blocks": [block]},
                    get_image_info=list,
                )
                result, = pdfio.extract_native(page, 8)
                self.assertEqual(result.content, span["text"])
                self.assertEqual(result.bbox, [])
                self.assertEqual(result.page, 8)
                self.assertEqual(result.warnings[0]["code"], "native_bbox_unavailable")

    def test_tiny_negative_float_error_clips_without_losing_valid_region(self):
        _, bbox, warnings = pdfio._native_bbox(
            (-1e-7, 20, 40, 40), pymupdf.Rect(0, 0, 400, 500), pymupdf.Matrix(1, 1),
        )
        self.assertEqual(bbox, [0.0, 40.0, 100.0, 80.0])
        self.assertEqual(warnings[0]["code"], "native_bbox_clipped")

    def test_table_enclosing_cell_preserves_rows_and_real_vertical_merge(self):
        # A real full-height label has adjacent cells, while the spurious
        # enclosing cell overlaps them. A true merged header is horizontal.
        table = SimpleNamespace(
            bbox=(0, 0, 300, 90),
            rows=[
                SimpleNamespace(cells=[(0, 0, 300, 90), (0, 0, 20, 90), (20, 0, 300, 30)]),
                SimpleNamespace(cells=[None, None, (20, 30, 300, 60)]),
                SimpleNamespace(cells=[None, None, (20, 60, 300, 90)]),
            ],
        )
        rows = [["duplicate", "Term A", "Class"], [None, None, "Lab"], [None, None, "Seminar"]]
        filtered, coverage, warnings = pdfio._table_rows(table, rows)
        self.assertEqual(filtered, [[None, "Term A", "Class"], rows[1], rows[2]])
        self.assertIn(pymupdf.Rect(0, 0, 20, 90), coverage)
        self.assertNotIn(pymupdf.Rect(table.bbox), coverage)
        self.assertEqual(warnings[0]["code"], "native_table_enclosing_cell")

    def test_render(self):
        with pymupdf.open(self._native_pdf()) as doc:
            image = pdfio.render_page(doc[0], 150)
        self.assertGreater(image.width, 100)
        self.assertGreater(image.height, 100)


if __name__ == "__main__":
    unittest.main()
