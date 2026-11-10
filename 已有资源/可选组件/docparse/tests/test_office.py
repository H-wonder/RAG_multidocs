import tempfile
import unittest
from pathlib import Path

import docx

from docparse import office


class TestDocx(unittest.TestCase):
    def test_heading_paragraph_and_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "purchase.docx"
            document = docx.Document()
            document.add_heading("采购单", level=1)
            document.add_paragraph("供应商：示例科技有限公司")
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "品名"
            table.cell(0, 1).text = "金额"
            table.cell(1, 0).text = "螺丝"
            table.cell(1, 1).text = "500.00"
            document.save(path)

            parsed = office.parse_docx(path)
            blocks = parsed.pages[0].blocks
            types = [b.type for b in blocks]
            self.assertIn("title", types)
            self.assertIn("text", types)
            self.assertIn("table", types)
            self.assertTrue(any("<td>螺丝</td>" in (b.html or "") for b in blocks))
            self.assertTrue(all(b.source == "native_office" for b in blocks))


if __name__ == "__main__":
    unittest.main()
