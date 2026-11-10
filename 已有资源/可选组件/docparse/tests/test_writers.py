import tempfile
import unittest
from pathlib import Path

from docparse import writers
from docparse.ir import Block, Document, Page


def make_doc() -> Document:
    page = Page(number=1, kind="scanned", mode="region")
    page.blocks = [
        Block(id="p1b0", page=1, type="title", content="Quarterly Report", reading_order=0),
        Block(id="p1b1", page=1, type="table",
              content="<table><tr><td>A</td><td>1</td></tr></table>",
              html="<table><tr><td>A</td><td>1</td></tr></table>", reading_order=1),
        Block(id="p1b2", page=1, type="formula", content="$$E=mc^2$$", reading_order=2),
    ]
    return Document(path="sample.pdf", pages=[page])


class TestWriters(unittest.TestCase):
    def test_markdown_keeps_html_table(self):
        markdown = writers.document_markdown(make_doc())
        self.assertIn("<table>", markdown)
        self.assertIn("Quarterly Report", markdown)

    def test_text_strips_tags_and_keeps_cells(self):
        text = writers.document_text(make_doc())
        self.assertIn("Quarterly Report", text)
        self.assertIn("A\t1", text)
        self.assertNotIn("<td>", text)

    def test_html_is_standalone(self):
        html = writers.document_html(make_doc())
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("<table>", html)

    def test_write_all_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = writers.write(make_doc(), Path(tmp), formats=["md", "json", "html", "txt"])
            self.assertEqual(set(paths), {"md", "json", "html", "txt"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                writers.write(make_doc(), Path(tmp), formats=["pdf"])


if __name__ == "__main__":
    unittest.main()
