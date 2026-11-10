import unittest

from docparse.ir import Block, Document, Page


class TestIR(unittest.TestCase):
    def test_serialization(self):
        page = Page(number=1, kind="scanned", mode="page")
        page.blocks.append(Block(id="p1b0", page=1, type="text", content="hello"))
        doc = Document(path="x.pdf", pages=[page])
        data = doc.to_dict()
        self.assertEqual(data["pages"][0]["blocks"][0]["content"], "hello")
        self.assertEqual(data["pages"][0]["blocks"][0]["source"], "vlm")
        self.assertFalse(data["needs_review"])

    def test_high_severity_forces_review(self):
        page = Page(number=1)
        page.warnings.append({"code": "x", "severity": "high", "message": "m"})
        self.assertTrue(Document(path="x", pages=[page]).needs_review)

    def test_medium_does_not_force_review(self):
        page = Page(number=1)
        page.warnings.append({"code": "x", "severity": "medium", "message": "m"})
        self.assertFalse(Document(path="x", pages=[page]).needs_review)


if __name__ == "__main__":
    unittest.main()
