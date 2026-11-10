import unittest

from docparse.evaluate import compare, normalize
from docparse.quality import assess, ground_fields, parse_kie


class TestGrounding(unittest.TestCase):
    def test_numeric_field_found(self):
        out = ground_fields({"总金额": "RM 5.90"}, "Total RM 5.90")
        self.assertTrue(out["总金额"]["ok"])

    def test_date_normalization_is_accepted(self):
        out = ground_fields({"日期": "2017-10-18"}, "18-10-17 15:16")
        self.assertTrue(out["日期"]["ok"])
        self.assertEqual(out["日期"]["reason"], "date_components_present")

    def test_hallucinated_amount_is_rejected(self):
        out = ground_fields({"总金额": "999.00"}, "Total RM 5.90")
        self.assertFalse(out["总金额"]["ok"])

    def test_hallucinated_date_is_rejected(self):
        out = ground_fields({"日期": "2024-05-06"}, "18-10-17 15:16")
        self.assertFalse(out["日期"]["ok"])

    def test_empty_field(self):
        out = ground_fields({"发票号码": ""}, "anything")
        self.assertEqual(out["发票号码"]["reason"], "empty")


class TestAssess(unittest.TestCase):
    def test_clean_output(self):
        result = assess("Quarterly revenue grew 18.4%.", "stop")
        self.assertFalse(result["suspicious"])
        self.assertEqual(result["score"], 1.0)
        self.assertFalse(result["needs_retry"])

    def test_repetition_detected(self):
        result = assess("abc" * 300, "stop")
        self.assertTrue(result["suspicious"])
        self.assertTrue(result["needs_retry"])

    def test_truncation_detected(self):
        result = assess("some partial text", "length")
        self.assertTrue(result["needs_retry"])
        self.assertTrue(any(w["code"] == "truncated" for w in result["warnings"]))

    def test_empty_detected(self):
        result = assess("", "stop")
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["score"], 0.0)


class TestKieParsing(unittest.TestCase):
    def test_extracts_json(self):
        data, error = parse_kie('here: {"a": "1"}', ["a"])
        self.assertEqual(data, {"a": "1"})
        self.assertEqual(error, "")

    def test_invalid_json(self):
        data, error = parse_kie("no object here")
        self.assertIsNone(data)
        self.assertEqual(error, "no_json_object")


class TestCompare(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(compare("你好 世界", "你好 世界")["similarity"], 1.0)

    def test_normalization_strips_markup(self):
        self.assertEqual(normalize("<table><tr><td>a</td></tr></table>"), "a")


if __name__ == "__main__":
    unittest.main()
