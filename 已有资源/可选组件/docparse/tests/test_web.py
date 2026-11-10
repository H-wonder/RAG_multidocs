import unittest

from docparse.web.jobs import JobManager, sanitize_name


class TestSanitizeName(unittest.TestCase):
    def test_path_traversal_is_stripped(self):
        self.assertEqual(sanitize_name("../../etc/passwd"), "passwd")

    def test_cjk_and_spaces_preserved(self):
        self.assertEqual(sanitize_name("发票 2026.pdf"), "发票 2026.pdf")

    def test_empty_falls_back(self):
        self.assertEqual(sanitize_name(""), "file")


class TestOptions(unittest.TestCase):
    def test_defaults(self):
        opts = JobManager._build_options({})
        self.assertEqual(opts.mode, "auto")
        self.assertEqual(opts.dpi, 200)
        self.assertIsNone(opts.kie_fields)

    def test_fields_string(self):
        opts = JobManager._build_options({"mode": "page", "dpi": 150,
                                          "fields": "发票号码, 价税合计"})
        self.assertEqual(opts.mode, "page")
        self.assertEqual(opts.dpi, 150)
        self.assertEqual(opts.kie_fields, ["发票号码", "价税合计"])


if __name__ == "__main__":
    unittest.main()
