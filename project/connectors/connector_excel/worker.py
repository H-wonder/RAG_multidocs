"""Thin bridge to the existing public docparse Office service."""
import json
import sys

if __name__ == "__main__":
    try:
        from docparse.office import structured_parse
        request = json.load(sys.stdin)
        result = structured_parse(request["path"], request["options"])
    except ImportError:
        result = {"error": "MISSING_DEPENDENCY", "message": "docparse 或 openpyxl 无法导入，请检查 DOCPARSE_REPO 和 DOCPARSE_PYTHON 所指向的已有文档环境"}
    except Exception as exc:
        result = {"error": "PARSE_ERROR", "message": type(exc).__name__}
    print(json.dumps(result, ensure_ascii=False))
