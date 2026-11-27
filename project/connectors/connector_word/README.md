# Independent word Connector

Manifest name: connector.word; entry adapter:load, Connector contract v1. Supported input: DOCX, one file or recursive directory. Copy this directory under any explicit ConnectorRegistry plugins_dir and include connector.word in the host enabled allowlist. No core source modification, build, package installation or hot loading is required; restart the caller after installing.

Dependencies: python-docx 1.2.0 in existing parser environment (MIT). No dependencies are installed automatically. Office manifest parser_dependencies describes dependencies in the separate parser_python environment; host requires is intentionally empty. options.parser_python overrides context.settings.docparse_python, then current Python. Image uses options.docparse_python / settings.docparse_python; options.docparse_repo / settings.docparse_repo must point to the existing docparse package root. options.timeout_s limits each file. Workers return MISSING_DEPENDENCY, FILE_NOT_FOUND, UNSUPPORTED_TYPE, PARSE_ERROR/PARSE_TIMEOUT or CONNECTOR_UNAVAILABLE; partial directory success is returned with per-file issues. Registry disablement returns CONNECTOR_DISABLED.

Only public rag_multidocs.contracts/model are imported by the adapter. Parsing runs as ordinary trusted local code, not a security sandbox. No macro, formula, external link or document instruction execution. Input SHA-256 and original logical source are retained. Errors are data, not fake success.

Word page=0 and bbox empty: location identifies paragraph/table/image/archive part and SHA; no physical pagination is invented. Title hierarchy, body/table order and merged cell anchors are retained in segments.meta. Set options.ocr_images=true and docparse_repo to OCR embedded image bytes via existing docparse; otherwise IMAGE_OCR_NOT_REQUESTED and archive location/caption remain. Charts/floating shapes and legacy .doc are not supported.

## Tests

From repository root: `PYTHONDONTWRITEBYTECODE=1 /path/to/rag/python -m pytest -q -p no:cacheprovider project/tests/test_p1_connectors.py`. Office fixtures use the existing parser environment; missing optional environment is explicitly reported. Real model/OCR and Milvus acceptance is separate from mocked OCR tests. See project/docs/RAG_P1_PROGRESS.md and final report for actual evidence. This AI-created plugin is not evidence of another member’s independent authorship or Review.


## P1 reuse revision

The native implementation now lives in the existing public `docparse.office.structured_parse(path, options)` service (`已有资源/可选组件/docparse/docparse/office_structured.py`). Both the old docparse Office entry and this independent plugin use it. `worker.py` is a thin process bridge, not a second parser; configure `docparse_repo` and `docparse_python`/`parser_python`. No core or other plugin internals are imported. Parser service absence is reported, never auto-installed. Existing source/permit notices must be retained; docparse redistribution remains a team decision.

Headers are explicit: Word `table_header_rows={"2":2}` (table number strings); Excel `sheet_header_rows={"Merged":2}`. Common `header_rows` defaults1. The Core repeats the declared real headers when grouping rows; it does not guess damaged cells or silently repair decimal points. Embedded Word OCR reuses the existing Parser/backend; Office still has no physical page/bbox.
