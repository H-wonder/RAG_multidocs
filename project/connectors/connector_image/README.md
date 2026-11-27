# Independent image Connector

Manifest name: connector.image; entry adapter:load, Connector contract v1. Supported input: PNG/JPG/JPEG, one file or recursive directory. Copy this directory under any explicit ConnectorRegistry plugins_dir and include connector.image in the host enabled allowlist. No core source modification, build, package installation or hot loading is required; restart the caller after installing.

Dependencies: Pillow on host; existing docparse/Ollama OCR in parser environment. No dependencies are installed automatically. Office manifest parser_dependencies describes dependencies in the separate parser_python environment; host requires is intentionally empty. options.parser_python overrides context.settings.docparse_python, then current Python. Image uses options.docparse_python / settings.docparse_python; options.docparse_repo / settings.docparse_repo must point to the existing docparse package root. options.timeout_s limits each file. Workers return MISSING_DEPENDENCY, FILE_NOT_FOUND, UNSUPPORTED_TYPE, PARSE_ERROR/PARSE_TIMEOUT or CONNECTOR_UNAVAILABLE; partial directory success is returned with per-file issues. Registry disablement returns CONNECTOR_DISABLED.

Only public rag_multidocs.contracts/model are imported by the adapter. Parsing runs as ordinary trusted local code, not a security sandbox. No macro, formula, external link or document instruction execution. Input SHA-256 and original logical source are retained. Errors are data, not fake success.

Page-mode OCR reuses the public docparse CLI and configured cached OCR model. It does not provide localized word boxes: bbox is empty with IMAGE_NO_BBOX, and no visual embedding or complete scene understanding is claimed. Corrupt files do not abort other images.

## Tests

From repository root: `PYTHONDONTWRITEBYTECODE=1 /path/to/rag/python -m pytest -q -p no:cacheprovider project/tests/test_p1_connectors.py`. Office fixtures use the existing parser environment; missing optional environment is explicitly reported. Real model/OCR and Milvus acceptance is separate from mocked OCR tests. See project/docs/RAG_P1_PROGRESS.md and final report for actual evidence. This AI-created plugin is not evidence of another member’s independent authorship or Review.
