"""Black-box independent extension driver; no Milvus, model or old test helpers."""
import importlib.util
from pathlib import Path
import sys


def test_independent_driver_real_files_complete_order_and_faults(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location('personal_extensions', tools / 'rag_extensions_acceptance.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.run(tmp_path / 'experiment', 'keyword', True)
    finally:
        sys.modules.pop(spec.name, None)
    assert result['mode'] == 'mock_generation' and result['status'] == 'PASS'
    assert all(result['checks'].values())
    assert [h['text'] for h in result['reversed']['hits']] == [h['text'] for h in reversed(result['baseline']['hits'])]
    assert len(result['baseline']['hits']) == 3
    assert result['core_before'] == result['core_after']
