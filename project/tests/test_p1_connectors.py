import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.contracts import ConnectorConfig
from rag_multidocs.core import RAGCore
from rag_multidocs.index import KeywordIndex

PROJECT = Path(__file__).resolve().parents[1]
PARSER = Path("/home/genevievelin/anaconda3/envs/docparse/bin/python")


@pytest.fixture
def files(tmp_path):
    if not PARSER.exists():
        pytest.skip("Optional existing document environment unavailable")
    subprocess.run([str(PARSER), str(PROJECT / "tests/p1_fixtures.py"), str(tmp_path)], check=True)
    return tmp_path


def registry(enabled=None):
    return ConnectorRegistry(PROJECT / "connectors", enabled=enabled, context={"settings": {"docparse_python": str(PARSER), "docparse_repo": str(PROJECT.parent/'已有资源/可选组件/docparse')}})


def core_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / "src").rglob("*.py")}


def test_word_native_order_table_image_location_and_independent_core(files):
    before = core_hashes()
    result = registry().load("connector.word", ConnectorConfig("connector.word", str(files / "handbook.docx")))
    assert not result.issues and len(result.documents) == 1
    doc = result.documents[0]
    blocks = doc.segments
    first_table = next(i for i,b in enumerate(blocks) if b["type"] == "table")
    assert "24 months" in blocks[first_table-1]["content"]
    assert "85% | 92%" in blocks[first_table]["content"]
    assert blocks[first_table+1]["content"] == "After the comparison table."
    figure = next(b for b in blocks if b["type"] == "figure")
    assert figure["caption"] == "Figure 1: purchase receipt."
    assert figure["meta"]["location"]["archive_part"].startswith("/word/media/")
    assert all(b["page"] == 0 and b["bbox"] == [] for b in blocks)
    assert blocks[-1]["meta"]["table_data"]["cells"][1]["anchor"] == blocks[-1]["meta"]["table_data"]["cells"][2]["anchor"]
    core = RAGCore(registry(), KeywordIndex(path=files / "index.json"))
    try:
        assert core.ingest("connector.word", str(files / "handbook.docx")).ok
        hits = core.search("warranty period")['hits']
        assert any("24 months" in h['text'] and h['source']=="handbook.docx" for h in hits)
    finally:
        core.close()
    assert core_hashes() == before


def test_excel_sheets_types_formula_and_input_change(files):
    result = registry().load("connector.excel", ConnectorConfig("connector.excel", str(files / "measurements.xlsx")))
    assert not result.issues
    doc = result.documents[0]
    assert len(doc.segments) == 3
    model = doc.segments[0]
    assert "LogReg | 90.00% | 80.00%" in model["content"]
    assert "SVM | 95.00% | 85.00%" in model["content"]
    assert model["meta"]["location"]["cell_range"] == "A1:C3"
    budget = doc.segments[1]
    cell = next(c for c in budget['meta']['table_data']['cells'] if c['address']=='F2')
    assert cell['formula']=='=B2+B3' and cell['cached_value'] is None
    assert any(w['code']=='FORMULA_UNCACHED' for w in doc.metadata['warnings'])
    assert all(b['page']==0 and not b['bbox'] for b in doc.segments)
    subprocess.run([str(PARSER), '-c', 'import openpyxl,sys; w=openpyxl.load_workbook(sys.argv[1]); w["Models"]["B2"]=0.77; w.save(sys.argv[1])', str(files/'measurements.xlsx')], check=True)
    changed = registry().load('connector.excel',ConnectorConfig('connector.excel',str(files/'measurements.xlsx'))).documents[0]
    assert '77.00%' in changed.segments[0]['content']
    assert changed.metadata['file_sha256'] != doc.metadata['file_sha256']


@pytest.mark.parametrize('name,ext',[('word','docx'),('excel','xlsx'),('image','png')])
def test_plugins_disabled_missing_corrupt_and_existing_connector(name,ext,tmp_path):
    file=tmp_path/f'bad.{ext}'
    file.write_bytes(b'not a valid document')
    key='connector.'+name
    assert registry(set()).load(key,ConnectorConfig(key,str(file))).issues[0].code=='CONNECTOR_DISABLED'
    assert registry().load(key,ConnectorConfig(key,str(tmp_path/'absent'))).issues[0].code=='FILE_NOT_FOUND'
    assert registry().load(key,ConnectorConfig(key,str(file))).issues[0].code=='PARSE_ERROR'
    assert registry().load('connector.markdown',ConnectorConfig('connector.markdown',str(PROJECT/'samples/campus-corpus/markdown'))).documents


@pytest.mark.parametrize('name,ext',[('word','docx'),('excel','xlsx')])
def test_worker_missing_dependency_is_explicit(files,name,ext):
    file=files/('handbook.docx' if ext=='docx' else 'measurements.xlsx')
    result=registry().load('connector.'+name,ConnectorConfig('connector.'+name,str(file),{'parser_python':sys.executable}))
    assert result.issues[0].code=='MISSING_DEPENDENCY'


def test_image_corrupt_batch_does_not_abort_other_files(files,monkeypatch):
    from rag_multidocs.extensions import load_entry
    reg=registry()
    info=next(i for i in reg.plugin_infos() if i.name=='connector.image')
    load=load_entry(info)
    adapter=sys.modules[load.__module__]
    (files/'bad.png').write_bytes(b'broken')
    def run(command,**kwargs):
        out=Path(command[command.index('-o')+1])
        (out/'image.json').write_text(json.dumps({'pages':[{'blocks':[{'type':'text','content':'Mock OCR amount 128.50','bbox':[0,0,1000,1000]}]}]}))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(adapter.subprocess,'run',run)
    result=load({'path':str(files),'options':{'docparse_repo':str(PROJECT.parent/'已有资源/可选组件/docparse')}})
    assert sorted(d.source for d in result.documents)==['receipt.jpg','receipt.png']
    assert result.issues[0].source=='bad.png'
    assert all(not b['bbox'] for d in result.documents for b in d.segments)


def test_existing_excel_formula_cache_is_read_without_evaluation(files):
    import zipfile
    import xml.etree.ElementTree as ET
    path=files/'measurements.xlsx'
    namespace={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    # A producing spreadsheet application writes cached values into OOXML.
    # Author that source cache explicitly here; the parser must only read it.
    with zipfile.ZipFile(path) as source:
        entries={name:source.read(name) for name in source.namelist()}
    sheet=ET.fromstring(entries['xl/worksheets/sheet2.xml'])
    cell=sheet.find('.//s:c[@r="F2"]',namespace)
    assert cell.find('s:f',namespace).text=='B2+B3'
    cell.find('s:v',namespace).text='1220.5'
    entries['xl/worksheets/sheet2.xml']=ET.tostring(sheet)
    with zipfile.ZipFile(path,'w') as target:
        for name,data in entries.items():
            target.writestr(name,data)
    result=registry().load('connector.excel',ConnectorConfig('connector.excel',str(path)))
    assert not result.issues
    formula=next(c for b in result.documents[0].segments for c in b['meta']['table_data']['cells'] if c['address']=='F2')
    assert formula['formula']=='=B2+B3' and formula['cached_value']==1220.5
    assert 'cached=1220.5' in formula['display']
    assert not any(w['code']=='FORMULA_UNCACHED' for w in result.documents[0].metadata['warnings'])
