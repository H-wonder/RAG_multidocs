import json
import os

import pytest

from rag_multidocs.rerankers import RerankerRegistry


@pytest.fixture
def plugin(tmp_path):
    p=tmp_path/'independent'
    p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'name':'independent','kind':'reranker','entry':'adapter:rank','version':'1','contract_version':'1'}))
    (p/'adapter.py').write_text('''
import time
from rag_multidocs.contracts import RankedCandidate
calls=0
def rank(request):
    global calls
    calls+=1
    if request.query=='slow': time.sleep(5)
    if request.query=='crash': raise RuntimeError('failure')
    if request.query=='bad': return [RankedCandidate('alien',1,1.0)]
    return [RankedCandidate(h.candidate_id,i+1,1-i/10) for i,h in enumerate(reversed(request.candidates))]
rank.diagnostics=lambda:{'calls':calls,'fixture_initializations':1}
''')
    return p


def hits():
    return [{'chunk_id':'a','text':'original A','source':'first'},{'chunk_id':'b','text':'original B','source':'second'}]


def gone(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid,0)


def test_persistent_reuses_real_process_maps_ids_and_releases(plugin):
    reg=RerankerRegistry(plugin.parent,persistent_paths=[plugin],timeout_s=2)
    baseline=hits()
    try:
        first,warnings=reg.rerank('independent','q',baseline)
        pid=reg.last_execution['worker_pid']
        os.kill(pid,0)
        assert not warnings and [h['chunk_id'] for h in first]==['b','a']
        assert [(h['text'],h['source']) for h in first]==[('original B','second'),('original A','first')]
        second,warnings=reg.rerank('independent','q',baseline)
        assert not warnings and reg.last_execution['worker_pid']==pid
        assert reg.last_execution['diagnostics']['calls']==2
        assert reg.last_execution['request_id']
        assert second[0]['chunk_id']=='b' and baseline==hits()
    finally:
        reg.close()
    gone(pid)


@pytest.mark.parametrize('failure,code',[('slow','RERANKER_TIMEOUT'),('crash','RERANKER_EXECUTION_ERROR'),('bad','RERANKER_INVALID_OUTPUT')])
def test_timeout_exception_and_illegal_output_reap_then_limited_restart(plugin,failure,code):
    reg=RerankerRegistry(plugin.parent,persistent_paths=[plugin],timeout_s=2,max_worker_restarts=1)
    baseline=hits()
    try:
        reg.rerank('independent','q',baseline)
        pid=reg.last_execution['worker_pid']
        ranked,warnings=reg.rerank('independent',failure,baseline,timeout_s=0.1 if failure=='slow' else 1)
        assert ranked==baseline and warnings[0]['code']==code
        gone(pid)
        ranked,warnings=reg.rerank('independent','q',baseline)
        assert not warnings and reg.last_execution['worker_pid']!=pid
        next_pid=reg.last_execution['worker_pid']
        reg.rerank('independent','crash',baseline)
        gone(next_pid)
        ranked,warnings=reg.rerank('independent','q',baseline)
        assert ranked==baseline and warnings[0]['fallback']=='rrf'
        assert reg.persistent_worker.starts==2
    finally:
        reg.close()


def test_unapproved_plugin_uses_old_one_shot_and_disabled_falls_back(plugin):
    reg=RerankerRegistry(plugin.parent,timeout_s=2)
    baseline=hits()
    result,issues=reg.rerank('independent','q',baseline)
    assert not issues and [h['chunk_id'] for h in result]==['b','a']
    assert reg.last_execution['mode']=='process' and not reg.last_execution['worker_alive']
    reg.enabled=set()
    assert reg.rerank('independent','q',baseline)[0]==baseline
    reg.close()
