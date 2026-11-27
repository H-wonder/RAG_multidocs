"""Opt-in cached CrossEncoder lifecycle experiment; no model downloads."""
import argparse
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

from rag_multidocs.contracts import RerankRequest, SearchHit
from rag_multidocs.extensions import load_entry
from rag_multidocs.rerankers import RerankerRegistry


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.calls < 3:
        raise ValueError('At least one cold and two hot calls required')
    reg = RerankerRegistry(timeout_s=120)
    info = next(i for i in reg.plugin_infos() if i.name == 'crossencoder')
    query = '本科生借阅期限是多少天？'
    texts = ['天空是蓝色的。', '本科生借阅期限为30天。', '研究生借阅期限60天。', '主馆周一八点开放。', '模型使用交叉验证。', '借阅图书需要校园卡。', 'RAG从知识库获取证据。', '外部链接不会执行。']
    hits = [{'chunk_id': str(i), 'text': text, 'source': f'fixture-{i}'} for i, text in enumerate(texts)]
    request = RerankRequest(query, tuple(SearchHit(h['chunk_id'], h['text'], h['source']) for h in hits), {**info.config, 'local_files_only': True})
    result = {'model': 'BAAI/bge-reranker-base', 'device': 'cpu', 'batch_size': 8, 'max_length': 512, 'calls_per_variant': args.calls, 'query': query, 'candidates': hits, 'variants': {}, 'scope': 'small same-machine repeated experiment; cold direct call excludes interpreter startup, process variants include spawn; peak RSS is whole-process high-water mark'}
    direct = load_entry(info)
    for variant in ['direct', 'one_shot', 'persistent']:
        current = None if variant == 'direct' else RerankerRegistry(timeout_s=120, persistent_paths=[info.path] if variant == 'persistent' else [])
        rows = []
        pid = None
        try:
            for call in range(args.calls):
                started = time.perf_counter()
                if variant == 'direct':
                    ranked = direct(request)
                    row = {'order': [r.candidate_id for r in ranked], 'scores': [r.score for r in ranked], 'diagnostics': direct.diagnostics(), 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                else:
                    ranked, warnings = current.rerank('crossencoder', query, hits)
                    row = {'order': [h['chunk_id'] for h in ranked], 'scores': [h.get('rerank_score') for h in ranked], 'warnings': warnings, 'execution': dict(current.last_execution)}
                    pid = current.last_execution.get('worker_pid')
                    row['pid_exists_after_call'] = alive(pid) if pid else False
                row.update(seconds=time.perf_counter()-started, label='cold_first_call' if call == 0 else 'hot_same_configuration')
                rows.append(row)
                (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        finally:
            if current:
                current.close()
        hot = [r['seconds'] for r in rows[1:]]
        result['variants'][variant] = {'calls': rows, 'cold_seconds': rows[0]['seconds'], 'hot_mean_seconds': statistics.mean(hot), 'hot_p95_seconds': sorted(hot)[math.ceil(.95*len(hot))-1], 'pid_gone_after_close': not alive(pid) if pid else None}
    orders = [r['order'] for v in result['variants'].values() for r in v['calls']]
    result['all_full_orders_identical'] = all(r == orders[0] for r in orders)
    result['status'] = 'PASS' if result['all_full_orders_identical'] and all(not r.get('warnings') for v in result['variants'].values() for r in v['calls']) and result['variants']['persistent']['pid_gone_after_close'] else 'FAIL'
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(result['status'],{k: {'cold':v['cold_seconds'],'hot_mean':v['hot_mean_seconds'],'p95':v['hot_p95_seconds']} for k,v in result['variants'].items()})
    reg.close()


if __name__ == '__main__':
    main()
