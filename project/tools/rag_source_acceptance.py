"""Public-API source identity/ambiguity regression in a fresh isolated collection."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

from rag_local import PROJECT, core, settings
from rag_multidocs.retrieval.profiles import recommended_policies
from rag_multidocs.qa.grounding import _numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    collection = 'p1_sources_'+uuid.uuid4().hex[:10]
    service = core(settings(out, collection, 'milvus'), recommended_policies()['stable'])
    result = {'checks': {}, 'answers': {}, 'collection': collection, 'ingestion': []}
    def save():
        (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    def ask(key, question):
        result['answers'][key] = service.ask(question, style='concise')
        save()
        return result['answers'][key]
    try:
        for name, content in [('money_a.txt', 'Amount: 17 CNY.\nWarranty period: 12 months.'),
                              ('money_b.txt', 'Amount: 31 CNY.\nWarranty period: 48 months.')]:
            path = out/name
            path.write_text(content)
            result['ingestion'].append(service.ingest('txt', str(path)).as_dict())
        # Existing source_namespace distinguishes logical identities; no new
        # identity subsystem and no accidental overwrite of same-basename files.
        for namespace, value in [('left', '43'), ('right', '57')]:
            path = out/namespace/'same.txt'
            path.parent.mkdir()
            path.write_text('Amount: '+value+' CNY.')
            result['ingestion'].append(service.ingest('txt', str(path), {'source_namespace': namespace}).as_dict())
        amount = ask('amount', 'money_a.txt中的金额是多少？')
        result['checks']['unique_amount'] = amount['status'] == 'answered' and '17cny' in _numbers(amount['answer']) and {c['source'] for c in amount['citations']} == {'money_a.txt'}
        comparison = ask('comparison', '比较money_a.txt和money_b.txt的保修期分别是多少？')
        result['checks']['two_source_values'] = {'12month', '48month'} <= _numbers(comparison['answer']) and comparison['status'] == 'answered'
        result['checks']['two_source_citations'] = {c['source'] for c in comparison['citations']} == {'money_a.txt', 'money_b.txt'}
        ambiguous = ask('ambiguous', 'same.txt中的金额是多少？')
        result['checks']['ambiguous_closed'] = ambiguous['status'] == 'insufficient_evidence' and ambiguous['index_calls'] == 0 and any(w['code'] == 'SOURCE_AMBIGUOUS' for w in ambiguous['warnings'])
        exact = ask('exact_path', 'left/same.txt中的金额是多少？')
        result['checks']['exact_namespace'] = exact['status'] == 'answered' and '43cny' in _numbers(exact['answer']) and all(c['source'] == 'left/same.txt' for c in exact['citations'])
        absent = ask('absent', '"absent.txt"中的金额是多少？')
        result['checks']['absent_closed'] = absent['status'] == 'insufficient_evidence' and absent['index_calls'] == 0 and not absent['citations']
        result['ordinary_search'] = service.search('Warranty period')
        result['checks']['ordinary_not_scoped'] = result['ordinary_search']['source_constraints']['explicit'] is False and bool(result['ordinary_search']['hits'])
        (out/'money_a.txt').write_text('Amount: 29 CNY.\nWarranty period: 18 months.')
        result['update'] = service.ingest('txt', str(out/'money_a.txt')).as_dict()
        updated = ask('updated', 'money_a.txt中的金额是多少？')
        result['checks']['active_version'] = updated['status'] == 'answered' and '29cny' in _numbers(updated['answer']) and bool(updated['citations']) and all(c['version'] == 2 for c in updated['citations'])
        service.controller.policy = recommended_policies()['experimental']
        result['recall'] = service.search('money_a.txt中工资和风险信息是什么？')
        allowed = result['recall']['source_constraints']['doc_ids']
        result['checks']['recall_really_exercised'] = len(result['recall']['retrieval_trace']) == 2
        result['checks']['all_recall_routes_scoped'] = bool(allowed) and all(t['source_doc_ids'] == allowed and all(h['doc_id'] in allowed for h in t['candidates']) for t in result['recall']['retrieval_trace'])
        result['checks']['actual_model_prompts_scoped'] = bool(comparison['attempts']) and all(
            len({h['source'] for h in json.loads(a['messages'][1]['content'])['evidence']}) == 1
            for a in comparison['attempts'])
        result['source_hashes'] = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').rglob('*.py')}
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)[:300]
        raise
    finally:
        service.close()
        result['status'] = 'PASS' if result['checks'] and all(result['checks'].values()) and not result.get('error') else 'PARTIAL'
        save()
    print(result['status'], result['checks'])


if __name__ == '__main__':
    main()
