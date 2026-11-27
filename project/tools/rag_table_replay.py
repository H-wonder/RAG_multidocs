"""Replay failed mapped-table questions against retained evidence, without writes."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from rag_local import PROJECT, core, settings
from rag_multidocs.retrieval.profiles import recommended_policies
from rag_multidocs.qa.grounding import _numbers, validate_claim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.original.read_text())
    result = {'cases': {}, 'checks': {}, 'model': 'gemma3:4b', 'original': str(args.original.resolve()),
              'scope': 'Same original question and recorded context; generation-only replay, no parsing/retrieval/index writes.'}
    cases = [
        ('excel', '比较LogReg和SVM的Accuracy和Recall', ['90%', '80%', '95%', '85%']),
        ('p0_table', '只根据表格，原始逻辑回归和类别加权逻辑回归的少数类漏判数分别是多少？', ['15', '3']),
    ]
    policy = replace(recommended_policies()['experimental'], max_generation_calls=4, model_context_tokens=4096, output_tokens=700)
    service = core(settings(out, 'unused_table_replay', 'keyword'), policy)
    try:
        for name, question, expected in cases:
            before = original['answers'][name]
            evidence = before['evidence']
            after = service.answer_evidence(question, evidence, style='concise')
            values = _numbers(after['answer'])
            result['cases'][name] = {'question': question, 'before': before, 'after': after,
                                    'evidence_sha256': hashlib.sha256(json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            result['checks'][name + '_values'] = after['status'] == 'answered' and all(_numbers(v) <= values for v in expected)
            result['checks'][name + '_quotes'] = bool(after.get('claims')) and all(validate_claim(c, after.get('answer_evidence', after['generation_evidence'])) for c in after['claims'])
            result['checks'][name + '_call_budget'] = after['model_calls'] <= 4
            if name == 'p0_table':
                result['checks']['original_pdf_location'] = bool(after['citations']) and all(c['page'] == 7 and c['bbox'] for c in after['citations'])
            print(name, after['status'], after['answer'], flush=True)
            (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        service.close()
    result['source_hashes'] = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / 'src').rglob('*.py')}
    result['status'] = 'PASS' if all(result['checks'].values()) else 'PARTIAL'
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result['status'], result['checks'])


if __name__ == '__main__':
    main()
