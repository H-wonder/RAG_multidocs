"""Replay retained real generation evidence without re-OCR or database writes."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from rag_local import core,settings,PROJECT
from rag_multidocs.evaluation.metrics import ablation_policies


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',default='multi_answer')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output')
    out.mkdir(parents=True,exist_ok=True)
    before=json.loads(args.original.read_text())['answers'][args.case]
    question=before['attempts'][0]['messages'][1]['content']
    question=json.loads(question)['question']
    evidence=before['generation_evidence']
    service=core(settings(out,'unused_replay','keyword'),replace(ablation_policies()['full'],model_context_tokens=4096,output_tokens=700))
    try:
        after=service.answer_evidence(question,evidence,style='concise')
    finally:
        service.close()
    result={'question':question,'original':str(args.original.resolve()),'original_status':before['status'],'evidence_sha256':hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'source_hashes':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').rglob('*.py')},'after':after,'mode':'real gemma3:4b generation replay on recorded real OCR evidence; no retrieval or parsing rerun'}
    result['status']='PASS' if after['status']=='answered' and '36' in after['answer'] and any('36 months' in q for c in after['citations'] for q in c['excerpts']) else 'PARTIAL'
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(result['status'],after['status'],after['answer'])


if __name__=='__main__':
    main()
