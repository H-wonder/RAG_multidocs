"""Separate authored OCR challenge suite on this utility own retained real state."""
import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict,replace
from pathlib import Path

from rag_local import PROJECT,core,settings
from rag_multidocs.evaluation.metrics import ablation_policies,run_suite,write_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ocr-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    original=args.ocr_run.resolve()
    verification=json.loads((original/'result.json').read_text())
    if verification.get('backend')!='milvus' or not verification['checks'].get('changed_actual_ocr_active_version'):
        raise ValueError('Requires the completed isolated OCR real-model driver state')
    database=next((original/'state/catalogs').glob('*.sqlite'))
    connection=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    try:
        identity=json.loads(connection.execute("SELECT value FROM settings WHERE key='index_identity'").fetchone()[0])
    finally:
        connection.close()
    if not identity['collection'].startswith('p1_ocr_'):
        raise ValueError('Refusing unrelated existing knowledge base')
    out.mkdir(parents=True,exist_ok=True)
    config=settings(original,identity['collection'],'milvus')
    service=core(config,ablation_policies()['baseline_rrf'])
    path=PROJECT/'samples/eval/p1_ocr_challenge.json'
    cases=json.loads(path.read_text())['cases']
    manifest={'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dataset':'p1_ocr_challenge.json','source_run':str(original),'collection':identity['collection'],'source_hashes':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').rglob('*.py')},'results':{},'policies':{},'notes':['Seen authored development challenges; PENDING_HUMAN. No cases deleted after results.','Separate from original60. Raw real OCR and version2 image kept in source_run.','No new parsing/ingestion; fixed existing isolated versions for every policy.']}
    try:
        for generate in [False,True]:
            for name in ['baseline_rrf','full']:
                policy=replace(ablation_policies()[name],model_context_tokens=4096,output_tokens=700,request_seconds=120)
                service.controller.policy=policy
                label=('generated_' if generate else 'retrieval_')+name
                manifest['policies'][label]=asdict(policy)
                def checkpoint(row):
                    with (out/(label+'_partial.jsonl')).open('a') as file:
                        file.write(json.dumps(row,ensure_ascii=False)+'\n')
                result=run_suite(service,cases,k=3,generate=generate,on_row=checkpoint)
                write_report(result,out/label)
                manifest['results'][label]=result['summary']
                (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                print(label,result['summary'],flush=True)
    finally:
        service.close()


if __name__=='__main__':
    main()
