"""Opt-in frozen-corpus local experiments, including real-model ablations."""
import argparse
import hashlib
import json
import subprocess
import shutil
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from rag_local import PROJECT, ROOT, PARSER, core, settings, import_corpus
from rag_multidocs.evaluation.metrics import ablation_policies, run_suite, write_report
from rag_multidocs.retrieval.controller import RetrievalPolicy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['milvus','keyword'],default='milvus')
    parser.add_argument('--generate',action='store_true',help='Also generate baseline/full/L3-off answers for all cases')
    parser.add_argument('--policies',nargs='*')
    parser.add_argument('--parser-python',default=PARSER)
    parser.add_argument('--reuse-run',type=Path,help='Read the frozen corpus and isolated state of this tool previous run; never reingest')
    parser.add_argument('--corpus-from',type=Path,help='Copy these frozen binary fixtures for a fresh backend experiment')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    fixtures=out/'corpus'
    previous=None
    if args.reuse_run:
        prior=args.reuse_run.resolve()
        previous=json.loads((prior/'manifest.json').read_text())
        if previous['backend']!=args.backend:
            raise ValueError('Cannot reuse across backends')
        fixtures=prior/'corpus'
        actual={str(p.relative_to(fixtures)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures.rglob('*') if p.is_file()}
        if actual!=previous['corpus_hashes']:
            raise ValueError('Frozen corpus changed')
        config=settings(prior,previous['collection'],args.backend,args.parser_python)
    else:
        if args.corpus_from:
            shutil.copytree(args.corpus_from.resolve(),fixtures)
        else:
            subprocess.run([args.parser_python,str(PROJECT/'tools/rag_samples.py'),str(fixtures)],check=True)
        config=settings(out,'p1_eval_'+uuid.uuid4().hex[:10],args.backend,args.parser_python)
    variants=ablation_policies()
    variants.pop('l3_off')  # L3 requires generated claims, not retrieval-only runs
    if args.policies:
        variants={name:variants[name] for name in args.policies}
    # Fix window/output/deadline across all variants, keep experimental resource
    # allocation free to differ only where that module is being measured.
    variants={name:replace(p,model_context_tokens=4096,output_tokens=700,request_seconds=120) for name,p in variants.items()}
    cases_path=PROJECT/'samples/eval/p1_questions.json'
    cases=json.loads(cases_path.read_text())['cases']
    manifest={'backend':args.backend,'collection':config.milvus_collection,'dataset_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').rglob('*.py')},'policies':{k:asdict(v) for k,v in variants.items()},'notes':['All cases are seen development regressions / PENDING_HUMAN, not independent held-out tests.','Legacy30 use an explicit legacy-source cohort; new30 use documented source scopes.','Real Milvus/BGE/OCR/Ollama and keyword runs must be reported separately.','Historical P0 baseline used normalized reference fixtures; not a strict before/after numerical comparison with new actual parsers.','Retrieval-only full variants may invoke the rewrite LLM, generation_calls stays zero.'],'results':{}}
    service=core(config,RetrievalPolicy.baseline())
    try:
        if previous:
            if previous['dataset_sha256']!=manifest['dataset_sha256']:
                raise ValueError('Dataset changed since frozen run')
            manifest['ingestion']=previous['ingestion']
            manifest['ingestion_mode']='reuse frozen prior isolated catalog/index; no new parsing/embedding ingestion'
            manifest['reuse_run']=str(args.reuse_run.resolve())
            manifest['previous_source_hashes']=previous['source_hashes']
        else:
            manifest['ingestion']=import_corpus(service,fixtures)
        manifest['corpus_hashes']={str(p.relative_to(fixtures)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures.rglob('*') if p.is_file()}
        from rag_multidocs.ingestion.service import canonical_content,digest
        from rag_multidocs.model import Document
        manifest['normalized_document_hashes']={d['source']:digest(canonical_content(Document.from_dict(service.document(d['doc_id'])))) for d in service.list_documents()}
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        for name,policy in variants.items():
            service.controller.policy=policy
            def checkpoint(row):
                with (out/(name+'_partial.jsonl')).open('a') as file:
                    file.write(json.dumps(row,ensure_ascii=False)+'\n')
            report=run_suite(service,cases,k=3,run_label='PENDING_HUMAN_seen_development_'+name,on_row=checkpoint)
            write_report(report,out/name)
            manifest['results'][name]=report['summary']
            (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            print(name,report['summary']['mrr'],report['summary']['gold_evidence_coverage'],flush=True)
        if args.generate:
            for name,policy,level3 in [('generated_baseline',variants.get('baseline_rrf',RetrievalPolicy.baseline()),True),('generated_full',variants.get('full',RetrievalPolicy()),True),('generated_full_l3_off',variants.get('full',RetrievalPolicy()),False)]:
                service.controller.policy=policy
                service.support_level3=level3
                def checkpoint(row):
                    with (out/(name+'_partial.jsonl')).open('a') as file:
                        file.write(json.dumps(row,ensure_ascii=False)+'\n')
                report=run_suite(service,cases,k=3,generate=True,run_label='PENDING_HUMAN_seen_development_'+name,on_row=checkpoint)
                write_report(report,out/name)
                manifest['results'][name]=report['summary']
                (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                print(name,report['summary']['generation_failed_rate'],flush=True)
    finally:
        service.close()
    lines=['# P1 固定语料消融结果','','全部为开发回归；人工答案/引用质量待审核。Keyword与真实Milvus分开。','','| 策略 | Hit@3 | MRR@3 | Gold覆盖 | 估计证据Tokens | 检索均值s | P95s | 重召回率 | Gold救回率 |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,r in manifest['results'].items():
        lines.append(f"| {name} | {r['hit_at_3']} | {r['mrr']} | {r['gold_evidence_coverage']} | {r['context_tokens_mean']} | {r['retrieval_seconds_mean']} | {r['retrieval_seconds_p95']} | {r['second_recall_rate']} | {r['gold_recall_rescue_rate']} |")
    lines+=['','所有目标是待检验目标；未达到不修改gold或删除难题。literal_answer_check不等于Answer Accuracy。','无效重召回/类别资源/调用次数/实际Prompt和generation_evidence见原始JSON。真实成员/独立验证集为PENDING_HUMAN。']
    (out/'ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
