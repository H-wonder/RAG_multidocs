"""Final-source evaluation in NEW collections cloned from frozen real parser IR.

Old catalogs are read-only. Public Catalog.publish re-embeds the exact saved
active chunks into new isolated indexes. No OCR output, question or gold edited.
"""
import argparse
import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from rag_local import PROJECT, ROOT, core, settings
from rag_multidocs.evaluation.metrics import ablation_policies, run_suite, write_report
from rag_multidocs.ingestion.service import canonical_content, digest
from rag_multidocs.model import Document, Chunk
from rag_multidocs.retrieval.profiles import recommended_policies


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [PROJECT/'src', PROJECT/'connectors', PROJECT/'rerankers',
                           ROOT/'已有资源/可选组件/docparse/docparse']
            for p in folder.rglob('*.py') if '__pycache__' not in p.parts}


def clone_frozen(original, service):
    databases = list((original/'state/catalogs').glob('*.sqlite'))
    if len(databases) != 1:
        raise ValueError('Expected this utility one isolated source catalog')
    con = sqlite3.connect(databases[0].as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    clones = []
    try:
        for row in con.execute('SELECT * FROM documents WHERE deleted=0 ORDER BY doc_id').fetchall():
            document = Document.from_dict(json.loads(row['payload']))
            old_chunks = [json.loads(r['payload']) for r in con.execute(
                'SELECT payload FROM chunks WHERE doc_id=? AND version=? ORDER BY position',
                (row['doc_id'], row['active_version'])).fetchall()]
            chunks = [Chunk.from_dict(c) for c in old_chunks]
            if not chunks:
                raise ValueError('Empty frozen active document')
            count, version = service.ingestion.catalog.publish(document, chunks, 'frozen-clone:'+row['fingerprint'])
            original_hash = digest(canonical_content(document))
            actual_hash = digest(canonical_content(Document.from_dict(service.document(document.doc_id))))
            if actual_hash != original_hash or count != len(chunks):
                raise AssertionError('Frozen content changed')
            fields = ['text', 'source', 'index', 'page', 'bbox', 'metadata']
            actual_chunks = [json.loads(r['payload']) for r in service.ingestion.catalog.db.execute(
                'SELECT payload FROM chunks WHERE doc_id=? AND version=? ORDER BY position', (document.doc_id, version))]
            before_values = [{k: c.get(k) for k in fields} for c in old_chunks]
            after_values = [{k: c.get(k) for k in fields} for c in actual_chunks]
            if before_values != after_values:
                raise AssertionError('Cloned chunk text/location/metadata changed')
            clones.append({'doc_id': document.doc_id, 'source': document.source, 'chunks': count,
                           'original_active_version': row['active_version'], 'isolated_version': version,
                           'normalized_document_hash': actual_hash,
                           'raw_sha256': document.metadata.get('raw_sha256'),
                           'chunk_content_and_metadata_equal': True,
                           'frozen_chunk_contents_sha256': hashlib.sha256(json.dumps([
                               {k: c.get(k) for k in ['text','source','index','page','bbox','metadata']}
                               for c in old_chunks], ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
            print('cloned', document.source, count, 'chunks', flush=True)
    finally:
        con.close()
    return clones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['main', 'challenge'], default='main')
    parser.add_argument('--pilot', action='store_true', help='Only c02,c04,c11 on new isolated challenge state')
    parser.add_argument('--generation-only', action='store_true')
    parser.add_argument('--reuse-final-run', type=Path, help='Read-only reuse of this round own validated p1_final index; copy catalog to the new data directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new evidence directory')
    out.mkdir(parents=True, exist_ok=True)
    dataset = PROJECT/'samples/eval'/('p1_questions.json' if args.dataset == 'main' else 'p1_ocr_challenge.json')
    cases = json.loads(dataset.read_text())['cases']
    if args.pilot:
        if args.dataset != 'challenge':
            raise ValueError('Pilot only for challenge')
        cases = [c for c in cases if c['id'] in {'c02', 'c04', 'c11'}]
    presets = recommended_policies()
    variants = {'baseline_rrf': presets['stable'], 'crossencoder': presets['quality'], 'full': presets['experimental']}
    if args.dataset == 'main' and not args.pilot and not args.generation_only:
        # Re-run only affected comparisons; all other old16 remain historical.
        variants.update({name: replace(ablation_policies()[name], atomic_generation=True, structured_table_answers=True)
                         for name in ['full_no_rewrite', 'full_no_mmr', 'full_no_coverage', 'full_no_recall']})
    variants = {name: replace(policy, model_context_tokens=4096, output_tokens=700, request_seconds=120)
                for name, policy in variants.items()}
    collection = 'p1_final_'+uuid.uuid4().hex[:10]
    reused = None
    if args.reuse_final_run:
        prior = args.reuse_final_run.resolve()
        reused = json.loads((prior/'manifest.json').read_text())
        if reused.get('status') != 'COMPLETE' or not reused['collection'].startswith('p1_final_'):
            raise ValueError('Only this tool complete, owned final-round isolated state may be reused')
        if reused['dataset_sha256'] != hashlib.sha256(dataset.read_bytes()).hexdigest():
            raise ValueError('Frozen question/gold changed')
        collection = reused['collection']
        for database in (prior/'state/catalogs').glob('*.sqlite'):
            destination = out/'state/catalogs'/database.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
    service = core(settings(out, collection, 'milvus'), variants['baseline_rrf'])
    frozen = source_hashes()
    manifest = {'collection': collection, 'backend': 'milvus', 'source_run': str(args.source_run.resolve()),
                'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'dataset': str(dataset),
                'questions': len(cases), 'source_hashes': frozen, 'policies': {k: asdict(v) for k,v in variants.items()},
                'model': {'chat': 'gemma3:4b', 'temperature': 0, 'window': 4096, 'output': 700, 'embedding': 'BAAI/bge-small-zh-v1.5'},
                'results': {}, 'notes': ['All seen development/regression; PENDING_HUMAN, not held-out.',
                    'Stable==baseline_rrf; Quality==crossencoder; Experimental==full in this final configuration.',
                    'Every configuration has explicit task constraints and structured table extraction enabled. Globals unchanged.',
                    'No original index writes. Active IR/chunks cloned via public publish into NEW index; isolated versions start at1.',
                    'Original generation budgets2/2/4 preserved, no budget increase to fix errors.',
                    'Table extraction is not LLM generation; model Prompt/evidence and extraction evidence reported separately.']}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        if reused:
            manifest['cloned_documents'] = reused['cloned_documents']
            manifest['reuse_final_run'] = str(args.reuse_final_run.resolve())
            manifest['notes'].append('Read-only existing test index created THIS round; new catalog backup/data_dir, same vectors/IDs. No original/user collection modification.')
            for doc in manifest['cloned_documents']:
                actual = digest(canonical_content(Document.from_dict(service.document(doc['doc_id']))))
                if actual != doc['normalized_document_hash']:
                    raise AssertionError('Owned frozen test index changed')
        else:
            manifest['cloned_documents'] = clone_frozen(args.source_run.resolve(), service)
        save()
        for generate in ([True] if args.generation_only else [False, True]):
            names = ['baseline_rrf', 'crossencoder', 'full'] if generate else list(variants)
            if generate and args.dataset == 'main' and not args.pilot:
                names += ['full_l3_off']
            for name in names:
                if source_hashes() != frozen:
                    raise RuntimeError('SOURCE_DRIFT_DURING_FINAL_EVALUATION')
                policy = variants['full' if name == 'full_l3_off' else name]
                service.controller.policy = policy
                service.support_level3 = name != 'full_l3_off'
                label = ('generated_' if generate else 'retrieval_')+name
                def checkpoint(row):
                    with (out/(label+'.jsonl')).open('a') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                report = run_suite(service, cases, generate=generate, k=3, on_row=checkpoint,
                                   run_label='final_source_PENDING_HUMAN_'+label)
                write_report(report, out/label)
                manifest['results'][label] = report['summary']
                save()
                print(label, report['summary']['status_counts'], report['summary']['mrr'], flush=True)
        manifest['source_hashes_after'] = source_hashes()
        manifest['source_unchanged'] = frozen == manifest['source_hashes_after']
        manifest['status'] = 'COMPLETE' if manifest['source_unchanged'] else 'SOURCE_DRIFT'
        save()
    finally:
        service.close()


if __name__ == '__main__':
    main()
