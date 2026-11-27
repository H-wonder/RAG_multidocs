"""Independent, inspectable CSV + reranker experiment; no pytest imports.

Always writes a fresh isolated directory. Never touches prior user experiments.
"""
import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from rag_local import PROJECT, core, settings
from rag_multidocs.contracts import ConnectorConfig
from rag_multidocs.rerankers import RerankerRegistry
from rag_multidocs.retrieval.controller import RetrievalPolicy

CSV_ADAPTER = '''
import csv
from pathlib import Path
from rag_multidocs.contracts import ConnectorResult,ConnectorIssue
from rag_multidocs.model import Document,sha256_file
def load(payload,context=None):
    path=Path(payload['path'])
    result=ConnectorResult()
    try:
        with path.open(newline='',encoding='utf8') as file:
            for number,row in enumerate(csv.DictReader(file),1):
                if not row.get('id') or not row.get('warranty_months'):
                    result.issues.append(ConnectorIssue('CSV_ROW_ERROR','Missing required id/warranty_months',path.name))
                    continue
                text='Device: '+row['id']+'\\nWarranty: '+row['warranty_months']+' months'
                doc=Document.from_text(text,path.name+'#'+row['id'],metadata={'row':number,'file_sha256':sha256_file(path)})
                result.documents.append(doc)
    except OSError:
        result.issues.append(ConnectorIssue('CSV_READ_ERROR','Cannot read input',path.name))
    return result
'''
RANK_ADAPTER = '''
from rag_multidocs.contracts import RankedCandidate
def rank(request):
    mode=request.config.get('mode','reverse')
    if mode=='raise': raise RuntimeError('Deliberate extension failure')
    rows=[RankedCandidate(h.candidate_id,i+1,1-i/(len(request.candidates)+1)) for i,h in enumerate(reversed(request.candidates))]
    if mode=='invalid' and rows: rows[0]=RankedCandidate('unregistered-id',1,1)
    return rows
'''


def hashes():
    return {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').rglob('*.py')}


def run(output,backend='keyword',mock=False):
    out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh extension experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    before=hashes()
    plug=out/'connectors'/'csv_independent'
    plug.mkdir(parents=True)
    (plug/'manifest.json').write_text(json.dumps({'name':'connector.csv','kind':'connector','contract_version':'1','version':'1.0','entry':'adapter:load','enabled':True}))
    (plug/'adapter.py').write_text(CSV_ADAPTER)
    rank=out/'rerankers'/'reverse_independent'
    rank.mkdir(parents=True)
    rank_manifest={'name':'reverse','kind':'reranker','contract_version':'1','version':'1.0','entry':'adapter:rank','config':{'mode':'reverse'}}
    (rank/'manifest.json').write_text(json.dumps(rank_manifest))
    (rank/'adapter.py').write_text(RANK_ADAPTER)
    file=out/'devices.csv'
    def write(months):
        with file.open('w',newline='',encoding='utf8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['id','warranty_months'])
            writer.writerows([['A',months],['B',24],['C',36]])
    write(12)
    config=settings(out,'p1_extension_'+__import__('uuid').uuid4().hex[:10],backend)
    config.connectors_dir=str(plug.parent)
    config.enabled_connectors=['txt','connector.csv']
    base=RetrievalPolicy.baseline(candidate_k=12,context_k=6,context_tokens=2000,model_context_tokens=4096,output_tokens=700,atomic_generation=True)
    service=core(config,base)
    original_rerankers=service.controller.rerankers
    service.controller.rerankers=RerankerRegistry(rank.parent,timeout_s=5)
    if mock:
        class Echo:
            def complete(self,messages):
                import re
                value=re.search(r'Warranty: (\d+) months', messages[1]['content'])[1]
                return f'Warranty: {value} months [1].'
        service.chat=Echo()
    result={'mode':'mock_generation' if mock else 'real_local_gemma3:4b','backend':backend,'collection':config.milvus_collection if backend=='milvus' else None,'checks':{}}
    try:
        parsed=service.ingestion.registry.load('connector.csv',ConnectorConfig('connector.csv',str(file)))
        result['parsed_before']=[d.to_dict() for d in parsed.documents]
        result['checks']['csv_three_correct_documents']=len(parsed.documents)==3 and [d.content for d in parsed.documents]==['Device: A\nWarranty: 12 months','Device: B\nWarranty: 24 months','Device: C\nWarranty: 36 months']
        result['ingestion']=service.ingest('connector.csv',str(file)).as_dict()
        ids=[d['doc_id'] for d in result['ingestion']['documents']]
        baseline=service.search('Warranty',doc_ids=ids)
        result['baseline']=baseline
        baseline_ids=[h['chunk_id'] for h in baseline['hits']]
        service.controller.policy=replace(base,reranker='reverse')
        reversed_result=service.search('Warranty',doc_ids=ids)
        result['reversed']=reversed_result
        result['checks']['complete_reversed_order']=len(baseline_ids)==3 and [h['chunk_id'] for h in reversed_result['hits']]==list(reversed(baseline_ids))
        original={h['chunk_id']:(h['text'],h['source']) for h in baseline['hits']}
        result['checks']['original_text_source_ownership']=all((h['text'],h['source'])==original[h['chunk_id']] for h in reversed_result['hits'])
        service.controller.rerankers.enabled=set()
        disabled=service.search('Warranty',doc_ids=ids)
        result['disabled']=disabled
        result['checks']['disabled_restores_complete_baseline']=[h['chunk_id'] for h in disabled['hits']]==baseline_ids and any(w['code']=='RERANKER_DISABLED' for w in disabled['warnings'])
        service.controller.rerankers.enabled=None
        for mode,code in [('raise','RERANKER_EXECUTION_ERROR'),('invalid','RERANKER_INVALID_OUTPUT')]:
            rank_manifest['config']['mode']=mode
            (rank/'manifest.json').write_text(json.dumps(rank_manifest))
            response=service.search('Warranty',doc_ids=ids)
            result[mode]=response
            result['checks'][mode+'_complete_fallback']=[h['chunk_id'] for h in response['hits']]==baseline_ids and any(w['code']==code and w['fallback']=='rrf' for w in response['warnings'])
        service.controller.policy=base
        first_id=result['ingestion']['documents'][0]['doc_id']
        result['csv_answer']=service.ask('What is the warranty period for device A?',doc_ids=[first_id])
        result['checks']['csv_rag_answer']='12' in result['csv_answer']['answer'] and bool(result['csv_answer']['citations'])
        service.ingestion.registry.enabled=set()
        disabled_parse=service.ingestion.registry.load('connector.csv',ConnectorConfig('connector.csv',str(file)))
        result['checks']['csv_disabled']=not disabled_parse.documents and disabled_parse.issues[0].code=='CONNECTOR_DISABLED'
        service.ingestion.registry.enabled={'txt','connector.csv'}
        write(18)
        changed=service.ingestion.registry.load('connector.csv',ConnectorConfig('connector.csv',str(file)))
        result['parsed_after']=[d.to_dict() for d in changed.documents]
        result['checks']['csv_input_changes_output']=changed.documents[0].content=='Device: A\nWarranty: 18 months' and changed.documents[1].content==parsed.documents[1].content
        result['update']=service.ingest('connector.csv',str(file)).as_dict()
        updated=service.search('Warranty',doc_ids=[first_id])
        result['updated_hits']=updated['hits']
        result['checks']['csv_active_update']=bool(updated['hits']) and all('18 months' in h['text'] and '12 months' not in h['text'] for h in updated['hits'])
    except Exception as exc:
        result['error']=type(exc).__name__
        raise
    finally:
        service.close()
        original_rerankers.close()
        result['core_before']=before
        result['core_after']=hashes()
        result['checks']['core_unchanged']=bool(before) and before==result['core_after']
        result['status']='PASS' if result['checks'] and all(result['checks'].values()) and not result.get('error') else 'FAIL'
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['keyword','milvus'],default='keyword')
    parser.add_argument('--mock',action='store_true',help='Explicit offline mock generation, never labeled real QA')
    args=parser.parse_args()
    report=run(args.output,args.backend,args.mock)
    print(json.dumps({'status':report['status'],'mode':report['mode'],'checks':report['checks']},ensure_ascii=False,indent=2))
