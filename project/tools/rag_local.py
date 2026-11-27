"""Local acceptance composition helpers; no UI, env mutation or paid endpoints."""
from pathlib import Path
from rag_multidocs.config import Settings
from rag_multidocs.qa.llm import OpenAICompatibleChat
from rag_multidocs.retrieval.rewrite import SemanticRewriter
from rag_multidocs.rerankers import RerankerRegistry
from rag_multidocs.services import create_core

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
PARSER = '/home/genevievelin/anaconda3/envs/docparse/bin/python'


def chat():
    properties = {'text':{'type':'string'},'citation_ids':{'type':'array','items':{'type':'integer','minimum':1},'minItems':1,'maxItems':1},'supporting_quotes':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':1}}
    schema={'type':'object','properties':{'status':{'type':'string','enum':['answered','partial','insufficient_evidence']},'claims':{'type':'array','maxItems':3,'items':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}},'required':['status','claims'],'additionalProperties':False}
    return OpenAICompatibleChat('http://127.0.0.1:11434/v1','gemma3:4b',temperature=0,max_tokens=700,timeout=60,retries=1,response_format={'type':'json_schema','json_schema':{'name':'cited_claims','strict':True,'schema':schema}})


def rewriter():
    model=OpenAICompatibleChat('http://127.0.0.1:11434/v1','gemma3:4b',temperature=0,max_tokens=180,timeout=30,retries=1,response_format={'type':'json_schema','json_schema':{'name':'query_rewrites','strict':True,'schema':{'type':'array','maxItems':2,'items':{'type':'string'}}}})
    return SemanticRewriter(model)


def settings(output, collection, backend='milvus', parser_python=PARSER):
    return Settings(index_backend=backend,fallback_to_keyword=False,milvus_uri='http://127.0.0.1:19530',milvus_db='a_plus_validation',milvus_collection=collection,embed_backend='local',embed_dim=512,data_dir=str(Path(output).resolve()/'state'),connectors_dir=str(PROJECT/'connectors'),enabled_connectors=['txt','pdf','connector.markdown','connector.html','connector.jsonl','connector.image','connector.word','connector.excel'],docparse_repo=str(ROOT/'已有资源/可选组件/docparse'),docparse_python=parser_python)


def core(config, policy, *, chunk_options=None):
    registry=RerankerRegistry(timeout_s=60,persistent_paths=[PROJECT/'rerankers/crossencoder'])
    return create_core(config,base_dir=PROJECT,offline=True,policy=policy,rerankers=registry,chat=chat(),rewriter=rewriter(),chunk_options=chunk_options)


def import_corpus(service, fixtures):
    campus=PROJECT/'samples/campus-corpus'
    inputs=[('txt',campus/'txt',{}),('connector.markdown',campus/'markdown',{}),('connector.html',campus/'html',{}),('pdf',campus/'pdf/course-project-guide.pdf',{}),('connector.word',campus/'office/course-faq.docx',{}),('connector.image',campus/'pdf/scanned-library-notice.png',{}),('connector.image',fixtures/'receipt.png',{}),('connector.image',fixtures/'receipt.jpg',{}),('connector.word',fixtures/'handbook.docx',{'ocr_images':True,'table_header_rows':{'2':2}}),('connector.excel',fixtures/'measurements.xlsx',{'sheet_header_rows':{'Merged':2}})]
    results=[]
    for name,path,options in inputs:
        report=service.ingest(name,str(path),options)
        results.append(report.as_dict())
        print('ingest',name,path.name,report.ok,report.chunks,flush=True)
    return results
