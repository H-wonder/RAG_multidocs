# Connector与Reranker公开契约

B版保留A+双插件扩展点，契约版本为字符串`"1"`。Embedding、Chat和Index保持原Provider/后端接口，不是第三类插件。独立扩展只依赖 `rag_multidocs.contracts`、`rag_multidocs.model`及文档列出的服务，不导入CLI、检索内部或其他插件源码。

## 1 目录与Manifest

```text
project/connectors/connector_jsonl/
  adapter.py
  manifest.json
  README.md
project/rerankers/custom/
  adapter.py
  manifest.json
  README.md
```

每个插件独立目录包含manifest和入口代码，可有相对导入的helper。交付独立目录即可，Python插件无需额外编译；wheel只含Python包，不自动包含仓库外插件/docparse目录。部署需保留目录并配置路径，接入新插件需重启，不是热插拔。

| Manifest字段 | 实际含义 |
|---|---|
| name | 必填非空唯一字符串；冲突不按目录先后任意选 |
| kind | connector或reranker；缺省按connector兼容旧Manifest |
| version | 插件自身版本，供诊断/处理指纹；不等于契约版本 |
| contract_version | 默认1，宿主支持1；不兼容则明确反馈 |
| entry | 单模块名:函数，如adapter:load或adapter:rank；相对helper导入受支持 |
| enabled | 默认true；宿主显式enabled允许列表优先，None用Manifest默认，空集合禁用全部外部插件 |
| requires | 宿主Python分发包依赖名及有限版本比较；不自动安装，不是完整PEP508解析 |
| config | 配置dict，Reranker调用覆盖Manifest默认，传给插件独立副本 |
| execution | 默认process；inprocess只有宿主批准精确路径后才生效 |
| file_extensions | Connector后缀列表，供auto路由；扩展名大小写规范化 |
| auto_priority | Connector整数优先级；高优先级可选择专用Office/Image插件，通用pdf为-10 |
| description / input_schema / output_schema | 说明性字段；当前宿主不执行完整JSON Schema输入输出校验 |
| parser_dependencies | Office等插件声明外部解析环境依赖；不是宿主requires检查，需worker实际导入 |

两类注册表共用 `rag_multidocs.extensions.discover/load_entry`。`ConnectorRegistry`位于 `rag_multidocs.connectors.registry`；`RerankerRegistry`位于 `rag_multidocs.rerankers`。错误Manifest进入discovery_issues；plugin_infos()返回发现对象而非执行成功证明。Core.plugins()汇总并不返回Manifest.config。

## 2 Connector公开函数

```python
from rag_multidocs.contracts import ConnectorResult, ConnectorIssue
from rag_multidocs.model import Document

def load(payload: dict, context: dict | None = None) -> ConnectorResult:
    # payload = {"name": "...", "path": "...", "options": {...}}
    return ConnectorResult(documents=[Document.from_text('示例正文', 'example.record')])
```

payload的name/path/options来自公开ConnectorConfig；context是独立副本，含settings/source_base，约定只读。原内置对象式 `Connector.load(config: ConnectorConfig)`保持兼容，外部 `ConnectorAdapter`是函数协议。推荐返回ConnectorResult，Registry仍兼容旧可规范化输出，非法Document对象会明确报错。

Document字段：doc_id/source/content/title/media_type/metadata/segments。Connector负责解析，不负责切Chunk、Embedding、入库、上下文或模型生成。source保留逻辑来源，Document.from_text生成稳定ID及正文hash；不得伪造原文、表格数值、页码或bbox。完整字段及序列化见[Core API](RAG_CORE_API.md)。

结构化segments是公开Block字典，meta中可保留location/table_data/table_headers/semantic_cells/source_table_id。表格rows/cells保留字符串显示值、空字段、小数和单位；semantic_cells是派生映射，应保留真实row quote。OCR缺位置就为空，不能把图片整框冒充词框。

### 现有JSONL独立示例

实际Manifest的name=connector.jsonl，kind=connector，contract_version=1，version=1.0.0，entry=adapter:load，enabled=true，file_extensions=[.jsonl]。当前读取UTF-8；每行一个JSON对象：

```json
{"source":"public.example#1","title":"示例开放说明","text":"示例资料室周二开放。"}
```

可选blocks进入Document.segments；可选provenance/source_sha256/annotation/origin进入metadata。无source时用文件名#行号；损坏记录返回RECORD_ERROR并继续其他行，文件读取错误READ_ERROR。它不负责任意JSON数组或非UTF-8编码自动判断。

将插件目录放到配置connectors_dir；在enabled_connectors允许列表加入connector.jsonl（B默认已包含），重启后：

```bash
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json ingest --source connector.jsonl --path project/samples/plugin-example.jsonl
```

上面的文件为下文交付整理创建的安全虚构样例，不包含真实业务资料。禁用/启用使用已有 `connectors disable/enable connector.jsonl` 命令，会写本地配置；对已运行的其他进程需重启，已入库内容不会自动删除。新插件名称还需进入宿主允许列表，不能只复制目录就宣称可用。

### 错误隔离

单文件/记录失败返回ConnectorIssue(code,message,source)，部分成功可同时有documents和issues；整体抛错由Registry转CONNECTOR_ERROR。未知、禁用、重名、契约/依赖错误分别反馈。无权假定所有I/O、宿主索引异常均由插件捕获。Connector在宿主进程执行，Office/PDF插件自行调用现有解析子进程；注册表本身不是Connector沙箱。

## 3 Reranker公开函数

```python
from rag_multidocs.contracts import RerankRequest, RankedCandidate

def rank(request: RerankRequest) -> list[RankedCandidate]:
    term = request.config.get('preferred_term', '表格')
    rows = sorted(request.candidates, key=lambda h: term in h.text, reverse=True)
    return [RankedCandidate(h.candidate_id, i+1, 1-i/max(1,len(rows)))
            for i, h in enumerate(rows)]
```

这是现有 `project/rerankers/custom` 的机制示例，不能宣称其是更好的通用排序。无需修改Core：配置/注入Registry、使用已有policy.reranker选择该名称即可。此示例不是组员独立贡献或真人Review证明。

RerankRequest(query, candidates: tuple[SearchHit,...], config=dict, contract_version="1")；SearchHit包含candidate_id/text/source/doc_id/score/semantic_score，frozen dataclass。Reranker只持有排序所需副本，无完整宿主元数据或流程控制权。

返回RankedCandidate(candidate_id, rank, score)列表，必须恰好覆盖全部输入ID，不增不漏不重复。rank为1..N整数排列（bool不接受），score为[0,1]有限数，按rank顺序不增，允许相同分值。不接受NaN/Inf或修改原文/来源；宿主按ID回联原始命中，分数不表示正确概率。

### 注册与调用

```python
from pathlib import Path
from rag_multidocs.rerankers import RerankerRegistry

def main():
    ranks = RerankerRegistry(plugins_dir=Path('project/rerankers'), enabled={'custom'})
    hits = [{'chunk_id':'a','doc_id':'example','source':'example.txt','text':'说明文字'},
            {'chunk_id':'b','doc_id':'example','source':'example.txt','text':'表格内容'}]
    try:
        ordered, warnings = ranks.rerank('custom', '表格', hits)
        print([h['chunk_id'] for h in ordered], warnings)
    finally:
        ranks.close()

if __name__ == '__main__':
    main()
```

`rerank(name, query, hits, config=None, *, timeout_s=None)`返回(命中list, warning list)，config覆盖Manifest.config。通过 `dataclasses.replace(recommended_policies()['stable'], reranker='custom')`在create_core(policy=...)选择；不需要新增策略名或改核心if/else。name=none保持该重排阶段输入顺序。

缺插件、禁用、重名、版本/依赖不符、非法输出、启动/超时/异常时回退输入顺序，warning常有code/message/plugin/fallback=rrf/severity。此处rrf是兼容标签，不保证后续MMR/上下文完全保持排序，也不改变底层Milvus的WeightedRanker融合语义。完整基础路径对照使用RetrievalPolicy.baseline()。

## 4 执行方式与资源释放

默认外部Reranker在spawn子进程执行，超时terminate/join，必要时kill。只有宿主trusted_inprocess精确路径批准且Manifest要求inprocess，才可同进程；默认随包Lightweight被批准，其任意阻塞代码没有可强杀保证。没有网络/文件恶意代码沙箱，插件应为受信本地代码。

`RerankerRegistry(..., persistent_paths=[absolute_path], max_worker_restarts=1)`授权一个串行常驻Worker，插件不能自授常驻权限。CLI build_runtime对随包CrossEncoder路径显式批准；单独create_core默认Registry并不会自动得到相同常驻授权，需宿主按API示例传入。常驻进程内模型缓存，失败回收，本请求不无限重试，下次重启受次数限制；不是多进程池或热加载。

`registry.close()`或 `RAGCore.close()/Runtime.close()`释放Worker和连接。requires检测的是宿主依赖；docparse的独立环境需通过配置和解析器自检确认。重排模型缺缓存默认告警回退，不偷偷下载；正常启动不应通过新增模型绕开依赖错误。

## 5 开发与测试

1. 建独立目录，实现公开入口，写Manifest和自己的README。
2. 声明必要依赖、输入后缀、错误和配置；不导入宿主私有内部编排。
3. 配置目录/允许列表/策略并重启；核对plugin_infos/discovery_issues和实际调用Trace。
4. 用临时虚构资料检查正常、坏输入、禁用、抛错与排序ID/rank/score不合法的行为。
5. 对比Core源码hash，证明只添加扩展即可改变实际行为；真实成员贡献/Review另行记录。

已有专项入口：

```bash
PYTHONPATH=project/src python -m pytest project/tests/test_a_plus_plugins.py::test_independent_connector_full_flow_disabled_and_failure -v
PYTHONPATH=project/src python -m pytest project/tests/test_a_plus_plugins.py::test_independent_reranker_changes_order_without_core_edits -v
```

用户已确认JSONL专项终端结果1 passed及全新文件实际导入问答；三道业务题只记录用户完成反馈，没有虚构逐题日志。本轮不重新执行大规模模型或插件验收。详见[最终验收](testing/FINAL_ACCEPTANCE.md)。
