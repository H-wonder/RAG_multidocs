# connector.markdown

独立 Markdown Connector。核心只通过 `manifest.json` + `adapter.py` 发现它，
不需要修改核心源码。

## 接口

```python
def load(payload: dict, context: dict | None = None) -> ConnectorResult
```

`payload`：

```json
{"name": "connector.markdown", "path": "samples/markdown", "options": {}}
```

返回 `ConnectorResult(documents=[Document...], issues=[ConnectorIssue...])`。

## 行为

- 输入可以是单个 `.md`/`.markdown` 文件，或目录；
- 目录递归读取；
- 标题取第一个 `# ` 标题，否则用文件名；
- 保留相对路径作为 `Document.source`；
- 单文件读取失败记录 `READ_ERROR`，其他文件继续；
- 空目录报 `EMPTY_SOURCE`。

## 接入

插件目录已放在 `project/connectors/connector_markdown/`，默认启用。
核心通过 `ConnectorRegistry` 扫描 `connectors_dir`，不需要改核心。

禁用（验证扩展可插拔）：

```bash
rag-cli connectors disable connector.markdown
rag-cli ingest --source txt --path samples/notes.txt   # 核心仍可用
```

## 依赖

只依赖 `rag_multidocs.model` / `rag_multidocs.contracts`，不 import 核心的
IngestService、CLI 或索引实现。
