# connector_pdf

独立 PDF / 扫描件 / 图片 / Office Connector。核心通过 `manifest.json` 发现，
按 `adapter:load` 调用；插件只依赖 `rag_multidocs.contracts` 与
`rag_multidocs.model`，不修改核心源码。

## 接入

`config.json` 保持：

```json
{
  "enabled_connectors": ["txt", "pdf", "connector.markdown", "connector.html"],
  "docparse_repo": "../已有资源/可选组件/docparse",
  "docparse_python": "/path/to/docparse/env/bin/python",
  "docparse_mode": "auto"
}
```

命令不变：

```bash
rag-cli ingest --source pdf --path samples/some.pdf
```

核心通过 `ConnectorRegistry` 的 `context["settings"]` 把 docparse 配置传给插件；
插件不 import `Settings`、`ingest` 或 `cli`。

## 依赖

- 无 docparse 时返回 `CONNECTOR_UNAVAILABLE`，不影响 TXT/Markdown/HTML；
- 原生文本 PDF 由 docparse 原生抽取，不调用 OCR；
- 扫描件/图片/坏文本层才走 docparse → Ollama → Xiaomi-OCR-0；
- docparse CLI 在 `parse` 前会检查 Ollama 后端可达，因此原生 PDF/DOCX 也需要
  Ollama 服务在跑（原生路径不实际调用模型）；
- `samples/campus-corpus/` 已包含原生 PDF、DOCX、扫描 PNG，可用于真实验收；
- 单元测试使用假 docparse，不依赖 Ollama；真实 OCR 验收属于部署/验收步骤。

## 错误

| 情况 | code |
| --- | --- |
| 路径不存在 | `FILE_NOT_FOUND` |
| 扩展名不支持 | `UNSUPPORTED_TYPE` |
| 目录为空 | `EMPTY_SOURCE` |
| docparse 未配置 | `CONNECTOR_UNAVAILABLE` |
| docparse 退出码非 0 / 无 IR | `PARSE_ERROR` |
| 插件自身异常 | `CONNECTOR_ERROR`（registry 统一兜底） |

## 构建/打包

Python 插件无需编译；复制本目录到 `connectors_dir` 后重启 CLI 即可。
禁用：

```bash
rag-cli connectors disable pdf
```

不需要修改 `src/rag_multidocs/`。
