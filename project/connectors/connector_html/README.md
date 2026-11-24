# connector.html

独立 HTML 目录 Connector：递归读取 `.html` / `.htm`，用标准库
`html.parser` 提取 `<title>` 和可见正文，跳过 `<script>` / `<style>` /
`<noscript>`，再转换为统一 `Document`。

## 接入

```bash
rag-cli ingest --source connector.html --path samples/html
rag-cli search "关键词"
```

`config.json`：

```json
{
  "enabled_connectors": ["txt", "pdf", "connector.markdown", "connector.html"]
}
```

插件只新增 `connectors/connector_html/`，不修改 `src/rag_multidocs/`；
禁用：

```bash
rag-cli connectors disable connector.html
```

## 错误

| 情况 | code |
| --- | --- |
| 路径不存在 | `FILE_NOT_FOUND` |
| 扩展名不是 `.html/.htm` | `UNSUPPORTED_TYPE` |
| 目录中没有 HTML | `EMPTY_SOURCE` |
| 文件没有可见正文 | `EMPTY_FILE` |
| 读取失败 | `READ_ERROR` |

## 构建/打包

Python 插件无需编译；复制目录到 `connectors_dir` 后重启 CLI 即可。
