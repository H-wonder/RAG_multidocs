# 输出格式

所有输出都是 **IR（`docparse/ir.py`）的序列化视图**，解析逻辑不依赖具体格式。
当前只实现了两种；新增格式不需要改动流水线。

## 1. Markdown（`<stem>.md`）

| 元素 | 表示 |
|---|---|
| 正文 | 纯文本 / Markdown |
| 表格 | HTML `<table><tr><td>…`（模型输出 OTSL，经 `vendor/otsl.py` 转换） |
| 公式 | LaTeX：行内 `$…$`，块级 `$$…$$` |
| 多页 | `<!-- page N -->` 分隔 |
| 页眉/页脚 | 整页模式按上游 prompt 忽略；region 模式按标签丢弃 |
| 图片 | 不导出 |

示例：

```markdown
<!-- page 1 -->

Quarterly Report 2026

<table><tr><td>项目</td><td>金额</td></tr><tr><td>螺丝</td><td>500.00</td></tr></table>

$$ E = mc^2 $$
```

## 2. IR JSON（`<stem>.json`）

```jsonc
{
  "path": "invoice.pdf",
  "sha256": "…",
  "doc_type": "pdf",
  "needs_review": false,
  "warnings": [ { "code": "…", "severity": "medium", "message": "…" } ],
  "meta": {
    "pages_total": 3,
    "kie": [ { "page": 1, "json": {…}, "grounding": {…}, "raw": "…" } ]
  },
  "pages": [
    {
      "number": 1,
      "width": 1240, "height": 1754,
      "kind": "scanned",          // native | scanned | hybrid | blank | image
      "mode": "region",           // native | page | region | failed
      "warnings": [],
      "latency_ms": 7021,
      "usage": { "prompt_tokens": 1503, "completion_tokens": 304 },
      "blocks": [
        {
          "id": "p1b0",
          "page": 1,
          "type": "table",        // title|text|table|formula|figure|page|page_markdown
          "bbox": [131, 210, 840, 519],   // 归一化 0–1000
          "reading_order": 0,
          "content": "<table>…</table>",
          "html": "<table>…</table>",
          "source": "vlm_region",         // native_text|vlm_page|vlm_region|layout|native_office
          "confidence": 1.0,
          "warnings": [],
          "meta": { "layout_label": "table", "layout_score": 0.97 }
        }
      ]
    }
  ]
}
```

说明：

- `bbox` 统一归一化到 0–1000，跨 DPI 稳定，可直接映射回页面图像做高亮。
- `confidence` 是启发式分数，不是模型概率；可信信号看 `warnings`。
- `region` 模式的页里有一个合成块 `type="page_markdown"`，内容是上游合并后的整页 Markdown；
  其余块是逐区域结果。渲染 Markdown 时优先用这个合成块。
- `source` 决定可信度：`native_text`（文本层）> `vlm_region` > `vlm_page`。
- 只有 `severity="high"` 的告警会把 `needs_review` 置为 true。

## 3. HTML（`<stem>.html`）

独立文件，内嵌样式（自动适配浅色/深色），由 Markdown 渲染而来。
原始 `<table>` 块原样保留，公式以 LaTeX 源码显示（不做前端 KaTeX，保持离线可用）。

## 4. TXT（`<stem>.txt`）

纯文本。HTML 表格转换为制表符分隔行，其他标签剥离，实体反转义。
适合全文检索或喂给不支持 Markdown 的旧系统。

## 5. 当前不支持

- Word `.docx`
- 表格单独导出 CSV / XLSX
- 文档内插图 / 页面图片导出
- CLI 批量 ZIP（Web 界面已支持整批打包）
- RAG chunks（JSONL）

## 6. 格式矩阵

| 格式 | 用途 | 状态 |
|---|---|---|
| `md` | 人读、喂给 LLM | 已实现 |
| `json` | 下游系统、审计 | 已实现 |
| `html` | 浏览器直接查看，保留表格样式 | 已实现 |
| `txt` | 纯文本检索 | 已实现 |
| `zip` | 批量下载 | Web 已实现（CLI 无） |
| `csv` / `xlsx` | 表格逐 sheet 导出 | P1 |
| `docx` | 可编辑交付 | P1 |
| `assets/` | 插图、页面图 | P1 |
| `chunks.jsonl` | RAG / 向量化 | P2 |

CLI：`--format md,json,html,txt`；Web：勾选输出格式。

设计约束：**格式是序列化器，不得反向影响解析**。新增格式只实现 `(Document) -> str|bytes`。
