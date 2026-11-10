# 架构

## 设计目标

1. **能不上模型就不上模型**：矢量 PDF 的文本层比任何 OCR 都准，且快两个数量级。
2. **分诊在页级，不在文档级**：真实 PDF 几乎都是混合的（矢量正文 + 扫描插页）。
3. **失败可自愈**：所有可检测的失败模式走自动重试/升级，而不是丢给人工。
4. **表示与渲染分离**：内部 IR 是唯一事实来源，Markdown/JSON 只是它的视图。

## 数据流

```
输入文件
  │
  ├─ docx/pptx/xlsx/txt ──→ 原生 XML 解析 ───────────────┐
  │                                                      │
  └─ PDF ──→ 逐页探测 (字符数/乱码率/图像覆盖率)          │
               ├─ native / hybrid ──→ PyMuPDF 抽文字+bbox+表格
               │                       （不经过模型）
               └─ scanned ──→ 渲染 (DPI) ──→ VLM
                                            ├─ page 模式：整页 Markdown
                                            └─ region 模式：版面检测
                                                              → 裁块
                                                              → 逐区域 OCR
                                                              → 按阅读顺序合并
  │                                                      │
  └──────────────────────→ 质量门 ←──────────────────────┘
                              │
                   通过 ──────┴────── 失败
                    │                  │
                    ▼                  ▼
                   IR          重试 / 升级 / 标记 needs_review
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Markdown       JSON      RAG chunks（后续）
```

## 页级分诊

`docparse/pdfio.py::probe_page` 输出 `kind`：

| kind | 判据 | 路由 |
|---|---|---|
| `native` | 字符数 ≥10，乱码率 ≤5%，图像覆盖 <30% | 原生抽取 |
| `hybrid` | 同上，但图像覆盖 >30% | 原生抽取 + 图像块 |
| `scanned` | 无文本层 / 乱码率高 / 整页大图 | 渲染 + VLM |
| `blank` | 无文字且无图像 | 跳过 |

关键点：**有文本层的页绝不送 OCR**。即使它是扫描件，只要 OCR 文本层质量健康，直接用它。

## 自动升级阶梯

```
page 模式
  ├─ 重复/截断/空输出 → presence_penalty=1.5 重试
  ├─ 仍失败          → region 模式（若已安装 PaddleX）
  └─ 仍失败          → needs_review=true + 具体原因
region 模式
  ├─ 单区域重复      → 该区域重试
  └─ 单区域异常      → 丢弃该区域并告警，不影响整页
```

`--mode` 可强制 `page` 或 `region`；`auto` 走上面的阶梯。

## 文档中间表示（IR）

`docparse/ir.py`：

```json
{
  "path": "invoice.pdf",
  "sha256": "...",
  "doc_type": "pdf",
  "needs_review": false,
  "warnings": [],
  "pages": [{
    "number": 1, "kind": "scanned", "mode": "page",
    "blocks": [{
      "type": "page",
      "bbox": [0, 0, 1000, 1000],
      "reading_order": 0,
      "content": "...",
      "source": "vlm_page",
      "confidence": 0.94,
      "warnings": []
    }]
  }],
  "meta": {"kie": [{"page": 1, "json": {...}, "grounding": {...}}]}
}
```

- `bbox` 统一归一化到 0–1000，跨 DPI 稳定。
- `source` 记录来源：`native_text` / `vlm_page` / `vlm_region` / `layout`。
- `confidence` 是启发式分数，不是模型概率；`warnings` 才是可操作的信号。

## 模块边界

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `pipeline.py` | 编排、分诊、升级 | 不拼 HTTP、不解析 PDF 细节 |
| `pdfio.py` | PDF 探测 / 原生抽取 / 渲染 | 不调用模型 |
| `backend.py` | Ollama 协议、图像编码 | 不管业务逻辑 |
| `layout.py` | 版面检测 | 不 OCR |
| `quality.py` | 质量判定、KIE 回查 | 不改写正文 |
| `writers.py` | IR → 输出 | 不产生新内容 |
| `evaluate.py` | 与参考对比 | 不参与解析 |
| `web/app.py` | HTTP API、上传/下载、健康检查 | 不直接解析 |
| `web/jobs.py` | 任务队列、页级进度、取消、持久化 | 不实现解析 |

## 运行形态

- **CLI**：`python -m docparse parse ...`，同步执行，适合脚本和批处理。
- **Web**：`python -m docparse web`，单 worker 队列 + 页级进度 + 取消/重试 + 历史。

两者共用同一个 `Parser` 和 `writers`；Web 只增加编排层，不改变解析行为。
进度与取消通过 `Parser.parse(progress=..., should_cancel=...)` 注入，取消检查点在页之间和区域之间。

## 关键决策

1. **VLM 是可替换后端**：`backend.py` 只暴露 `chat(image, prompt, ...)`，换 vLLM/llama.cpp 不影响上层。
2. **不重复造轮子**：上游的 OTSL 转换、重复检测、Markdown 合并直接 vendor（见 `docs/third-party.md`），只改 import。
3. **page 优先、region 兜底**：手写/书法/不规则版面用 region 会因切块丢上下文；印刷件用 region 更稳。auto 先 page，失败再 region。
