# 校园图书馆 / RAG 课程多格式样例集

这是一个用于功能演示和回归测试的小型领域语料库：
主题是“校园图书馆服务 + RAG 课程项目”，覆盖 6 种输入格式，便于比较
TXT、Markdown、HTML、PDF、Office 和扫描图片的接入行为。

## 文件

| 文件 | 类型 | 建议 Connector | 说明 |
| --- | --- | --- | --- |
| `txt/library-hours.txt` | 纯文本 | `txt` | 图书馆开放时间、馆际互借 |
| `txt/campus-faq.txt` | 纯文本 | `txt` | 借阅期限、逾期、研讨间 |
| `markdown/rag-basics.md` | Markdown | `connector.markdown` | 检索器、切块、引用 |
| `markdown/lab-guide.md` | Markdown | `connector.markdown` | 实验步骤、索引、评测 |
| `html/library-services.html` | HTML | `connector.html` | 馆际互借与研讨间；含应被忽略的 script/style |
| `pdf/course-project-guide.pdf` | 原生 PDF | `pdf` | 课程项目提交指南，带文本层，不调用 OCR（docparse 预检需 Ollama 可达） |
| `pdf/scanned-library-notice.png` | 扫描图片 | `pdf` | 期末开放时间通知，需要 OCR |
| `office/course-faq.docx` | DOCX | `pdf` | 课程常见问题，docparse 原生解析，不调用 OCR（同上） |

`pdf` 插件同时处理 PDF、图片和 Office；它通过 docparse 子进程解析。

## 快速测试（离线模式）

TXT/Markdown/HTML 不依赖 Milvus、docparse 或模型：

```bash
cd project
export RAG_INDEX_BACKEND=keyword
export RAG_DATA_DIR=/tmp/rag-campus-data

rag-cli ingest --source txt --path samples/campus-corpus/txt
rag-cli ingest --source connector.markdown --path samples/campus-corpus/markdown
rag-cli ingest --source connector.html --path samples/campus-corpus/html

rag-cli search "馆际互借" --top-k 3
rag-cli search "切块" --top-k 3
rag-cli search "借阅期限" --top-k 3
rag-cli docs list
```

预期来源包括 `library-hours.txt`、`library-services.html`、`rag-basics.md`、
`campus-faq.txt`。

## PDF / Office / 图片

需要 docparse 环境，并启动 Ollama。注意：**docparse CLI 在 `parse` 前会检查
Ollama 后端可达**，所以即使原生 PDF/DOCX 不做 OCR，也需要 Ollama 服务在跑；
原生路径不会实际调用 OCR 模型。

```bash
# 启动 Ollama（模型已拉取时；首次见 docparse README）
scripts/start_ollama.sh
python -m docparse doctor          # 在 docparse 环境执行

# 原生 PDF：docparse 原生文本层，不调用 OCR
rag-cli ingest --source pdf --path samples/campus-corpus/pdf/course-project-guide.pdf
rag-cli search "扩展要求" --top-k 3

# DOCX：docparse 原生解析，不调用 OCR
rag-cli ingest --source pdf --path samples/campus-corpus/office/course-faq.docx
rag-cli search "提交" --top-k 3

# 扫描图片：走 OCR
rag-cli ingest --source pdf --path samples/campus-corpus/pdf/scanned-library-notice.png
rag-cli search "期末开放时间" --top-k 3
```

本次样例已实测：

- `course-project-guide.pdf`：原生 PDF，1 个文档/1 个块；
- `course-faq.docx`：DOCX 原生解析，1 个文档/1 个块；
- `scanned-library-notice.png`：Ollama + Xiaomi-OCR-0，1 个文档/1 个块，
  检索命中“期末开放时间”。

如果 Ollama 未启动，docparse 会以 `inference backend is not reachable` 退出，
PDF/DOCX/图片都会失败；这是 docparse 的预检，不是 TXT/Markdown/HTML 的问题。

## 用途

- 演示“接入 → 统一 Document → 切块 → 检索 → 来源”的完整链路；
- 比较原生 PDF、DOCX 与扫描图片是否触发 OCR；
- 作为新增 Connector 回归测试的固定输入；
- 所有内容为课程演示编写，不含真实个人信息。
