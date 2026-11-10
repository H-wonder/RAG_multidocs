# Web 界面

## 目标

本地启动一个服务，浏览器打开即可完成：**批量选文件 → 选参数与输出格式 → 看进度 → 预览/下载**。
不需要前端构建，不需要联网，默认只监听 `127.0.0.1`。

## 状态（P0 已实现）

```bash
python -m docparse web          # http://127.0.0.1:8787
```

已实现：

- 批量拖拽/多选，混合格式
- 模式 / DPI / 最大页数 / KIE 字段 / 输出格式（md、json、html、txt）
- 单 worker 队列，页级进度
- 取消（页边界生效）、失败/取消文件重试
- 任务持久化：`~/.docparse/jobs/<id>/`，重启后历史可查；未完成任务标为 `interrupted`
- 结果预览：Markdown/HTML（iframe）/JSON/TXT/告警汇总
- 单文件下载 + 整批 ZIP
- `/api/health` 检查 Ollama、模型、layout、磁盘

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DOCPARSE_HOME` | `~/.docparse` | 任务与产物目录 |
| `DOCPARSE_MAX_UPLOAD_MB` | `200` | 单文件上传上限 |

启动参数：`--host`（默认 `127.0.0.1`）、`--port`（默认 `8787`）。

P0 未包含（见下方路线）：bbox 叠加、DOCX/CSV/XLSX 导出、鉴权、删除任务。

## 技术选型

| 选择 | 理由 |
|---|---|
| FastAPI + 原生 HTML/JS | 本地单用户，无需 Node/打包；静态文件直接由 Python 服务 |
| 单 worker 线程 | Ollama `NUM_PARALLEL=1`，并行只会排队；串行还能避免显存峰值 |
| 文件系统持久化 | `~/.docparse/jobs/<job_id>/` 存 `job.json` + 输入 + 输出，重启可恢复 |
| 轮询（1s） | 比 SSE 简单，任务粒度是秒级；后续可换 SSE |
| 服务端渲染 Markdown | 不依赖 CDN；用 `markdown-it-py` 或 `markdown` 生成预览 HTML |

## 页面结构

```
┌────────────────────────────────────────────────────────────┐
│ docparse            [新建任务] [历史] [设置]     ● Ollama 正常 │
├───────────────┬────────────────────────────────────────────┤
│ 1. 上传       │  拖拽多个文件 / 选择文件夹                    │
│               │  支持 pdf png jpg tiff docx pptx xlsx        │
├───────────────┼────────────────────────────────────────────┤
│ 2. 参数       │  模式: auto | page | region                  │
│               │  DPI: 200   页范围: ___-___                  │
│               │  KIE 字段: [发票号码] [开票日期] [+ 自定义]   │
│               │  输出: ☑ md ☑ json ☐ html ☐ txt ☐ docx      │
│               │        ☐ 表格 csv/xlsx   ☐ 页面图片           │
├───────────────┼────────────────────────────────────────────┤
│ 3. 任务队列   │  文件 | 状态 | 页数 | 耗时 | 告警 | 操作      │
│               │  a.pdf  完成   3/3   12s   —     预览 下载   │
│               │  b.jpg  解析中 2/5    8s   —     取消        │
│               │  c.pdf  待复核 1/1    4s   2高   查看        │
├───────────────┴────────────────────────────────────────────┤
│ 4. 结果抽屉   │ [Markdown] [IR JSON] [告警] [bbox 叠加]      │
│               │ 预览 / 单文件下载 / 打包 ZIP                  │
└────────────────────────────────────────────────────────────┘
```

### 关键交互

- **批量**：拖拽多文件或选择整个文件夹；队列逐个处理。
- **每任务参数**：可以整批统一，也可以对单个文件覆盖。
- **进度**：文件级 + 页级；取消在当前页结束后生效（不中断进行中的请求）。
- **告警**：按 severity 折叠；`high` 的文件在列表里标红，但**不阻塞下载**。
- **预览**：
  - Markdown 渲染后查看；表格保留 HTML。
  - IR 树形查看器，点 block 定位 bbox。
  - bbox 叠加：把页面缩略图与 block 框画在 canvas 上，颜色按类型区分。
- **下载**：单文件、多选、整批 ZIP。
- **历史**：按时间/文件名/状态筛选；支持用不同参数重跑。

## API 草案

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/jobs` | multipart：files + options(JSON) → `{job_id}` |
| `GET` | `/api/jobs` | 任务列表（分页/筛选） |
| `GET` | `/api/jobs/{id}` | 状态、每文件进度、告警 |
| `POST` | `/api/jobs/{id}/cancel` | 取消排队/当前任务 |
| `POST` | `/api/jobs/{id}/retry` | 重跑指定文件 |
| `GET` | `/api/jobs/{id}/files/{name}` | 下载单个产物 |
| `GET` | `/api/jobs/{id}/bundle.zip` | 打包下载 |
| `GET` | `/api/jobs/{id}/pages/{n}.png` | 页面缩略图（bbox 叠加用） |
| `GET` | `/api/health` | Ollama、模型、layout、磁盘 |
| `GET/PUT` | `/api/settings` | 默认模式/DPI/格式/设备 |

## 任务模型

```jsonc
{
  "id": "20261106-1130-ab12",
  "created_at": "2026-11-06T11:30:00+08:00",
  "status": "running",              // queued|running|done|partial|failed|cancelled
  "options": { "mode": "auto", "dpi": 200, "formats": ["md","json"],
               "fields": ["发票号码","价税合计"] },
  "files": [
    { "name": "a.pdf", "status": "done", "pages": 3, "done_pages": 3,
      "elapsed_ms": 12000, "warnings": 0, "outputs": ["a.md","a.json"] }
  ]
}
```

存储：

```
~/.docparse/jobs/<job_id>/
├── job.json
├── input/            # 原始文件（可选，按需保留）
└── output/           # 产物 + pages/*.png（若开启）
```

## 并发与资源

- 单 worker；队列长度在 UI 可见。
- Ollama 请求不并发（`OLLAMA_NUM_PARALLEL=1`），避免 8GB 显存峰值翻倍。
- layout 默认 CPU；如果切到 GPU，worker 会在 layout 阶段与 VLM 竞争显存，不建议。
- 每页渲染后及时释放 PIL 图像；大 PDF 只保留当前页位图。
- 取消检查点放在“页之间”和“区域之间”。

## 安全

- 默认 `127.0.0.1`，不对外监听。
- 文件名清洗，禁止 `..` 路径穿越；输出目录按 job 隔离。
- 上传大小上限（可配置），单文件页数上限。
- 不访问外部网络；模型与布局权重全部本地。
- 如需局域网访问：加 token 鉴权 + HTTPS 反代，并把 `needs_review` 的文件隔离。

## 后续路线

P1–P3 的详细方案（目标、做法、验收、估算、依赖、推荐顺序）见 [路线图](roadmap.md)。

- **P1 生产可用**：表格 CSV/XLSX、DOCX、bbox 叠加、KIE 模板 + 校验、复核队列、去重缓存、版本记录、systemd
- **P2 质量与规模化**：字段交叉校验、跨页表格/span、页眉页脚策略、插图 caption、评测面板、异常输入、增量重解析、吞吐
- **P3 扩展**：RAG chunks/检索、插件式格式、多用户鉴权、任务优先级、多推理后端

## 验收标准（P0）

1. `python -m docparse web` 启动后浏览器可访问。
2. 一次选 10 个混合格式文件，全部进入队列并有页级进度。
3. 勾选 `md + json + zip`，下载包内文件齐全、命名正确。
4. 取消任务后不再产生新输出，状态为 `cancelled`。
5. 重启服务后历史任务仍可查看和下载。
6. 单文件失败不影响同批其他文件，失败原因可见。
