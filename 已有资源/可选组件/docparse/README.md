# docparse

本地多模态文档解析系统：**Xiaomi-OCR-0 (0.8B) + Ollama**，面向扫描件、发票/票据、合同、论文、Office 文档。
核心目标：**尽量不依赖人工复核**——可检测的失败模式自动重试/升级，最终以结构化告警输出。

运行时自包含：上游需要的 5 个后处理模块已 vendor 到 `docparse/vendor/`，**不需要克隆 Xiaomi-OCR-0 也能跑**。

## 仓库结构

```
docparse/                     # 项目根（git 仓库）
├── docparse/                 # Python 包（全部业务代码）
│   ├── cli.py                # 命令行：parse / doctor / eval
│   ├── pipeline.py           # 分诊、OCR、自动升级阶梯
│   ├── ir.py                 # 文档中间表示（IR）
│   ├── pdfio.py              # PDF 探测、原生文本/表格抽取、渲染
│   ├── backend.py            # Ollama（OpenAI 兼容）客户端
│   ├── imageops.py           # 图像归一化（EXIF、像素上下限）
│   ├── layout.py             # PP-DocLayoutV3 版面检测（可选）
│   ├── office.py             # docx/pptx/xlsx/txt 原生解析
│   ├── quality.py            # 质量门、KIE 幻觉回查
│   ├── writers.py            # IR → Markdown / JSON / HTML / TXT
│   ├── evaluate.py           # 与参考 Markdown 对比（相似度/CER）
│   ├── web/                  # 本地 Web UI（FastAPI + 单页）
│   │   ├── app.py            # API 路由
│   │   ├── jobs.py           # 持久化任务队列（单 worker）
│   │   └── static/           # index.html / app.js / style.css
│   └── vendor/               # 上游 vendored 代码（Apache-2.0）
├── docs/
│   ├── architecture.md       # 数据流、分诊、升级阶梯、IR
│   ├── quality-gates.md      # 告警码、重复检测、KIE 回查、调参原则
│   ├── operations.md         # 环境、Ollama、参数、实测、排障
│   ├── output-formats.md     # 当前/计划输出格式与 IR schema
│   ├── web-ui.md             # Web 界面设计、API、P0 状态
│   ├── roadmap.md            # P1–P3 优化路线（验收/估算/依赖）
│   └── third-party.md        # vendored 来源与许可
├── scripts/
│   └── start_ollama.sh       # 启动 Ollama（上下文/并发/模型检查）
├── tests/                    # 不需要模型的单元测试
├── requirements.txt
├── pyproject.toml
└── README.md
```

上游 `Xiaomi-OCR-0/` 不放进本仓库。需要官方示例时，克隆到与项目**同级**的目录（`../Xiaomi-OCR-0`）。

## 模块地图

| 层 | 模块 | 职责 |
|---|---|---|
| 接口 | `cli.py` `web/app.py` `web/jobs.py` | CLI / Web API / 任务队列 |
| 编排 | `pipeline.py` | 按格式/页类型路由，执行重试与升级 |
| 接入 | `pdfio.py` `imageops.py` `office.py` | 把任意输入变成文本块或位图 |
| 推理 | `backend.py` `layout.py` | 调模型；版面检测 |
| 质量 | `quality.py` | 判定失败、决定重试、KIE 回查 |
| 表示 | `ir.py` `writers.py` | 统一 IR；序列化为 Markdown/JSON |
| 评测 | `evaluate.py` | 参考集回归 |
| 复用 | `vendor/` | 上游 OTSL/重复检测/合并 |

## 快速开始

```bash
conda create -y -n docparse -c conda-forge python=3.11
conda activate docparse
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

scripts/start_ollama.sh          # Ollama 需 >= 0.35
python -m docparse doctor

# 命令行
python -m docparse parse 文件.pdf 扫描件.jpg -o output --format md,json,html,txt

# Web 界面
python -m docparse web           # 打开 http://127.0.0.1:8787
```

票据字段抽取：

```bash
python -m docparse parse 发票.jpg --fields "发票号码,开票日期,价税合计"
```

## Region 模式（可选）

`region` 模式先用 PP-DocLayoutV3 做版面检测，再逐区域 OCR。印刷表格/密集公式更准，手写/书法更差。

### 1. 安装

```bash
conda activate docparse
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  paddlepaddle "paddlex[cv]" shapely
```

实测版本：`paddlepaddle 3.3.1` / `paddlex 3.7.2` / `opencv 4.10.0` / `shapely 2.2.0`（Python 3.11）。
安装体积约 **1.1GB**（paddle 721MB、cv2 92MB、scipy 113MB、skimage 32MB）。

### 2. 版面模型（首次自动下载）

PP-DocLayoutV3 权重约 **126MB**，首次调用自动下载到：

```
~/.paddlex/official_models/PP-DocLayoutV3/
```

离线环境可提前拷贝该目录；源连通性检查慢时：

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

### 3. 验证

```bash
python -m docparse doctor
# layout  : PP-DocLayoutV3 available
```

### 4. 设备配置

默认 **CPU** 推理，把 8GB 显存留给 VLM。切到 GPU：

```bash
python -m docparse parse doc.pdf --mode region --layout-device gpu:0
# 或全局
export DOCPARSE_LAYOUT_DEVICE=gpu:0
```

### 5. 使用与取舍

```bash
python -m docparse parse 表格.pdf --mode region
```

| 文档 | 建议 |
|---|---|
| 印刷表格、密集公式、多栏 | `region` |
| 手写、书法、古籍、不规则版面 | `page` |
| 不确定 | `auto`（整页优先，硬失败才升级 region） |

所有区域都失败时会自动回退整页并告警，不会输出空文档。详细排障见 [docs/operations.md](docs/operations.md)。

## 输出格式

CLI 用 `--format` 选择，Web 界面勾选；所有格式均以 IR 为源：

| 文件 | 内容 |
|---|---|
| `<name>.md` | Markdown；表格为 HTML `<table>`（OTSL 转换），公式为 LaTeX，多页用 `<!-- page N -->` 分隔 |
| `<name>.json` | 完整 IR：`pages[].blocks[]` 含 `type/bbox/reading_order/content/html/source/confidence/warnings`；`meta.kie` 放字段抽取 |
| `<name>.html` | 独立 HTML，内嵌样式，表格原样保留 |
| `<name>.txt` | 纯文本，HTML 表格转为制表符分隔 |

ZIP 由 Web 界面整批打包。暂不支持 DOCX / CSV / XLSX / 图片资源。
格式矩阵与 IR schema 见 [输出格式](docs/output-formats.md)。

## Web 界面

```bash
python -m docparse web          # 默认 http://127.0.0.1:8787
```

- 批量拖拽/多选，混合格式
- 模式 / DPI / 最大页数 / KIE 字段 / 输出格式
- 页级进度、取消、重试
- 任务持久化（`~/.docparse/jobs/`），重启后历史可查
- 结果预览（Markdown / HTML / JSON / TXT / 告警汇总）
- 单文件下载与整批 ZIP

单 worker 运行（Ollama 单并发），默认只监听 `127.0.0.1`。
设计与 API 见 [Web 界面与完善路线](docs/web-ui.md)。

## 文档

- [架构与设计](docs/architecture.md)
- [自动质量门](docs/quality-gates.md)
- [部署、参数与排障](docs/operations.md)
- [输出格式](docs/output-formats.md)
- [Web 界面](docs/web-ui.md)
- [路线图 P1–P3](docs/roadmap.md)
- [第三方代码与许可](docs/third-party.md)

## 实测（RTX 4060 Laptop 8GB）

官方样例相似度（page / region）：收据 1.000 / 0.999，密集公式 0.972 / **0.986**，手写公式 **0.932** / 0.864，书法 0.846 / 0.846。
印刷表格/公式用 region，手写用 page。矢量 PDF 原生抽取 **0.4–0.5s/页**；扫描页约 **4s/页**；稳态解码约 100 tok/s。
完整数据见 [docs/operations.md](docs/operations.md)。

## 已知边界

- Ollama 后端在上游属于 experimental，官方只在 Apple Silicon 做过一致性抽检
- 手写/书法/古籍建议 `--mode page`；表格密集的印刷件建议 `--mode region`
- 跨页表格合并、合并单元格 span 还原尚未实现
- 语义级错误（如把条码识别成单号）无法被字面回查发现
- `region` 模式需要额外安装 PaddleX + PP-DocLayoutV3（约 1.1GB，见上文 Region 模式）
- Web 界面为单用户、单 worker；bbox 叠加、DOCX/CSV 导出、鉴权仍在 P1

## 上游仓库（可选）

仓库不再依赖上游即可运行。仅在需要官方示例图/评测数据时，克隆到项目同级目录：

```bash
cd ..
git clone --depth 1 https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
```

它位于本仓库之外，不会进入版本控制。
