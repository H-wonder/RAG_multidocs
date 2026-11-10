# 部署与运维

## 环境

```bash
conda create -y -n docparse -c conda-forge python=3.11
conda activate docparse
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## Region 模式（可选）

### 安装

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  paddlepaddle "paddlex[cv]" shapely
```

- 实测版本：`paddlepaddle 3.3.1` / `paddlex 3.7.2` / `opencv 4.10.0` / `shapely 2.2.0`（Python 3.11）
- 体积约 **1.1GB**：paddle 721MB、cv2 92MB、scipy 113MB、skimage 32MB
- `paddlex[cv]` 里的 `opencv-contrib-python` 是必须的；只装 `paddlex` 会在 `create_model()` 时报 `PDFReaderBackend` 缺失

### 模型权重（首次自动下载）

PP-DocLayoutV3 约 **126MB**，首次 `--mode region` 自动下载到：

```
~/.paddlex/official_models/PP-DocLayoutV3/
```

| 场景 | 做法 |
|---|---|
| 离线部署 | 提前把整个目录拷到目标机器同一路径 |
| 强制重新下载 | 删除该目录 |
| 模型源连通性检查很慢 | `export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |

### 设备配置

默认 **CPU**，把 8GB 显存留给 0.8B VLM。PP-DocLayoutV3 在 16 核 CPU 上约 0.5–1s/页。

```bash
# 单次
python -m docparse parse doc.pdf --mode region --layout-device gpu:0

# 全局
export DOCPARSE_LAYOUT_DEVICE=gpu:0
```

可选值：`cpu`（默认）、`gpu:0`、`gpu:1`（PaddleX 设备语法）。

### 验证

```bash
python -m docparse doctor
# layout  : PP-DocLayoutV3 available
```

### 行为

- `--mode region`：强制版面检测 + 逐区域 OCR
- `--mode auto`：整页优先；重复/截断/空输出等硬失败才升级 region
- 所有区域都失败：自动回退整页并告警，不会输出空文档
- 单区域失败：只丢弃该区域，其余照常合并

### 无害告警

| 输出 | 说明 |
|---|---|
| `No ccache found` | Paddle 的提示，不影响推理 |
| `faiss ... AVX2/AVX512` | PaddleX 启动日志，加载失败会回退普通 faiss |
| `Model files already exist` | 权重已缓存，正常 |

## Ollama

模型架构是 `qwen35`，**Ollama 必须 ≥ 0.35**。旧版会报：

```
unknown model architecture: 'qwen35'
```

启动（含上下限设置、模型检查）：

```bash
scripts/start_ollama.sh
```

关键参数：

| 变量 | 值 | 原因 |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | 16384 | 下限；实测模型自身申请 32768（`ollama ps` 可查）。默认 4096 装不下整页图像 + 输出 |
| `OLLAMA_NUM_PARALLEL` | 1 | 8GB 显存下避免 KV 缓存翻倍 |
| `OLLAMA_KEEP_ALIVE` | 30m | 避免频繁重载模型 |

模型：`longwayxu/xiaomi-ocr-0:bf16`（社区 BF16 GGUF，含视觉投影器，约 1.8GB）。

> 社区量化版（Q4/Q8）可以用，但上游未评测，准确率不等同于官方基准。

## 使用

```bash
python -m docparse doctor

# 通用
python -m docparse parse 文件.pdf 扫描件.jpg -o output

# 票据 KIE
python -m docparse parse 发票.jpg --fields "发票号码,开票日期,价税合计"

# 评测（A/B mode、DPI、后端版本）
python -m docparse eval samples/ --refs refs/ --mode auto -o eval_report.json
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `auto` | `auto`=page 优先，失败升级 region；`region`=强制版面分块；`page`=强制整页 |
| `--layout-device` | `cpu` | 版面检测设备：`cpu` / `gpu:0`；也可用 `DOCPARSE_LAYOUT_DEVICE` |
| `--dpi` | 200 | 扫描页渲染 DPI。page 模式超过约 176 DPI 会被压到 2048 长边；region 模式裁块才能吃到高 DPI 收益 |
| `--max-pages` | 全部 | 调试用 |
| `--fields` | 无 | 逗号分隔的 KIE 字段 |
| `--format` | `md,json` | 输出格式：`md,json,html,txt` |
| `--no-retry` | 关 | 关闭自动重复重试 |

## Web 界面

```bash
python -m docparse web                 # http://127.0.0.1:8787
python -m docparse web --port 9000
```

- 任务目录：`~/.docparse/jobs/<id>/`；`DOCPARSE_HOME` 可改根目录
- 上传上限：`DOCPARSE_MAX_UPLOAD_MB`（默认 200MB）
- 单 worker：Ollama 单并发；取消在页边界生效
- 重启后未完成任务标为 `interrupted`，可在界面重试
- 仅监听 `127.0.0.1`，不要直接暴露到公网（无鉴权）

## 本机实测（RTX 4060 Laptop 8GB）

page vs region（官方样例，相似度越高越好 / CER 越低越好）：

| 样例 | page 相似度 | page CER | region 相似度 | region CER | 建议 |
|---|---:|---:|---:|---:|---|
| receipt.jpg | 1.000 | 0.000 | 0.999 | 0.002 | 均可 |
| dense-equations.jpg | 0.972 | 0.056 | **0.986** | **0.028** | region |
| handwrite-formula.jpg | **0.932** | **0.133** | 0.864 | 0.253 | page |
| qianziwen.jpeg（书法） | 0.846 | 0.308 | 0.846 | 0.308 | page |
| **均值** | **0.938** | **0.124** | 0.924 | 0.148 | — |

结论与上游建议一致：**印刷表格/公式用 region，手写/书法用 page**。region 会多一次 CPU 版面检测：

- 扫描表格页：7.0s（region）
- 收据：3.2s（page） / 8.2s（region）
- 密集公式页：~8s（page） / 23.1s（region）

其他路径：

- 矢量 PDF → 原生抽取 **0.4–0.5s/页**，不经过模型
- 扫描 PDF → 3.9s/页
- 矢量表格 → `find_tables` 直接出 HTML
- 扫描表格 → 模型 OTSL → HTML，单元格全部正确
- 混合 PDF（矢量页 + 扫描页）→ 逐页正确路由
- docx → 原生解析，标题/段落/表格保留

稳态解码约 **100 tok/s**；首次请求约 50s（含模型加载，之后常驻显存）。

## 排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `unknown model architecture: 'qwen35'` | Ollama < 0.35 | 升级 Ollama |
| 请求报 `maximum context length` | 整页图像 + 输出超过上下文 | 代码会自动减半输出预算；仍失败则 `--mode region` |
| 首次请求 50s+ | 模型加载 | 正常，`OLLAMA_KEEP_ALIVE=30m` 保持常驻 |
| 输出重复 | 小模型长输出的已知问题 | 自动重试一次；仍失败会升级 region 并标 high |
| `layout: not installed` | 未装 PaddleX | 见上文 [Region 模式](#region-模式可选)：`paddlepaddle` + `paddlex[cv]` + `shapely` |
| pip 安装卡住 | `~/.cache/pip` 缓存损坏/过大 | `pip cache purge` 或加 `--no-cache-dir` |
| 显存不足 | 并发/上下文过大 | `OLLAMA_NUM_PARALLEL=1`，不要同时跑其他大模型 |

## 开机自启

`scripts/start_ollama.sh` 是手动启动方式，模型在 `~/.ollama`。系统自带的 systemd 服务以 `ollama` 用户运行、模型目录不同，**不要直接启用**，否则会重新下载模型。如需自启，建议加用户级 systemd 服务并把 `OLLAMA_MODELS` 指到当前用户目录。
