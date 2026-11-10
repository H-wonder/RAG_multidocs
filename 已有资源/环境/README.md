# 环境安装入口

项目使用**两个职责分离的 conda 环境**：

| 环境 | 默认名 | 职责 |
| --- | --- | --- |
| 平台环境 | `rag` | CLI/Core/Connector/索引/检索 + pymilvus + 嵌入 + 测试 |
| 解析环境 | `docparse` | docparse、PyMuPDF、PaddleX、OCR 相关依赖 |

平台通过子进程调用解析环境，不 import docparse；两边依赖互不污染。

完整说明、外部服务和排错见 [../文档/07-环境搭建与依赖.md](../文档/07-环境搭建与依赖.md)。

## 一键安装

前置：Miniconda/Anaconda、Docker、Ollama；建议 20GB 可用磁盘、8GB 以上内存。

```bash
cd 已有资源
bash 环境/setup.sh
```

脚本会：

1. 创建平台环境 `rag`（Python 3.11），安装 torch + `requirements-rag.txt` + `requirements-core.txt`；
2. 创建解析环境 `docparse`（Python 3.11），安装 `requirements-docparse.txt` + paddle/paddlex；
3. 分别执行 `pip check`、import 验证，并跑一遍核心测试。

自定义环境名或 CUDA：

```bash
PLATFORM_ENV=my-rag PARSER_ENV=my-docparse \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash 环境/setup.sh
```

## 手动安装

```bash
# 平台环境
conda create -y -n rag -c conda-forge python=3.11
conda activate rag
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r 环境/requirements-rag.txt
pip install -r 环境/requirements-core.txt
pip check

# 解析环境
conda create -y -n docparse -c conda-forge python=3.11
conda activate docparse
pip install -r 环境/requirements-docparse.txt
pip install paddlepaddle==3.3.1
pip install "paddlex[cv]==3.7.2" shapely==2.2.0
pip check
```

## 配置

```bash
cp 环境/.env.example .env
# 关键：DOCPARSE_PYTHON 指向解析环境的 Python
DOCPARSE_PYTHON=$(conda run -n docparse which python)
# DOCPARSE_REPO 指向包含 docparse/ 包的目录
```

平台运行时只激活 `rag`；docparse 由平台通过子进程启动，不需要手动激活。

## 启动外部服务

```bash
# Milvus（Docker）
docker start milvus-standalone     # 首次部署见文档 07 第 4.2 节

# Ollama（OCR 模型）
bash 可选组件/docparse/scripts/start_ollama.sh
```

## 验证

```bash
bash 环境/check_env.sh             # 只检查，不安装

conda activate rag
cd 代码
python -m pytest -q                # 37 passed
cd ../样例 && python demo_rag.py   # 离线闭环：接入→切块→索引→检索
python demo.py                     # 只看切块结果

# 解析环境单独验证
conda activate docparse
PYTHONPATH=../可选组件/docparse python -m docparse doctor
```

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `setup.sh` | 一键创建两个环境并安装依赖 |
| `check_env.sh` | 检查工具、环境、依赖和服务状态，不安装 |
| `requirements-rag.txt` | 平台环境依赖（Milvus + 嵌入） |
| `requirements-core.txt` | 平台测试工具（pytest/ruff/jsonschema） |
| `requirements-docparse.txt` | 解析环境依赖 |
| `milvus/` | Docker 部署用的 `embedEtcd.yaml` / `user.yaml` |
| `.env.example` | 配置模板，复制为项目根目录 `.env` |
