#!/usr/bin/env bash
# 一键创建两个 conda 环境：
#   rag       平台：CLI/Core/Connector/索引/检索 + pymilvus + 嵌入
#   docparse  解析：docparse + PyMuPDF + PaddleX
#
# 用法：
#   bash 环境/setup.sh
#   PLATFORM_ENV=my-rag PARSER_ENV=my-docparse bash 环境/setup.sh
#
# 有 CUDA 时通过 TORCH_INDEX_URL 指定官方 CUDA 索引：
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash 环境/setup.sh
#
# 说明：
# - 不修改系统 Python；脚本可重复执行，已存在的环境只补装依赖。
# - 两个环境分开验证；平台不 import docparse，通过子进程调用。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_ENV="${PLATFORM_ENV:-rag}"
PARSER_ENV="${PARSER_ENV:-docparse}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未找到 conda。请先安装 Miniconda/Anaconda。" >&2
  exit 1
fi

ensure_env() {
  local name="$1"
  if conda env list | awk 'NR>2 {print $1}' | grep -qx "$name"; then
    echo "环境 $name 已存在，将在其中补装依赖。"
  else
    echo "创建 conda 环境 $name (Python $PYTHON_VERSION) ..."
    conda create -y -n "$name" -c conda-forge "python=$PYTHON_VERSION"
  fi
}

# ---------------------------------------------------------------- 平台环境
ensure_env "$PLATFORM_ENV"
eval "$(conda shell.bash hook)"
conda activate "$PLATFORM_ENV"

python -m pip install --upgrade pip

if python -c "import torch" >/dev/null 2>&1; then
  echo "检测到已安装 torch，跳过 torch 安装。"
else
  echo "安装 torch（索引：$TORCH_INDEX_URL）..."
  python -m pip install "torch==2.6.0" --index-url "$TORCH_INDEX_URL"
fi

echo "安装平台依赖 ..."
python -m pip install -r "$HERE/requirements-rag.txt"
python -m pip install -r "$HERE/requirements-core.txt"

echo "检查平台依赖 ..."
python -m pip check
python - <<'PY'
import pymilvus
import sentence_transformers

print("pymilvus", pymilvus.__version__)
print("sentence-transformers", sentence_transformers.__version__)
PY

echo "运行核心测试 ..."
python -m pytest -q "$HERE/../代码/tests"

# ---------------------------------------------------------------- 解析环境
ensure_env "$PARSER_ENV"
conda activate "$PARSER_ENV"

python -m pip install --upgrade pip
echo "安装 docparse 依赖 ..."
python -m pip install -r "$HERE/requirements-docparse.txt"
python -m pip install "paddlepaddle==3.3.1"
python -m pip install "paddlex[cv]==3.7.2" "shapely==2.2.0"

echo "检查解析依赖 ..."
python -m pip check
python - <<'PY'
import fitz
import paddle

print("PyMuPDF", fitz.__doc__.splitlines()[0] if fitz.__doc__ else "ok")
print("paddle", paddle.__version__)
PY

cat <<EOF

安装完成。

  平台环境：$PLATFORM_ENV
  解析环境：$PARSER_ENV

下一步：

  conda activate $PLATFORM_ENV
  cp 环境/.env.example .env
  # 把 DOCPARSE_PYTHON 改成 $(conda run -n "$PARSER_ENV" which python)
  # 修改 DOCPARSE_REPO、MILVUS_URI 等

  cd 代码 && python -m pytest -q

Milvus 与 Ollama 按文档 07 第 4 节启动。
EOF
