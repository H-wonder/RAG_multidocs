#!/usr/bin/env bash
# 环境自检：只检查，不安装、不启动服务。
#
# 用法：
#   bash 环境/check_env.sh
#   PLATFORM_ENV=my-rag PARSER_ENV=my-docparse bash 环境/check_env.sh
#
# 退出码：
#   0  基础检查通过（外部服务未启动只给 WARN）
#   1  conda / 两个环境 / 核心依赖缺失
set -u

PLATFORM_ENV="${PLATFORM_ENV:-rag}"
PARSER_ENV="${PARSER_ENV:-docparse}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

pass=0
warn=0
fail=0

ok()   { echo "  PASS  $*"; pass=$((pass + 1)); }
warn() { echo "  WARN  $*"; warn=$((warn + 1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail + 1)); }

echo "== 1. 基础工具 =="
command -v conda >/dev/null 2>&1 && ok "conda" || bad "conda 未安装"
command -v docker >/dev/null 2>&1 && ok "docker" || warn "docker 未安装（Milvus 无法运行）"
command -v ollama >/dev/null 2>&1 && ok "ollama" || warn "ollama 未安装（扫描件 OCR 不可用）"

echo "== 2. conda 环境 =="
if command -v conda >/dev/null 2>&1; then
  envs="$(conda env list 2>/dev/null | awk 'NR>2 {print $1}')"
  echo "$envs" | grep -qx "$PLATFORM_ENV" && ok "平台环境 $PLATFORM_ENV" || bad "缺少平台环境 $PLATFORM_ENV（运行 环境/setup.sh）"
  echo "$envs" | grep -qx "$PARSER_ENV" && ok "解析环境 $PARSER_ENV" || bad "缺少解析环境 $PARSER_ENV（运行 环境/setup.sh）"
fi

echo "== 3. 平台依赖 =="
if conda run -n "$PLATFORM_ENV" python -c "import pymilvus, sentence_transformers" >/dev/null 2>&1; then
  ok "pymilvus + sentence-transformers"
else
  bad "平台环境缺少 pymilvus/sentence-transformers"
fi
if conda run -n "$PLATFORM_ENV" python -c "import pytest" >/dev/null 2>&1; then
  ok "pytest"
else
  warn "平台环境缺少 pytest（pip install -r 环境/requirements-core.txt）"
fi

echo "== 4. 解析依赖 =="
if conda run -n "$PARSER_ENV" python -c "import paddle, fitz" >/dev/null 2>&1; then
  ok "paddle + PyMuPDF"
else
  bad "解析环境缺少 paddle/PyMuPDF"
fi

echo "== 5. 项目代码 =="
if [ -f "$ROOT/代码/document_model.py" ] && [ -f "$ROOT/代码/keyword_index.py" ]; then
  ok "复用模块"
else
  bad "复用模块不完整"
fi
if conda run -n "$PLATFORM_ENV" python -m pytest -q "$ROOT/代码/tests" >/dev/null 2>&1; then
  ok "核心测试通过"
else
  warn "核心测试未运行（平台环境缺 pytest，或测试失败；激活 $PLATFORM_ENV 后手动运行 pytest 查看）"
fi

echo "== 6. 外部服务 =="
if command -v curl >/dev/null 2>&1; then
  if curl -sf --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama :11434"
  else
    warn "Ollama 未运行（bash 可选组件/docparse/scripts/start_ollama.sh）"
  fi
  if curl -sf --max-time 3 http://localhost:9091/healthz >/dev/null 2>&1; then
    ok "Milvus :9091"
  else
    warn "Milvus 未运行（docker start milvus-standalone）"
  fi
else
  warn "curl 未安装，跳过外部服务检查"
fi

echo
echo "结果：PASS=$pass WARN=$warn FAIL=$fail"
[ "$fail" -eq 0 ] || exit 1
