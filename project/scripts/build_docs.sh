#!/usr/bin/env bash
# 重新生成设计文档中的图和 DESIGN.docx。
# 依赖：graphviz(dot)、pandoc。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$HERE/../docs"

cd "$DOCS/diagrams"
dot -Tpng -Gdpi=150 architecture.dot -o architecture.png
dot -Tpng -Gdpi=150 flow.dot -o flow.png

cd "$DOCS"
pandoc DESIGN.md -o DESIGN.docx --resource-path=.
echo "generated: $DOCS/DESIGN.docx"
