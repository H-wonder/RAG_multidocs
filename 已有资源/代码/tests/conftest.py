import sys
from pathlib import Path

# 让测试直接 import 上一级目录里的复用模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
