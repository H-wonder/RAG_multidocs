"""Test wrapper over the standalone, reusable fixture authoring utility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rag_samples import build  # noqa: E402

if __name__=='__main__':
    build(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '128.50')
