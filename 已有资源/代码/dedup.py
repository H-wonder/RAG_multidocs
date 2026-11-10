"""跨文档近重复检测（64 位 SimHash，纯标准库）。

来源：``任务二/docflow/dedup.py``，逻辑未改，仅补充文件级入口和更完整注释。

为什么需要：同一份资料经常以不同文件名、不同版本重复入库，直接索引会产生
重复片段，检索时同一段话反复出现，还会虚增知识库规模。

做法：把文本切成 3-gram，每个 gram 哈希成 64 位，按词频加权求和得到指纹。
实测：改一个词约 8 位，加一句话约 15 位，无关文档 30 位以上，所以默认阈值取 10。
注意阈值取决于语料与语言，上线前应在自己的样例上校准。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[a-z0-9]+")


def _shingles(text: str, size: int = 3) -> list[str]:
    text = text.lower()
    tokens = _CJK.findall(text) + _WORD.findall(text)
    if len(tokens) < size:
        return ["".join(tokens)] if tokens else []
    return ["".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def simhash64(text: str, shingle_size: int = 3) -> str:
    """返回 16 位十六进制的 SimHash 指纹；空文本返回全 0。"""
    weights = [0] * 64
    grams = _shingles(text, shingle_size)
    if not grams:
        return "0" * 16

    counts: dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1

    for gram, count in counts.items():
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(64):
            weights[bit] += count if (value >> bit) & 1 else -count

    fingerprint = 0
    for bit in range(64):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return f"{fingerprint:016x}"


def hamming(a: str, b: str) -> int:
    """两个十六进制指纹的汉明距离；非法输入按最大距离 64 处理。"""
    if not a or not b or len(a) != len(b):
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def find_duplicate(
    fingerprint: str,
    existing: list[tuple[str, str]],
    max_distance: int = 10,
) -> tuple[str, int] | None:
    """在 [(doc_id, simhash), ...] 里找最近的一个；超过阈值返回 None。

    只应把“已成功入库”的文档作为基准。把 DUPLICATE / FAILED 的指纹也纳入比较，
    会让两份重复文件互相判重，最后一份都进不了库。
    """
    best: tuple[str, int] | None = None
    for doc_id, other in existing:
        if not other:
            continue
        distance = hamming(fingerprint, other)
        if distance <= max_distance and (best is None or distance < best[1]):
            best = (doc_id, distance)
    return best


def simhash_file(path: str | Path, shingle_size: int = 3) -> str:
    """对文本文件计算 SimHash；二进制文件请先转成文本再调用。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return simhash64(text, shingle_size)


__all__ = ["find_duplicate", "hamming", "simhash64", "simhash_file"]
