from dedup import find_duplicate, hamming, simhash64

BASE = (
    "检索增强生成把参数化记忆与非参数化记忆结合：生成器负责产生答案，检索器从维基百科索引中"
    "取回相关段落；训练时联合优化检索器与生成器，在开放域问答、事实核查和摘要任务上都取得了稳定收益。"
    "实验表明，端到端微调比固定检索器更能适应不同任务。"
)


def test_same_text_has_zero_distance():
    assert hamming(simhash64(BASE), simhash64(BASE)) == 0


def test_near_duplicate_is_close():
    """改一个词应被判为近重复（实测距离约 8）。"""
    variant = BASE.replace("稳定收益", "明显提升")
    assert hamming(simhash64(BASE), simhash64(variant)) <= 10


def test_different_topics_are_far():
    other = "今天的天气晴朗，气温二十度，适合出门散步和骑车。"
    assert hamming(simhash64(BASE), simhash64(other)) > 20


def test_find_duplicate_picks_nearest_match():
    fingerprint = simhash64(BASE)
    existing = [("doc-a", "0" * 16), ("doc-b", fingerprint)]
    assert find_duplicate(fingerprint, existing, max_distance=6) == ("doc-b", 0)


def test_find_duplicate_respects_threshold():
    fingerprint = simhash64(BASE)
    other = simhash64("完全不同的内容" * 40)
    assert find_duplicate(fingerprint, [("doc-c", other)], max_distance=0) is None


def test_empty_text_is_safe():
    assert simhash64("") == "0" * 16
