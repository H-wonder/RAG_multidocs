"""Explainable finite evidence requirements, not a semantic fact judge."""
from dataclasses import dataclass, asdict
import re

METRICS = {
    "accuracy": ("准确率", "accuracy"),
    "recall": ("召回率", "recall"),
    "precision": ("精确率", "precision"),
    "f1": ("F1", "f1-score"),
    "missed": ("漏判数", "漏判"),
    "false_positive": ("误报数", "误报"),
    "cost": ("费用", "金额", "成本", "cost", "amount"),
}
ALIASES = {"SVM": ("SVM", "支持向量机"), "逻辑回归": ("逻辑回归", "LogReg", "Logistic Regression")}


def compact(value):
    return re.sub(r"\s+", "", str(value)).casefold()


def matches(label, text):
    for alias in ALIASES.get(label, (label,)):
        if alias.isascii():
            pattern = re.escape(alias).replace(r"\ ", r"\s*")
            if re.search(r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])", str(text), re.I):
                return True
        elif compact(alias) in compact(text):
            return True
    return False


@dataclass(frozen=True)
class EvidenceNeed:
    label: str
    object: str = ""
    metric: str = ""
    kind: str = "topic"
    peers: tuple[str, ...] = ()

    def as_dict(self):
        return asdict(self)


def build_needs(query):
    metric_keys = [key for key, aliases in METRICS.items() if any(compact(a) in compact(query) for a in aliases)]
    if re.search(r"比较|对比|区别|异同|相差|差值|合计|compare|difference|sum|versus|\bvs\b", query, re.I):
        prefix = re.split(r"的|在|with|for", re.sub(r"^(?:请)?(?:比较|对比|计算|求|compare)\s*", "", query, flags=re.I), maxsplit=1)[0]
        for aliases in METRICS.values():
            for alias in aliases:
                prefix = re.split(re.escape(alias), prefix, maxsplit=1, flags=re.I)[0]
        objects = [p.strip(" ,，。？?：:的") for p in re.split(r"与|和|及|、|\band\b|versus|\bvs\b", prefix, flags=re.I) if p.strip(" ,，。？?：:的")]
        if len(objects) >= 2:
            if metric_keys:
                return [EvidenceNeed(f"{obj} {METRICS[key][0]}", obj, key, "relation", tuple(objects[:4])) for obj in objects[:4] for key in metric_keys[:4]]
            return [EvidenceNeed(obj, obj, "", "object_data", tuple(objects[:4])) for obj in objects[:4]]
    tasks = [s.strip() for s in re.split(r"[？?；;]", query) if s.strip()]
    return [EvidenceNeed(task) for task in tasks[:8]] or [EvidenceNeed(query)]


def numeric(value, metric):
    value = str(value).strip()
    if re.search(r"\d\s+\d|[%％]\s*\.", value):
        return False
    if not re.search(r"(?<![\d.])-?\d+(?:\.\d+)?", value):
        return False
    if metric in {"accuracy", "recall", "precision", "f1"}:
        return bool(re.search(r"\d+(?:\.\d+)?\s*[%％]|\b0\.\d+\b", value))
    return True


def support(hit, need):
    """Recognized row/column or explicit local numeric relation only.

    A topic match remains a topic signal and cannot satisfy a numeric relation.
    Missing/ambiguous relations stay unknown, even if all names occur somewhere.
    """
    if isinstance(need, str):
        return {"supported": False, "kind": "legacy_topic_only"}
    if need.kind == "topic":
        return {"supported": False, "kind": "topic_unverified"}
    aliases = METRICS.get(need.metric, ())
    cells = hit.get("semantic_cells", hit.get("metadata", {}).get("semantic_cells", []))
    for cell in cells:
        row, column = cell.get("row", ""), cell.get("column", "")
        metric_match = not aliases or any(compact(a) in compact(column) for a in aliases)
        alternate = not aliases or any(compact(a) in compact(row) for a in aliases)
        if ((matches(need.object, row) and metric_match) or (matches(need.object, column) and alternate)) and numeric(cell["value"], need.metric):
            return {"supported": True, "kind": "mapped_cell_relation", "value": cell["value"], "quote": cell["quote"], "object": need.object, "metric": need.metric}
    # Sentence-local scan: never collect an object's name in one sentence and
    # another object's metric elsewhere. Multiple explicit subjects are ambiguous.
    for sentence in re.split(r"[。；;\n]", hit["text"]):
        if not matches(need.object, sentence):
            continue
        if need.kind == "object_data" and numeric(sentence, ""):
            return {"supported": True, "kind": "explicit_object_data", "quote": sentence}
        # An explicit metric must be followed by its own numeric value; don't
        # assign another metric's number simply because both occur in a passage.
        for clause in re.split(r"[，,]", sentence):
            if not matches(need.object, clause):
                continue
            if sum(matches(obj, clause) for obj in need.peers) > 1:
                continue
            for alias in aliases:
                pattern = re.escape(alias) + r"[^\d。；;，,]{0,10}(-?\d+(?:\.\d+)?\s*(?:[%％]|元|万元|CNY|USD|ms)?)"
                found = re.search(pattern, clause, re.I)
                if found and numeric(found[1], need.metric):
                    return {"supported": True, "kind": "explicit_text_relation", "value": found[1], "quote": clause, "object": need.object, "metric": need.metric}
    return {"supported": False, "kind": "relation_unknown"}


def covered(hits, needs):
    return {i for i, need in enumerate(needs) if any(support(h, need)["supported"] for h in hits)}


def directed_queries(needs):
    grouped = {}
    topics = []
    for need in needs:
        if need.kind in {"relation", "object_data"}:
            grouped.setdefault(need.object, [])
            if need.metric:
                grouped[need.object].append(METRICS[need.metric][0])
        else:
            topics.append(need.label)
    return [obj + " 结果 " + " ".join(dict.fromkeys(metrics)) + " 依据" for obj, metrics in grouped.items()] + topics
