"""Resolve explicit field questions against literal label/value lines."""
from dataclasses import dataclass
import re

from ..retrieval.sources import mentions, task_doc_ids

# An OCR line number is display metadata. Keep it in the original excerpt.
LABELED_LINE = re.compile(r"^\s*(?:\d+[.、)]?\s+)?([^:：\n]{1,40})[:：]\s*(\S.*?)\s*$")
QUESTION_END = re.compile(r"(?:分别|各自|各)?\s*(?:是多少|是什么|为何值|为多少|有多少|是几点|是多少呢|是多久|有多大)[？?。！!\s]*$")
FIELD_SEPARATOR = re.compile(r"\s*(?:以及|[、，,和与及])\s*")
SOURCE_JOINERS = r"(?:^\s*(?:(?:请)?(?:比较|对比|根据|在|从)|以及|和|与|及|、|的|中|里|[，,：:\s])+)"


def canonical(value):
    return "".join(value.split()).casefold()


@dataclass(frozen=True)
class FieldLine:
    index: int
    line: int
    source: str
    label: str
    value: str
    quote: str

    @property
    def line_id(self):
        return f"L{self.index}_{self.line}"


@dataclass(frozen=True)
class FieldTask:
    task_id: str
    field: str
    source: str
    line: FieldLine | None
    entity_line: FieldLine | None = None


def read_fields(evidence):
    records = []
    for index, hit in enumerate(evidence, 1):
        lines = []
        if hit.get("content_type") != "table":
            for line, quote in enumerate(hit["text"].splitlines(), 1):
                match = LABELED_LINE.fullmatch(quote)
                if match:
                    label, value = match.groups()
                    if any(char.isalpha() for char in label):
                        lines.append(FieldLine(index, line, hit.get("source", ""), label.strip(), value, quote))
        # A single colon in prose does not establish a record.
        if len(lines) >= 2:
            records.append((hit, lines))
    return records


def entity_anchor(lines, entity):
    """Use a literal identity field to bind a requested named entity."""
    return next((line for line in lines if canonical(line.value) == canonical(entity)
                 and canonical(line.label).endswith(("名称", "姓名", "编号", "标识", "型号", "name", "id"))), None)


def field_matches(requested, label, entity):
    requested, label = canonical(requested), canonical(label)
    return label == requested or (
        len(requested) >= 2 and re.fullmatch(r"[\u3400-\u9fff]+", requested) is not None
        and label.endswith(requested)
        and canonical(entity).endswith(label[:-len(requested)])
    )


def plan_labeled_fields(question, evidence, constraint):
    records = read_fields(evidence)
    if not records:
        return []
    tasks = []
    any_match = False
    for segment in filter(str.strip, re.split(r"[？?；;]+", question)):
        segment = segment.strip()
        end = QUESTION_END.search(segment)
        if not end:
            return []
        body = segment[:end.start()]
        ids = task_doc_ids(segment, constraint)
        names = [m["name"] for m in constraint.get("matches", []) if mentions(segment, m["name"])]
        for name in sorted(names, key=len, reverse=True):
            for wrapped in (f'"{name}"', f"“{name}”", name):
                body = body.replace(wrapped, " ")
        if names:
            body = re.sub(SOURCE_JOINERS, "", body, count=1).strip()
        entity, separator, fields = body.partition("的")
        if not separator:
            entity, fields = "", body
        coordinated = any(word in segment for word in ("分别", "各自", "各是", "各有"))
        requested = [f.strip() for f in FIELD_SEPARATOR.split(fields) if f.strip()] if coordinated else [fields.strip()]
        if not requested or any(not f or len(f) > 28 or "的" in f for f in requested):
            return []
        scopes = sorted(ids) if ids else [None]
        for source_id in scopes:
            eligible = [(hit, lines, entity_anchor(lines, entity) if entity else None)
                        for hit, lines in records
                        if source_id is None or hit.get("doc_id", hit.get("source")) == source_id]
            source = next((h.get("source", "") for h in evidence
                           if source_id is not None and h.get("doc_id", h.get("source")) == source_id), "")
            for field in requested:
                candidates = [(line, anchor, len({canonical(x.label) for x in lines}) == len(lines))
                              for _, lines, anchor in eligible if not entity or anchor
                              for line in lines if field_matches(field, line.label, entity)]
                # A record naming the requested entity also establishes when a
                # field is missing. Do not ask the model to infer it elsewhere.
                any_match |= bool(candidates) or bool(entity and any(anchor for _, _, anchor in eligible))
                line, anchor = None, None
                if len(candidates) == 1 and (not entity or candidates[0][2]):
                    line, anchor, _ = candidates[0]
                tasks.append(FieldTask(f"T{len(tasks) + 1}", field, source, line, anchor))
    # Use the usual prose/table path for queries outside the labeled-field form.
    return tasks if any_match else []
