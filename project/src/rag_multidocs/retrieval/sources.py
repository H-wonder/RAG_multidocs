"""Active-source constraints from exact paths, names, and catalog-derived titles."""
import re
from pathlib import PurePath

_LEAD = re.compile(r"根据|依据|结合|综合|参照|对照|参考|对比|比较|查看|读取|仅用|来自|总结|概括|归纳|\bfrom\b", re.I)
_JOIN = re.compile(r"^\s*[、，,]?\s*(?:以及|和|与|及)?\s*$")
_DATE = re.compile(
    r"(?<![\d.])((?:19|20)\d{2})(?!\d)(?=\s*(?:年|[./-]\d|"
    r"(?:以及|[和与、至到])\s*(?:19|20)\d{2}年))"
    r"|(?i:\b(?:in|during|as of)\s+)((?:19|20)\d{2})\b"
)


def mentions(query, name):
    if not name:
        return []
    # ASCII filename boundaries prevent scanned.pdf matching multi_scanned.pdf.
    return list(re.finditer(r'(?<![A-Za-z0-9_.-])' + re.escape(name) +
                            r'(?![A-Za-z0-9_.-])', query))


def resolve_sources(query, documents, doc_ids=None):
    aliases = {}
    known_suffixes = {PurePath(d.get('source', '')).suffix.casefold() for d in documents}
    for doc in documents:
        for name in {doc.get('source', ''), PurePath(doc.get('source', '')).name}:
            if name:
                aliases.setdefault(name, set()).add(doc['doc_id'])
        # A full quoted title/stem is an explicit identity; unquoted topics,
        # suffixes and semantic field matches must never restrict retrieval.
        for title in {doc.get('title', ''), PurePath(doc.get('source', '')).stem}:
            if title:
                for left, right in [('《', '》'), ('“', '”'), ('"', '"')]:
                    aliases.setdefault(left + title + right, set()).add(doc['doc_id'])
    matched, spans, issues = [], [], []
    candidates = [(m.start(), m.end(), name, False) for name in aliases for m in mentions(query, name)]
    for start, end, name, short in sorted(candidates, key=lambda m: (m[0], -(m[1] - m[0]), m[3])):
        if any(start < b and a < end for a, b in spans):
            continue
        left, right = query[:start], query[end:]
        prefix = re.search(r"[\u3400-\u9fff]+$", left)
        lead = any(not left[m.end():].strip(" \t\"“《") for m in _LEAD.finditer(left))
        lead |= bool(re.search(r"(?:^|[^存所现])(?:在|从)\s*$", left) and re.match(r"(?:中|里|所载)", right))
        lead |= bool(spans and _JOIN.fullmatch(query[spans[-1][1]:start]))
        quoted = bool((left.endswith('"') and right.startswith('"')) or
                      (left.endswith("“") and right.startswith("”")) or
                      (left.endswith("《") and right.startswith("》")))
        if short and not (lead or quoted):
            continue
        spans.append((start, end))
        if prefix and not (short or lead or quoted or re.fullmatch(r"(?:请)?(?:在|从|与|和|及|文档|文件)", prefix[0])):
            issues.append({'code': 'SOURCE_NAME_BOUNDARY_UNCERTAIN', 'name': name,
                           'message': 'Filename may be part of a different adjacent Chinese name; use an exact quoted name or path'})
        ids = sorted(aliases[name])
        matched.append({'name': name, 'doc_ids': ids, 'span': [start, end]})
        if len(ids) != 1:
            issues.append({'code': 'SOURCE_AMBIGUOUS', 'name': name,
                           'message': 'Source name identifies multiple active documents', 'doc_ids': ids})
    # Unknown explicit filenames/quoted document titles must not silently fall
    # back to another document. An extension is syntactic, not a format switch.
    patterns = [r'(?<![\w.-])[\w][\w.-]*\.[A-Za-z][A-Za-z0-9]{0,9}(?![A-Za-z0-9_.-])',
                r'《[^》]+》']
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            if not any(match.start() < b and a < match.end() for a, b in spans):
                quoted = match.group().startswith('《') or bool(re.search(r'["“《][^"”》]*' + re.escape(match.group()) + r'[^"”》]*["”》]', query))
                file_hint = bool(re.search(r'文件|文档|来源|\bfile\b|\bsource\b', query, re.I))
                api_tail = bool(re.match(r'\s*(?:\(|方法|函数|属性|接口|调用|\bmethod\b|\bfunction\b|\battribute\b)', query[match.end():], re.I))
                # A dot is not sufficient source intent: model.fit/np.array
                # are ordinary technical questions. Unknown arbitrary extensions
                # require quoting or an explicit file/source phrase. Extensions
                # observed in the active directory are recognized, not a fixed
                # list tied to concrete Connector implementations.
                suffix = PurePath(match.group()).suffix.casefold()
                if not quoted and (api_tail or not (file_hint or suffix in known_suffixes)):
                    continue
                issues.append({'code': 'SOURCE_NOT_FOUND', 'name': match.group(),
                               'message': 'No reliable active source identity'})
    inferred = sorted({did for m in matched for did in m['doc_ids']})
    effective = inferred if matched else doc_ids
    if matched and doc_ids is not None:
        effective = [did for did in inferred if did in doc_ids]
        if set(inferred) - set(effective):
            issues.append({'code': 'SOURCE_SCOPE_CONFLICT', 'message': 'Named source outside explicit doc_ids scope'})
    return {'original_query': query, 'explicit': bool(matched or issues or doc_ids is not None),
            'matches': sorted(matched, key=lambda m: m['span'][0]),
            'doc_ids': effective, 'blocked': bool(issues), 'warnings': issues}


def task_doc_ids(task, constraint):
    matches = constraint.get('matches', [])
    ids = {did for m in matches if mentions(task, m['name']) for did in m['doc_ids']}
    # Elliptical second questions inherit a single explicit document only.
    all_ids = {did for m in matches for did in m['doc_ids']}
    return ids or (all_ids if len(all_ids) == 1 else set())


def without_source_names(query, constraint):
    chars = list(query)
    for match in constraint.get("matches", []):
        # Subquestions do not share the full question's character offsets.
        for occurrence in mentions(query, match['name']):
            start, end = occurrence.span()
            if start and end < len(query) and (query[start - 1], query[end]) in {('《', '》'), ('“', '”'), ('"', '"')}:
                start, end = start - 1, end + 1
            chars[start:end] = " " * (end - start)
    return "".join(chars)


def fact_years(query, constraint):
    """Years used by source identity are already resolved by the catalog."""
    return list(dict.fromkeys(m[1] or m[2] for m in _DATE.finditer(without_source_names(query, constraint))))


def labeled_record_source(query, candidates):
    """Resolve factual record queries through actual labels, never filenames.

    A unique labeled record takes precedence over lists that merely mention
    the same topic. Multiple matching records remain unresolved.
    """
    if not re.search(r'什么|多少|哪|何时|几点', query) or re.search(r'推荐|建议|分析|比较|对比|总结|概括|综合', query):
        return None
    native, tables = {}, {}
    questions = [part for part in re.split(r'[？?；;]', query) if part.strip()]
    part_sources = [set() for _ in questions]
    for hit in candidates:
        labels = []
        if hit.get('content_type') == 'table':
            labels = hit.get('table_headers', hit.get('metadata', {}).get('table_headers', []))
        else:
            labels = [match[1] for line in hit['text'].splitlines()
                      if (match := re.fullmatch(r'\s*([^:：\n]{1,40})[:：]\s*\S.*', line))]
        matched = set()
        for label in labels:
            label = label.split(' > ')[-1].strip()
            aliases = [part.strip() for part in re.split(r'[（()）]', label) if part.strip()]
            aliases += [re.sub(r'(?:类别|类型)$', '', label)]
            for alias in aliases:
                if len(alias) >= 2 and ((alias.isascii() and re.search(r'(?<![A-Za-z])' + re.escape(alias) + r'(?![A-Za-z])', query, re.I))
                                       or (not alias.isascii() and alias in query)):
                    matched.add(label)
                    for number, question in enumerate(questions):
                        if (re.search(r'(?<![A-Za-z])' + re.escape(alias) + r'(?![A-Za-z])', question, re.I)
                                if alias.isascii() else alias in question):
                            part_sources[number].add(hit['doc_id'])
        if matched:
            pool = tables if hit.get('content_type') == 'table' else native
            pool.setdefault(hit['doc_id'], set()).update(matched)
    if len(native) == 1:
        did, labels = next(iter(native.items()))
        if any(sources and did not in sources for sources in part_sources):
            return None
        return {'doc_id': did, 'labels': sorted(labels), 'basis': 'unique_literal_labeled_record'}
    if native:
        return None
    qualifying = {did: labels for did, labels in tables.items() if len(labels) >= 2}
    if len(qualifying) == 1:
        did, labels = next(iter(qualifying.items()))
        return {'doc_id': did, 'labels': sorted(labels), 'basis': 'unique_matching_table_schema'}
    return None
