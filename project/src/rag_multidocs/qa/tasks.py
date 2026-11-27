"""Finite, source-bound generation tasks; complex semantics remain unverified."""
import hashlib
import re

from .table_evidence import table_view, label_in
from ..retrieval.sources import task_doc_ids, without_source_names
from ..retrieval.rewrite import comparison_parts, summary_parts, question_parts
from ..retrieval.needs import METRICS, matches
from ..ingestion.tables import column_name

FIELD_ALIASES = {**METRICS, 'warranty': ('保修期', '保修期限', '保修', 'warranty period', 'warranty'),
                 'duration': ('开放多久', '开放时间', '可用时间', '每天开放', '每天', 'hours'),
                 'date': ('日期', 'date'), 'identifier': ('编号', 'receipt number', 'order number')}
# Amount, fee and cost are distinct fields, not interchangeable synonyms.
FIELD_ALIASES.update(amount=('金额', 'amount'), fee=('费用', 'fee'), cost=('成本', 'cost'))
FIELD_ALIASES['tax'] = ('税率', 'tax rate')
# Literal source formatting is preserved. Units are read, never inferred from a
# metric name. Percentage points remain distinct from percent.
VALUE = r'-?\d+(?:\.\d+)?\s*(?:percentage points?|百分点|months?|minutes?|seconds?|hours?|days?|CNY|USD|万元|元|毫秒|秒|分钟|小时|个月|天|年|%|％)?'


def synthesis_question(question):
    return bool(re.search(
        r"(?:^|[，,。；;\s]|请(?:问)?)综合|总结|归纳|概括|哪些|列举|"
        r"(?:^|[，,。；;\s]|请|并|如何|进行|对.+)分析|"
        r"(?:比较|对比)[^？?]*(?:异同|区别|优缺点|适用|特点)|推荐(?!单位|人|免试)|建议|撰写|"
        r"\b(?:summari[sz]e|synthesi[sz]e|analy[sz]e|recommend)\b",
        question, re.I,
    ))


def table_tasks(question, evidence):
    tasks, seen = [], set()
    for hit in evidence:
        if hit.get('content_type') != 'table':
            continue
        view = table_view(hit, question)
        for row in [r for r in view['rows'] if label_in(r['row_label'], question)]:
            row = {**row, 'cells': {**row['cells'], **row.get('empty_cells', {})}}
            for label in [c for c in row['cells'] if label_in(c, question, field_query=True)]:
                location = hit.get('location', hit.get('metadata', {}).get('location', {}))
                key = (hit.get('doc_id', hit.get('source', '')), row['row_label'], label, location.get('worksheet', ''))
                if key in seen:
                    continue
                seen.add(key)
                locator = f"工作表{location['worksheet']}，" if location.get('worksheet') else ''
                suffix = (f"本次仅回答“{label}”列的“{row['row_label']}”，不要回答其他列。"
                          if len(view['rows']) == 1 else
                          f"本次仅回答“{row['row_label']}”行、“{label}”列，不要回答其他行列。")
                metadata = hit.get('metadata', {})
                semantic = hit.get('semantic_cells', metadata.get('semantic_cells', []))
                mapped = next((c for c in semantic if c['row'] == row['row_label'] and c['column'] == label
                               and c['value'] == row['cells'][label] and c['quote'] == row['supporting_quote']), {})
                headers = hit.get('table_headers', metadata.get('table_headers', []))
                address = (column_name(location.get('column_start', 1) + headers.index(label)) + str(mapped['source_row'])
                           if 'worksheet' in location and label in headers and mapped.get('source_row') else None)
                data = hit.get('table_data', metadata.get('table_data', {}))
                source_cell = next((c for c in data.get('cells', []) if c.get('address') == address), None)
                tasks.append({'text': f'请给出表格单元格的具体值（{locator}{suffix}）',
                              'quote_query': f"{row['row_label']} {label}", 'source_ids': [hit['chunk_id']],
                              'expected_value': row['cells'][label],
                              'target_cell': {'row': row['row_label'], 'column': label, 'value': row['cells'][label],
                                              'source_chunk_id': hit['chunk_id'], 'supporting_quote': row['supporting_quote'],
                                              'location': location, 'source_cell': source_cell, 'warnings': row['warnings'],
                                              'kind': 'derived_mapping_not_original_quote'}})
    return tasks


def enumeration_task(question, evidence):
    """Enumerate literal categorical cells, not people or numeric table rows."""
    if not re.search(r'哪些|列出|列举', question):
        return None
    requested = [key for key in ('意愿', '志愿', '类别', '类型', '状态') if key in question]
    if len(requested) != 1:
        return bullet_enumeration(question, evidence)
    fields, items = set(), {}
    for hit in evidence:
        if hit.get('content_type') != 'table':
            continue
        for row in table_view(hit)['rows']:
            for field, value in row['cells'].items():
                if not field.split(' > ')[-1].endswith(requested[0]):
                    continue
                fields.add(field.split(' > ')[-1])
                for part in re.split(r'[、，,;/；\n]+', value):
                    part = part.strip()
                    # A first-person comment is not an additional category.
                    # Keep the original cell unchanged in the citation/trace.
                    if not part or re.match(r'^(?:我|本人)(?:倾|想|希望|愿|更|比较)', part):
                        continue
                    items.setdefault(part, {'chunk_id': hit['chunk_id'], 'quote': row['supporting_quote']})
    if len(fields) != 1 or not items:
        return bullet_enumeration(question, evidence)
    return {'field': next(iter(fields)), 'items': items}


def bullet_enumeration(question, evidence):
    """Reuse literal enumeration for an unambiguous, explicitly named list.

    Read the complete bullet span, including numbered continuation lines.
    This is extraction of source structure, not inferred list membership.
    """
    def topic(value):
        return re.sub(r'[\W_的]', '', value)
    lists = []
    for hit in evidence:
        text = hit['text']
        headings = list(re.finditer(r'(?m)^#{1,6}[ \t]+([^\n]+)', text))
        for i, heading in enumerate(headings):
            label = heading[1].strip()
            name, query = topic(label), topic(question)
            if len(name) < 3 or not any(name[-width:] in query for width in range(3, len(name) + 1)):
                continue
            section = text[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(text)]
            bullets = list(re.finditer(r'(?m)^[ \t]*[-*+][ \t]+', section))
            if len(bullets) < 2:
                continue
            items = {}
            for j, marker in enumerate(bullets):
                end = bullets[j + 1].start() if j + 1 < len(bullets) else len(section)
                quote = section[marker.start():end].strip()
                value = section[marker.end():end].strip()
                if value:
                    items[value] = {'chunk_id': hit['chunk_id'], 'quote': quote}
            lists.append({'field': label, 'items': items, 'literal_list': True})
    return lists[0] if len(lists) == 1 else None


def split_tasks(question):
    tasks = [q.strip() for q in re.split(r'[？?；;]+', question) if q.strip()]
    coordinated = re.fullmatch(r'\s*(?:what|which)\s+are\s+(.+?)\s+and\s+(.+?)(\s+(?:in|for|from)\s+.+)?', question.rstrip('? '), flags=re.I)
    if len(tasks) < 2 and coordinated:
        first, second, scope = coordinated.groups()
        return [f"What is {field}{scope or ''}" for field in (first, second)]
    return tasks


def finite_value(task, evidence, metric=None):
    """Only unambiguous literal metric->numeric relations, not keyword coverage."""
    keys = [metric] if metric else [key for key, aliases in FIELD_ALIASES.items() if any(matches(a, task) for a in aliases)]
    if len(keys) != 1:
        return None
    values = set()
    # An explicit Chinese object must occur in the same clause as its metric.
    obj = re.search(r'(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9 _-]*)的', task)
    for hit in evidence:
        for sentence in re.split(r'[。\n；;]', hit['text']):
            if obj and not matches(obj[1], sentence):
                continue
            aliases = sorted(FIELD_ALIASES[keys[0]], key=len, reverse=True)
            labels = '(?:' + '|'.join((r'(?<![A-Za-z0-9_])' + re.escape(a) + r'(?![A-Za-z0-9_])' if a.isascii() else re.escape(a)) for a in aliases) + ')'
            value_pattern = (r'\d{4}-\d{1,2}-\d{1,2}' if keys[0] == 'date' else
                             r'[A-Z][A-Z0-9]*-[A-Z0-9-]+' if keys[0] == 'identifier' else VALUE)
            # Bilingual repeated labels must refer to this SAME field. Never
            # take an Amount on another line as the value of Fee or a named row.
            pattern = labels + r'\s*(?:' + labels + r')?\s*(?:[:：#]|为|是|is)?\s*(' + value_pattern + ')'
            for m in re.finditer(pattern, sentence, re.I):
                value = m[1].strip()
                if keys[0] == 'duration' and not re.search(r'(?:hours?|小时|minutes?|分钟|seconds?|秒|days?|天)$', value, re.I):
                    continue
                values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def plan_tasks(question, evidence, constraint=None):
    constraint = constraint or {}
    parts = comparison_parts(without_source_names(question, constraint).strip(' ，,：:'))
    aspects = summary_parts(without_source_names(question, constraint))
    factual = question_parts(without_source_names(question, constraint).strip(' ，,：:'))
    if (factual and not any('哪些' in facet for facet in factual)
            and not any(h.get('content_type') == 'table' and h.get('context_coverage_groups') for h in evidence)
            and any(h.get('context_coverage_groups') for h in evidence)):
        specs = []
        for task in split_tasks(question):
            numbers = [i for i, facet in enumerate(factual) if facet in task]
            eligible = [h for h in evidence if any(g.endswith(f':{i}')
                        for i in numbers for g in h.get('context_coverage_groups', []))]
            if not eligible:
                break
            specs.append({'text': task + '。每个所问字段只写一句对应事实，不添加本子问题未要求的字段；保留原文的对象和适用条件。',
                          'quote_query': task, 'source_ids': [h['chunk_id'] for h in eligible],
                          'factual_question': True, 'required_doc_ids': sorted({h['doc_id'] for h in eligible}),
                          'task_id': hashlib.sha256((question + task).encode()).hexdigest()[:16],
                          'original_question': task})
        if len(specs) == len(split_tasks(question)):
            return specs
    if aspects and any(h.get('context_coverage_groups') for h in evidence):
        specs = []
        for number, aspect in enumerate(aspects):
            eligible = [h for h in evidence if any(g.endswith(f':{number}') for g in h.get('context_coverage_groups', []))]
            text = f'请根据当前证据概括{aspect}。只回答这一方面，保留原文的完成状态和适用条件；文档记载挑战及对应措施时，不能擅自认定已解决或仍未解决。'
            specs.append({'text': text, 'quote_query': aspect, 'source_ids': [h['chunk_id'] for h in eligible],
                          'synthesis_aspect': aspect, 'required_doc_ids': sorted({h['doc_id'] for h in eligible}),
                          'task_id': hashlib.sha256((question + aspect).encode()).hexdigest()[:16], 'original_question': question})
        return specs
    if parts and any(h.get('context_coverage_groups') for h in evidence):
        first, second, event = parts
        tail = re.split(r'[，,]\s*(?:并|以及)', without_source_names(question, constraint), maxsplit=1)
        interpretation = tail[1] if len(tail) > 1 else ''
        specs = []
        for number, obj in enumerate((first, second)):
            eligible = [h for h in evidence if any(g.endswith(f':{number}') for g in h.get('context_coverage_groups', []))]
            if not eligible:
                break
            text = (f'请结合当前证据中的各个来源，概括{obj}在{event}中的具体职责。'
                    f'本次只说明{obj}，只写一句fact概括各来源共同支持的职责并引用各来源。保留原文限制与例外。'
                    '事实中的身份称谓、职责动词必须沿用原文，不能更换参与者或省略关键动词；不列人名、日期或流程细节。')
            if number == 1 and interpretation:
                text = (f'先用一句fact完整概括{obj}在{event}中的职责，覆盖各证据所记载的不同主要环节，再单独用一句analysis对照{first}与{second}的职责分工，回答“{interpretation}”。'
                        '只输出两项结论：一句fact和一句analysis，并引用对应原文。事实保留原文身份称谓、职责动词和条件，不列人名、日期或流程细节。')
            specs.append({'text': text, 'quote_query': text, 'source_ids': [h['chunk_id'] for h in eligible],
                          'comparison_object': obj, 'required_doc_ids': sorted({h['doc_id'] for h in eligible}),
                          'require_analysis': bool(number == 1 and interpretation),
                          'task_id': hashlib.sha256((question + obj).encode()).hexdigest()[:16],
                          'original_question': question})
            if number == 1 and interpretation:
                specs[-1]['source_ids'] = [h['chunk_id'] for h in evidence]
        if len(specs) == 2:
            return specs
    specs = []
    units = []
    for original in split_tasks(question):
        ids = task_doc_ids(original, constraint)
        if len(ids) > 1:
            for did in sorted(ids):
                names = [m['name'] for m in constraint.get('matches', []) if did in m['doc_ids']]
                units.append((f"请仅根据{names[0]}回答：{original}", {did}))
        else:
            units.append((original, ids))
    for original, ids in units:
        eligible = [h for h in evidence if not ids or h.get('doc_id', h.get('source', '')) in ids]
        if '原生文本' in original:
            eligible = [h for h in eligible if h.get('origin', h.get('metadata', {}).get('origin')) == 'native_extraction']
        elif '图片' in original:
            eligible = [h for h in eligible if h.get('origin', h.get('metadata', {}).get('origin')) == 'ocr_result']
        semantic = without_source_names(original, constraint)
        enumeration = enumeration_task(semantic, eligible)
        tables = table_tasks(semantic, eligible)
        if enumeration:
            parts = [{'text': original, 'quote_query': original, 'source_ids': [h['chunk_id'] for h in eligible],
                      'enumeration': enumeration}]
        elif tables:
            parts = tables
        else:
            semantic = original
            for match in constraint.get('matches', []):
                semantic = semantic.replace(match['name'], '')
            keys = [key for key, aliases in FIELD_ALIASES.items() if any(matches(a, semantic) for a in aliases)]
            coordinated = len(keys) > 1 and re.search(r'和|与|及|\band\b', original, re.I)
            parts = [{'text': f'请回答{FIELD_ALIASES[key][0]}的具体值' if coordinated else original,
                      'quote_query': FIELD_ALIASES[key][0] if coordinated else original,
                      'source_ids': [h['chunk_id'] for h in eligible] if ids else None,
                      'expected_value': finite_value(semantic, eligible, key),
                      'finite_relation_required': bool(ids and (key or len(keys) == 1))} for key in (keys if coordinated else [None])]
        for spec in parts:
            identity = '|'.join([original, str(sorted(ids)), spec['quote_query'], str(spec.get('source_ids'))])
            spec.update(task_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                        original_question=original, required_doc_ids=sorted(ids))
            specs.append(spec)
    return specs
