"""Finite same-unit arithmetic over explicit table operands; never eval()."""
from decimal import Decimal
import re

from ..retrieval.needs import build_needs, support
from .table_evidence import table_view, cell_with_unit


def calculation_requested(question):
    """Recognize the existing finite arithmetic operations, not a task router."""
    return bool(re.search(r'计算|合计|总计|总额|总和|之和|相加|相差|差值|之差|乘以|除以|减去|加上|\bsum\b|\bdifference\b', question, re.I))


def calculate(question, evidence, mapped_cells=()):
    product = conditional_total(question, evidence)
    if product:
        return product
    expression = binary_expression(question, evidence)
    if expression:
        return expression
    if not re.search(r"相差|差值|difference|合计|总计|总额|总和|之和|相加|\bsum\b", question, re.I):
        return None
    mapped = mapped_table_total(question, evidence, mapped_cells)
    if mapped:
        return mapped
    needs = build_needs(question)
    if len(needs) != 2 or any(n.kind != "relation" for n in needs) or needs[0].metric != needs[1].metric:
        return None
    operands = []
    for need in needs:
        candidates = []
        for i, hit in enumerate(evidence, 1):
            mapped = []
            if hit.get('content_type') == 'table':
                for row in table_view(hit, need.object)['rows']:
                    mapped.extend({'row': row['row_label'], 'column': label,
                                   'value': cell_with_unit(label, value), 'quote': row['supporting_quote']}
                                  for label, value in row['cells'].items())
            relation = support({**hit, 'semantic_cells': mapped} if mapped else hit, need)
            if mapped:
                cells = hit.get('semantic_cells', hit.get('metadata', {}).get('semantic_cells', []))
                original = support({**hit, 'semantic_cells': [
                    {**cell, 'value': cell_with_unit(cell.get('column', ''), cell['value'])} for cell in cells]}, need)
                if (original.get('kind') == relation.get('kind') == 'mapped_cell_relation'
                        and original['value'] != relation['value']):
                    return None  # inconsistent source mappings are not arithmetic operands
            if not relation["supported"] or relation["kind"] != "mapped_cell_relation":
                continue
            match = re.fullmatch(r"\s*([¥￥$]?)\s*(-?\d+(?:\.\d+)?)\s*(%|％|元|万元|CNY|USD|ms|秒|天)?\s*", relation["value"])
            if not match:
                continue
            prefix, number, suffix = match.groups()
            unit = prefix or suffix or "scalar"
            if unit == "％":
                unit = "%"
            candidates.append({"value": number, "unit": unit, "citation_id": i, "quote": relation["quote"], "object": need.object, "metric": need.metric, "location": hit.get("location", {})})
        if not candidates or len({(c["value"],c["unit"]) for c in candidates}) != 1:
            return None
        operands.append(candidates[0])
    if operands[0]["unit"] != operands[1]["unit"]:
        return None
    values = [Decimal(c["value"]) for c in operands]
    difference = bool(re.search(r"相差|差值|difference", question, re.I))
    result = abs(values[0]-values[1]) if difference else sum(values)
    unit = "percentage_points" if operands[0]["unit"] == "%" and difference else operands[0]["unit"]
    if unit == "%" and not difference:
        return None  # adding percentages needs an explicit shared denominator
    return {"operation": "absolute_difference" if difference else "sum", "operands": operands, "result": str(result), "unit": unit, "expression": ("abs("+str(values[0])+" - "+str(values[1])+")" if difference else str(values[0])+" + "+str(values[1]))+" = "+str(result)+" "+unit, "origin": "deterministic_calculation_not_source_quote", "verification_scope": "explicit operand mapping and Decimal arithmetic only"}


def mapped_table_total(question, evidence, targets):
    """Connect two already validated cells to the existing same-unit Decimal total."""
    if len(targets) != 2:
        return None
    operands = []
    for target in targets:
        if target.get('warnings'):
            return None
        ids = [i for i, hit in enumerate(evidence, 1) if hit['chunk_id'] == target['source_chunk_id']]
        value = cell_with_unit(target['column'], target['value'])
        match = re.fullmatch(r'\s*([¥￥$]?)\s*(-?\d+(?:\.\d+)?)\s*(%|％|元|万元|CNY|USD|ms|秒|天)?\s*', value)
        if len(ids) != 1 or not match:
            return None
        prefix, number, suffix = match.groups()
        operands.append({'value': number, 'unit': (prefix or suffix or 'scalar').replace('％', '%'),
                         'citation_id': ids[0], 'quote': target['supporting_quote'],
                         'object': target['row'], 'metric': target['column'], 'location': target['location']})
    if operands[0]['unit'] != operands[1]['unit']:
        return None
    difference = bool(re.search(r'相差|差值|difference', question, re.I))
    if operands[0]['unit'] == '%' and not difference:
        return None
    left, right = (Decimal(row['value']) for row in operands)
    value = abs(left - right) if difference else left + right
    unit = 'percentage_points' if difference and operands[0]['unit'] == '%' else operands[0]['unit']
    expression = f'abs({left} - {right})' if difference else f'{left} + {right}'
    return {'operation': 'absolute_difference' if difference else 'sum', 'operands': operands,
            'result': str(value), 'unit': unit, 'expression': expression + f' = {value} {unit}',
            'origin': 'deterministic_calculation_not_source_quote',
            'verification_scope': 'validated table cells, same units and Decimal arithmetic'}


def conditional_total(question, evidence):
    """One explicit per-group quantity times an unambiguous source group count."""
    if not re.search(r'一共|总共|总计|总数|总量|所有.+多少', question):
        return None
    rates = list(re.finditer(
        r'每(?:个|间|所|台|家|份|辆|组|套|批)([一-鿿]{1,16}?)(?:都)?(?:可以|能够|能|可)?'
        r'(?:放|容纳|装|配备|需要|含有|包含|有)\s*(\d+(?:\.\d+)?)\s*([一-鿿]{1,12})(?=[，,。；;]|$)', question))
    if len(rates) != 1:
        return None
    group, rate, unit = rates[0].groups()
    counts = []
    for i, hit in enumerate(evidence, 1):
        for line in hit['text'].splitlines():
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:个|间|所|台|家|份|辆|组|套|批)\s*' + re.escape(group), line)
            if match:
                counts.append({'value': match[1], 'unit': group, 'citation_id': i,
                               'quote': line, 'object': group, 'metric': 'count', 'origin': 'source'})
    if not counts or len({c['value'] for c in counts}) != 1:
        return None
    count = counts[0]
    result = Decimal(count['value']) * Decimal(rate)
    assumed = {**count, 'value': rate, 'unit': unit + '/' + group,
               'origin': 'question_assumption', 'question_excerpt': rates[0][0]}
    return {'direct_answer': True, 'operation': 'product', 'operands': [count, assumed], 'result': str(result), 'unit': unit,
            'expression': count['value'] + ' × ' + rate + ' = ' + str(result) + ' ' + unit + '（每' + group + '的数量按题设）',
            'origin': 'deterministic_calculation_not_source_quote',
            'verification_scope': 'literal source group count times explicit question assumption with Decimal arithmetic'}



def binary_expression(question, evidence):
    """A bounded, explicit binary operation over actual cited source values."""
    if not re.search(r'计算|求.+(?:结果|和|差|积|商)', question):
        return None
    units = r'(?:万元|元|kg|千克|米|箱|件|台|张|个|天|小时|%|％)'
    pattern = r'(-?\d+(?:\.\d+)?)\s*(' + units + r')?\s*(加上|加|\+|减去|减|-|乘以|乘|×|\*|除以|÷|/)\s*(-?\d+(?:\.\d+)?)\s*(' + units + ')?'
    matches = list(re.finditer(pattern, question))
    if len(matches) != 1:
        return None
    a, au, op, b, bu = matches[0].groups()
    operands = []
    for number, specified in ((a, au), (b, bu)):
        found = []
        for i, hit in enumerate(evidence, 1):
            for line in hit['text'].splitlines():
                for match in re.finditer(r'(?<![\d.])(-?\d+(?:\.\d+)?)\s*(' + units + ')?(?![\d.])', line):
                    value, unit = match.groups()
                    if Decimal(value) == Decimal(number) and (not specified or specified == unit):
                        found.append({'value': value, 'unit': unit or 'scalar', 'citation_id': i,
                                      'quote': line, 'origin': 'source'})
        if not found or len({row['unit'] for row in found}) != 1:
            return None
        operands.append(found[0])
    left, right = (Decimal(row['value']) for row in operands)
    u, v = (row['unit'] for row in operands)
    if op in ('加上', '加', '+', '减去', '减', '-'):
        if u != v:
            return None
        subtract = op in ('减去', '减', '-')
        result, unit, operation, sign = (left - right if subtract else left + right), u, ('difference' if subtract else 'sum'), ('-' if subtract else '+')
    elif op in ('乘以', '乘', '×', '*'):
        result, operation, sign = left * right, 'product', '×'
        unit = v if u == 'scalar' else u if v == 'scalar' else u + '×' + v
    else:
        if right == 0:
            return None
        result, operation, sign = left / right, 'quotient', '÷'
        unit = 'scalar' if u == v else u if v == 'scalar' else u + '/' + v
    return {'direct_answer': True, 'operation': operation, 'operands': operands, 'result': str(result), 'unit': unit,
            'expression': str(left) + ' ' + sign + ' ' + str(right) + ' = ' + str(result) + ('' if unit == 'scalar' else ' ' + unit),
            'origin': 'deterministic_calculation_not_source_quote',
            'verification_scope': 'explicit source operands, compatible units and Decimal arithmetic'}
