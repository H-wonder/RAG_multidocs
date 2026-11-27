import json
import re
from ..execution import record

ALIASES = {"SVM": "支持向量机", "RAG": "检索增强生成", "LLM": "大语言模型"}


def comparison_parts(query):
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'^(?:根据|结合|综合|在|从)\s*(?:(?:和|与|及|、)\s*)*[，,：:]\s*', '', query)
    primary = re.split(r"[，,]\s*(?:并|以及)", query, maxsplit=1)[0]
    primary = re.sub(r"^(?:请)?(?:分析|比较|对比)\s*", "", primary)
    separator = '与' if '与' in primary else '和'
    match = re.fullmatch(r"(.{1,40})" + separator + r"(.{1,40}?)在(.{1,80}?)中(?:的)?(?:不同|各自)?(?:作用|功能)[。？?]?", primary)
    return match.groups() if match else None


def summary_parts(query):
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'^(?:根据|结合|综合|在|从)\s*(?:(?:和|与|及|、)\s*)*[，,：:]\s*', '', query)
    match = re.fullmatch(r'(?:请)?(?:概括|总结|归纳)(.{4,160})[。？?]?', query)
    if match:
        parts = tuple(part.strip(' 。？?') for part in re.split(r'以及|[、和及]', match[1]))
        if 2 <= len(parts) <= 4 and all(2 <= len(part) <= 80 for part in parts):
            return parts
    return None


def question_parts(query):
    """Literal, independently answerable questions, including comma joins."""
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'^(?:根据|结合|综合|在|从)\s*(?:(?:和|与|及|、)\s*)*[，,：:]\s*', '', query)
    parts = [part.strip() for part in re.split(r'[，,？?；;]+', query) if part.strip()]
    if 1 < len(parts) <= 3 and all(re.search(r'多少|几(?:年|个|项|天|小时)|哪些|是什么', part) for part in parts):
        return parts
    return []


def constraints(query):
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|％|元|天|小时|个月|年|CNY|USD|ms|GB|MB)?", query))


def query_keywords(query):
    cleaned = re.sub(r"请问|为什么|如何|怎样|是什么|多少|什么|哪个|是否|有什么|吗|呢", " ", query)
    values = re.findall(r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?(?:%|％|元|天|小时|个月|年)?|[一-鿿]{2,}", cleaned)
    return list(dict.fromkeys(values))[:16]


def level1_queries(query, aliases=None):
    aliases = ALIASES if aliases is None else aliases
    expanded = query
    for original, alias in aliases.items():
        expanded = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(original)}(?![A-Za-z0-9_])", alias, expanded)
    keyword = " ".join(query_keywords(query))
    return list(dict.fromkeys(q for q in [expanded, keyword] if q and q != query))[:2]


def validate_rewrites(query, values):
    if not isinstance(values, list) or any(not isinstance(q,str) or not q.strip() or len(q)>1000 for q in values):
        raise ValueError("invalid query rewrite")
    protected = constraints(query)
    for value in values:
        if constraints(value) != protected:
            raise ValueError("rewrite changed numeric constraints")
        # Generated queries may select a subset of objects, but must not invent
        # new capitalized identifiers. Aliases are a transparent core dictionary.
        allowed = {w.lower() for w in re.findall(r"[A-Z][A-Za-z0-9_+.-]*", query)}
        generated = {w.lower() for w in re.findall(r"[A-Z][A-Za-z0-9_+.-]*", value)}
        if generated - allowed - {"what","which","how","compare","the","a","an","and"}:
            raise ValueError("rewrite invented a named identifier")
    return list(dict.fromkeys(values))[:2]


class SemanticRewriter:
    def __init__(self, chat):
        self.chat = chat
        self.model_calls = 0
        self.last_output = None

    def __call__(self, query):
        self.last_output = None
        self.model_calls += 1
        record("rewrite_calls")
        raw = self.chat.complete(
            [
                {
                    "role": "system",
                    "content": "将复杂检索问题改写为最多2个短检索子问题。比较多个对象时，分别查找各对象在具体事件中的行为或作用。优先检索支撑分析的具体事实；文化意义等概括性结论由后续生成完成，不单独改写成抽象概念检索。删除写作指令、重复铺垫及问句套话。不得输出答案、事实、结论或新数字；保留原数字、单位、名称及限定条件。只输出JSON字符串数组。原查询始终由核心保留。",
                },
                {"role": "user", "content": query},
            ]
        )
        self.last_output = raw
        # Accept one complete JSON code fence, as the answer parser does.
        # Never extract an embedded fragment or repair a malformed array.
        fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*)\n```", raw.strip())
        result = json.loads(fenced.group(1) if fenced else raw)
        # A common wire variant wraps each query in a single named field.
        # Unwrap only that exact shape; all content constraints still apply.
        if (isinstance(result, list) and result
                and all(isinstance(item, dict) and set(item) == {"query"}
                        and isinstance(item["query"], str) for item in result)):
            result = [item["query"] for item in result]
        return validate_rewrites(query, result)
