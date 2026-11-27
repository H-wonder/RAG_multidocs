"""OpenAI-compatible chat client and retrieval-augmented answer construction.

API keys are read from the process environment (see ``providers.py`` and
``credentials.py``).  The generated answer is always paired with the retrieved
evidence; callers must not present model output as if it were a retrieved fact.
"""

from __future__ import annotations

import copy
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..providers import api_key, chat_provider
from ..execution import record, remaining

SYSTEM_PROMPT = """你是课程资料检索助手。只能依据用户提供的“证据”回答问题。
要求：
1. 每个结论后用 [编号] 标注证据，例如 [1]、[2]；
2. 证据不足或互相矛盾时，明确说“根据现有资料无法确定”，不要编造；
3. 不要引用证据之外的事实；
4. 只回答所问对象与字段，逐项对应证据，不混入其他对象的信息；用简洁中文回答。"""


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class ChatCompletionError(RuntimeError):
    """Safe, structured failure without provider payloads or reasoning text."""

    def __init__(self, code, message, *, retryable=False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def warning(self):
        return {"code": self.code, "message": str(self), "severity": "warning"}


_DASHSCOPE_V41_FLASH = {"deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"}
_MODEL_LIMITS = {
    ("dashscope", "qwen-plus"): (1_000_000, 32_768),
    ("deepseek", "deepseek-flash"): (1_000_000, 393_216),
    ("deepseek", "deepseek-v4-flash"): (1_000_000, 393_216),
    ("deepseek", "deepseek-v4-pro"): (1_000_000, 393_216),
    ("openai", "gpt-4o-mini"): (128_000, 16_384),
    ("openai", "gpt-4o-mini-2024-07-18"): (128_000, 16_384),
}


@dataclass(frozen=True)
class ChatBudget:
    context_window: int
    initial_tokens: int
    retry_tokens: int
    reasoning_effort: str = ""

    def as_trace(self):
        return {
            "context_window": self.context_window,
            "initial_completion_limit": self.initial_tokens,
            "reserved_completion_limit": self.retry_tokens,
            "reasoning_effort": self.reasoning_effort or None,
        }


def _ollama_context(base_url, model):
    # Probe the allocated window of the local runner, not the model's
    # theoretical maximum. Older/cold runners can omit it.
    url = urlsplit(base_url)
    if (url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username or url.path.rstrip("/") not in {"", "/v1", "/v1/chat/completions"}):
        return None
    tag = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
    try:
        with urllib.request.urlopen(f"{url.scheme}://{url.netloc}/api/ps", timeout=0.5) as response:
            models = json.load(response)["models"]
        for entry in models:
            if entry.get("model", entry.get("name")) in {tag, model}:
                length = entry.get("context_length")
                if type(length) is int and length > 0:
                    return length
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def resolve_chat_budget(provider, model="", max_tokens=0, context_tokens=0, *, base_url=""):
    if max_tokens < 0 or context_tokens < 0:
        raise ValueError("Token budgets must be zero (auto) or positive")
    preset = chat_provider(provider)
    model = model or preset.default_model
    flash = provider == "dashscope" and model in _DASHSCOPE_V41_FLASH
    specs = (1_000_000, 393_216) if flash else _MODEL_LIMITS.get((provider, model))
    allocated = None
    if provider == "ollama" and not context_tokens:
        allocated = _ollama_context(base_url or preset.base_url, model)
    window = context_tokens or allocated or (specs[0] if specs else 4096 if provider == "ollama" else 8192)
    if specs and window > specs[0]:
        raise ValueError("Configured context exceeds the model window")
    initial = max_tokens or (8192 if flash else 1024)
    if specs and initial > specs[1]:
        raise ValueError("Configured completion limit exceeds the model output limit")
    # Shared DeepSeek limit covers reasoning + visible completion. Leave the
    # existing RAG evidence allowance available even when an explicit small
    # context override can accommodate only the first attempt.
    retry = min(12_288, initial + 4096, window - 4600) if flash else initial
    return ChatBudget(window, initial, max(initial, retry), "low" if flash else "")


def reasoning_output_budget(provider, model, max_tokens):
    return resolve_chat_budget(provider, model, max_tokens).retry_tokens


def _usage(data):
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return False
    details = usage.get("completion_tokens_details") or {}
    counts = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}
    counts["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
    for key, value in counts.items():
        if type(value) is int and value >= 0:
            record("reported_" + key, value)
    return type(counts["reasoning_tokens"]) is int and counts["reasoning_tokens"] > 0


def _answer(content, reason, has_reasoning):
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ChatCompletionError("LLM_PROTOCOL_ERROR", "模型回答字段格式无效。")
    answer = content.strip()
    if reason == "length":
        if not answer and has_reasoning:
            raise ChatCompletionError("LLM_REASONING_TRUNCATED", "生成达到长度上限，仅返回推理内容。")
        raise ChatCompletionError("LLM_OUTPUT_TRUNCATED", "生成达到长度上限，未返回完整答案。")
    if reason == "content_filter":
        raise ChatCompletionError("LLM_CONTENT_FILTERED", "生成被服务侧内容过滤拦截。")
    if reason not in (None, "stop"):
        raise ChatCompletionError("LLM_UNSUPPORTED_FINISH", "服务未返回普通文本答案。")
    if not answer:
        if has_reasoning:
            raise ChatCompletionError("LLM_REASONING_ONLY", "服务仅返回推理内容，缺少最终答案。")
        raise ChatCompletionError("LLM_EMPTY_CONTENT", "服务返回空的最终答案。")
    return answer


class OpenAICompatibleChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_value: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
        retries: int = 2,
        response_format: dict | None = None,
        reasoning_retry_tokens: int | None = None,
        budget: ChatBudget | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.url = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self.model = model
        self.api_key = api_key_value
        self.temperature = temperature
        if max_tokens <= 0 or retries < 1:
            raise ValueError("max_tokens and retries must be positive")
        self.budget = budget
        self.context_window = budget.context_window if budget else None
        self.max_tokens = budget.initial_tokens if budget else max_tokens
        self.output_budget = budget.retry_tokens if budget else max(max_tokens, reasoning_retry_tokens or max_tokens)
        self.reasoning_retry_tokens = self.output_budget if self.output_budget > self.max_tokens else 0
        self.timeout = timeout
        self.retries = retries
        self.response_format = response_format

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, stream, response_format_override=None, *, max_tokens=None,
                 answer_only=False) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if self.budget and self.budget.reasoning_effort:
            if answer_only:
                body["enable_thinking"] = False
            else:
                body["reasoning_effort"] = self.budget.reasoning_effort
        chosen_format = response_format_override if response_format_override is not None else self.response_format
        if chosen_format is not None:
            body["response_format"] = chosen_format
        if stream:
            body["stream"] = True
        return json.dumps(body).encode("utf-8")

    @staticmethod
    def _read_response(response):
        data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ChatCompletionError("LLM_PROTOCOL_ERROR", "服务返回了无效的响应格式。")
        if data.get("error"):
            raise ChatCompletionError("LLM_API_ERROR", "服务返回 API 错误。")
        has_reasoning = _usage(data)
        choice = data["choices"][0]
        message = choice["message"]
        return _answer(message.get("content"), choice.get("finish_reason"),
                       has_reasoning or bool(message.get("reasoning_content")))

    @staticmethod
    def _read_stream(response, on_token):
        pieces, reason, has_reasoning, done = [], None, False, False
        last_usage = {}
        for raw in response:
            remaining(float("inf"))
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ChatCompletionError("LLM_API_ERROR", "服务在流式响应中返回 API 错误。")
            if event.get("usage"):
                last_usage = event
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            reason = choice.get("finish_reason") or reason
            delta = choice.get("delta") or {}
            has_reasoning |= bool(delta.get("reasoning_content"))
            piece = delta.get("content")
            if piece is not None and not isinstance(piece, str):
                raise ChatCompletionError("LLM_PROTOCOL_ERROR", "流式回答字段格式无效。")
            if piece:
                pieces.append(piece)
                on_token(piece)
        if not done and reason is None:
            raise ChatCompletionError("LLM_STREAM_INCOMPLETE", "流式响应提前结束。")
        has_reasoning |= _usage(last_usage)
        return _answer("".join(pieces), reason, has_reasoning)

    def _request(self, messages, response_format_override=None, on_token=None):
        emitted = False
        max_tokens = self.max_tokens
        retried_length = False
        answer_only = False

        def emit(piece):
            nonlocal emitted
            emitted = True
            on_token(piece)

        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url, headers=self._headers(), method="POST",
                data=self._payload(messages, on_token is not None, response_format_override,
                                   max_tokens=max_tokens, answer_only=answer_only),
            )
            try:
                timeout = remaining(self.timeout)
                record("transport_attempts")
                record("requested_completion_tokens", max_tokens)
                if attempt:
                    record("transport_retries")
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if on_token is not None:
                        return self._read_stream(response, emit)
                    return self._read_response(response)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                error = ChatCompletionError("LLM_HTTP_ERROR", f"服务返回 HTTP {exc.code}。",
                                            retryable=retryable)
                exc.close()
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                error = ChatCompletionError("LLM_TRANSPORT_ERROR", "推理服务连接失败或超时。",
                                            retryable=True)
            except ChatCompletionError as exc:
                error = exc
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                error = ChatCompletionError("LLM_PROTOCOL_ERROR", "服务返回了无效的响应格式。")
            if not emitted and attempt + 1 < self.retries:
                reasoning_truncated = error.code == "LLM_REASONING_TRUNCATED"
                final_only_supported = bool(self.budget and self.budget.reasoning_effort)
                if (not retried_length and (reasoning_truncated or
                        (final_only_supported and error.code == "LLM_OUTPUT_TRUNCATED"))
                        and (self.output_budget > max_tokens or (reasoning_truncated and final_only_supported))):
                    retried_length = True
                    max_tokens = self.output_budget
                    answer_only = reasoning_truncated and final_only_supported
                    record("reasoning_budget_retries" if reasoning_truncated else "completion_budget_retries")
                    if answer_only:
                        record("answer_only_retries")
                    continue
                if error.retryable:
                    try:
                        time.sleep(min(1 + attempt, remaining(self.timeout)))
                    except TimeoutError:
                        raise ChatCompletionError("LLM_TRANSPORT_ERROR", "推理请求已超时。") from None
                    continue
            raise error from None

    def _post(self, messages, response_format_override=None) -> str:
        return self._request(messages, response_format_override)

    def complete_grounded(self, messages, quote_choices):
        """Optional per-call exact-quote grammar for an explicitly opted-in schema.

        No response repair or shared mutable schema change. Providers without
        schema support retain complete(); L2 still validates the returned quote
        against the particular cited evidence, not merely enum membership.
        """
        if not self.response_format or not quote_choices:
            return self.complete(messages)
        schema = copy.deepcopy(self.response_format)
        try:
            items = schema["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"]["supporting_quotes"]["items"]
            if items.get("type") != "string":
                return self.complete(messages)
            items["enum"] = list(quote_choices)
            schema["json_schema"]["schema"]["properties"]["claims"]["maxItems"] = 1
        except (KeyError, TypeError):
            return self.complete(messages)
        return self._post(messages, response_format_override=schema)

    def complete_synthesis(self, messages, line_ids, claim_limit=8):
        """Use the existing L1 wire contract when the local provider supports it."""
        if not getattr(self, "supports_json_schema", False):
            return self.complete(messages)
        properties = {"text": {"type": "string"},
                      "kind": {"type": "string", "enum": ["fact", "analysis"]},
                      "refs": {"type": "array", "minItems": 1,
                               "items": {"type": "string", "enum": list(line_ids)}}}
        schema = {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["answered", "partial", "insufficient_evidence"]},
            "claims": {"type": "array", "maxItems": claim_limit, "items": {
                "type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}}}, "required": ["status", "claims"],
            "additionalProperties": False}
        return self._post(messages, response_format_override={"type": "json_schema",
            "json_schema": {"name": "grounded_lines", "strict": True, "schema": schema}})

    def complete_fields(self, messages, task_ids, line_ids):
        """Return field values and line IDs, without asking the model to copy quotes."""
        if not self.response_format:
            return self.complete(messages)
        fields = {
            "task_id": {"type": "string", "enum": list(task_ids)},
            "line_id": {"type": "string", "enum": list(line_ids)},
            "value": {"type": "string"},
        }
        schema = {
            "type": "object",
            "properties": {"answers": {"type": "array", "maxItems": len(task_ids), "items": {
                "type": "object", "properties": fields, "required": list(fields),
                "additionalProperties": False,
            }}},
            "required": ["answers"],
            "additionalProperties": False,
        }
        return self._post(messages, response_format_override={
            "type": "json_schema",
            "json_schema": {"name": "labeled_field_answers", "strict": True, "schema": schema},
        })

    def stream(self, messages: list[dict[str, str]], on_token) -> str:
        """Legacy transport callback. Consumers must validate the complete result."""
        return self._request(messages, on_token=on_token)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._post(messages)


def build_chat_model(
    provider: str,
    model: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_tokens: int = 0,
    *,
    context_tokens: int = 0,
    budget: ChatBudget | None = None,
) -> OpenAICompatibleChat:
    preset = chat_provider(provider)
    model = model or preset.default_model
    budget = budget or resolve_chat_budget(provider, model, max_tokens, context_tokens, base_url=base_url)
    chat = OpenAICompatibleChat(
        base_url=base_url or preset.base_url,
        model=model,
        api_key_value=api_key(preset),
        temperature=temperature,
        budget=budget,
        timeout=110 if budget.reasoning_effort else 60,
    )
    chat.supports_json_schema = provider == "ollama"
    return chat


def build_evidence(hits: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Render retrieved chunks as numbered evidence with source metadata."""
    parts: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        section = f" §{hit['section_path']}" if hit.get("section_path") else ""
        header = f"[{index}] {hit.get('source', '')} p{hit.get('page', 0)}{section}"
        text = str(hit.get("text", ""))
        block = f"{header}\n{text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence = build_evidence(hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\n证据：\n{evidence}\n\n请依据证据回答，并用 [编号] 标注引用。"
            ),
        },
    ]


def answer_with_evidence(
    question: str,
    hits: list[dict[str, Any]],
    chat: ChatModel,
    max_evidence_chars: int = 8000,
    on_token=None,
    *,
    style="professional",
    context_tokens=None,
    output_tokens=1024,
    allow_calculation=False,
    check_numbers=True,
    structured_table_answers=False,
    source_constraints=None,
) -> dict[str, Any]:
    """Retrieve-backed answer + citations; never call the model without evidence.

    When ``on_token`` is given and the chat model supports streaming, tokens are
    forwarded as they arrive (used by the interactive CLI).
    """
    from ..ingestion.chunking import estimate_tokens
    from .grounding import citations
    from .calculations import calculate, calculation_requested
    from .labeled_fields import FIELD_SEPARATOR
    from .table_evidence import label_in
    from .tasks import plan_tasks, synthesis_question, split_tasks

    styles = {"professional": "", "teaching": "简要解释术语与依据。", "concise": "直接简短回答。"}
    if style not in styles:
        raise ValueError("unknown response style")
    result = {"answer": "没有检索到相关依据。", "status": "insufficient_evidence",
              "claims": [], "citations": [], "warnings": [], "used_llm": False,
              "model_calls": 0, "attempts": [], "generation_evidence": [],
              "generation_context_tokens_estimate": 0, "fact_verified": False,
              "answer_mode": "natural_language"}
    if not hits:
        return result
    from .grounding import GroundedQA
    from ..retrieval.sources import resolve_sources, without_source_names
    scope = source_constraints or (resolve_sources(question, hits) if structured_table_answers else {'matches': []})
    semantic_question = without_source_names(question, scope)
    specs, extracted = [], []
    if structured_table_answers and not synthesis_question(semantic_question):
        specs = plan_tasks(question, hits, scope)
        parts = [without_source_names(part, scope) for part in split_tasks(question)]
        if scope.get('matches'):
            parts = [re.sub(r'^\s*(?:请)?(?:根据|依据|结合|参照|参考|仅用)\s*[，,:：]?\s*', '', part) for part in parts]
        fields = [part.partition('的')[2] if '的' in part else part for part in parts]
        requested_fields = [field for part in fields for field in FIELD_SEPARATOR.split(part)]
        complete_fields = all(any(label_in(spec.get('target_cell', {}).get('column', ''), field, field_query=True)
                                  for spec in specs) for field in requested_fields)
        if specs and complete_fields and all(spec.get('target_cell') for spec in specs):
            reader = GroundedQA(chat, level3=check_numbers)
            extracted = [reader._extract_table(spec, hits) for spec in specs]
            if not all(extracted):
                extracted = []
    computation = calculate(semantic_question, hits, [spec['target_cell'] for spec in specs] if extracted else ()) if allow_calculation else None
    if computation and len(split_tasks(question)) == 1 and not synthesis_question(question):
        claims = [{'text': operand['quote'], 'citation_ids': [operand['citation_id']],
                   'supporting_quotes': [operand['quote']]} for operand in computation['operands']]
        result.update(answer='计算结果（非原文摘录）：' + computation['expression'] + ' ' +
                      ''.join(f'[{i}]' for i in sorted({o['citation_id'] for o in computation['operands']})),
                      status='answered', claims=claims, citations=citations(claims, hits),
                      calculations=[computation], calculation_citations=citations(claims, hits),
                      answer_mode='structured_calculation', answer_evidence=hits,
                      extraction_evidence=[hits[i-1] for i in sorted({o['citation_id'] for o in computation['operands']})])
        if not computation.get('direct_answer'):
            result['answer'] = '\n'.join(o['object'] + '：' + o['value'] + o['unit'] for o in computation['operands']) + '\n' + result['answer']
        return result
    if extracted and not computation:
        claims = [claim for row in extracted for claim in row['claims']]
        arithmetic_pending = calculation_requested(semantic_question)
        result.update(answer='\n\n'.join(row['answer'] for row in extracted),
                      status='partial' if arithmetic_pending else 'answered', claims=claims, citations=citations(claims, hits),
                      answer_mode='structured_extraction', task_requirements=specs,
                      extraction_evidence=list({h['chunk_id']:h for row in extracted for h in row['extraction_evidence']}.values()))
        if arithmetic_pending:
            result['answer'] += '\n未能依据这些单元格可靠完成所请求的计算。'
            result['warnings'].append({'code': 'CALCULATION_NOT_PERFORMED', 'message': '仅提取了操作数，未完成计算'})
        return result
    selected = []
    for hit in hits:
        trial = selected + [hit]
        # Keep whole chunks, including table headers and the complete target row.
        if len(build_evidence(trial, max_chars=10**9)) > max_evidence_chars:
            continue
        selected = trial
    def messages_for(rows):
        messages = build_messages(question, rows)
        # Upstream's short evidence prompt is the default. Extra output
        # protocols and per-task instructions caused regressions in comparison.
        if styles[style]:
            messages[0]['content'] += '\n' + styles[style]
        if computation:
            messages[1]['content'] += '\n程序根据证据核算的派生结果（不是原文）：' + computation['expression']
        return messages
    messages = messages_for(selected)
    def prompt_cost():
        return estimate_tokens(json.dumps(messages, ensure_ascii=False))
    while selected and context_tokens and prompt_cost() + output_tokens > context_tokens:
        selected.pop()
        messages = messages_for(selected)
    if len(selected) < len(hits):
        result['warnings'].append({'code': 'CONTEXT_TRIMMED', 'message': 'Whole evidence blocks omitted to respect the existing prompt budget'})
    if not selected:
        result['answer'] = '上下文预算不足以提供有效证据。'
        return result
    result.update(generation_evidence=selected,
                  generation_context_tokens_estimate=estimate_tokens(build_evidence(selected)),
                  prompt_tokens_estimate=prompt_cost(), used_llm=True, model_calls=1)
    started = time.perf_counter()
    try:
        record("generation_calls")
        if on_token is not None and hasattr(chat, "stream"):
            raw = chat.stream(messages, on_token)
        else:
            raw = chat.complete(messages)
            if on_token is not None:
                on_token(raw)
    except Exception as exc:
        warning = exc.warning() if isinstance(exc, ChatCompletionError) else {'code': 'LLM_GENERATION_ERROR', 'message': '生成服务不可用，检索证据已保留。'}
        result.update(status='generation_failed', answer='生成服务不可用，检索证据已保留。')
        result['warnings'].append(warning)
        return result
    answer = raw.strip()
    result['attempts'] = [{'attempt': 1, 'messages': messages, 'raw_response': raw,
                           'validation_errors': [], 'prompt_tokens_estimate': prompt_cost(),
                           'seconds': time.perf_counter() - started}]
    if not answer:
        result.update(status='generation_failed', answer='生成服务返回空内容。')
        return result
    # References are selected by the model; excerpts and source metadata come
    # from the actual prompt, never model-generated verbatim quote strings.
    refs = {int(value) for value in re.findall(r'\[(\d+)\]', answer)}
    invalid = refs - set(range(1, len(selected) + 1))
    valid = sorted(refs - invalid)
    # Natural language cannot be fact-checked by token-set containment.
    # Preserve the raw answer and its actual references; deterministic table
    # extraction and arithmetic retain their own structural validation.
    refusal = bool(re.match(r'^(?:根据(?:现有|当前|提供的|所提供的)?资料[，,：:\s]*)?(?:无法确定|无法回答|不能确定|资料不足|没有检索到)', answer))
    incomplete = bool(re.search(r'无法确定|资料不足|无法判断', answer))
    status = 'insufficient_evidence' if refusal else 'partial' if incomplete or invalid or not valid else 'answered'
    if invalid and not refusal:
        answer = re.sub(r'\[(\d+)\]', lambda m: m[0] if int(m[1]) not in invalid else '【引用无效】', answer)
        result['warnings'].append({'code': 'INVALID_SOURCE_REFERENCE', 'message': '模型引用了本次证据中不存在的编号'})
    elif not valid and not refusal:
        result['warnings'].append({'code': 'MISSING_SOURCE_REFERENCE', 'message': '回答未标明本次证据编号，不能视为完整溯源'})
    claims = [{'text': answer, 'citation_ids': valid,
               'supporting_quotes': [selected[i-1]['text'] for i in valid]}] if valid else []
    if refusal:
        claims = []
        answer = re.sub(r'\[\d+\]', '', answer).strip()
    result.update(answer=answer, status=status, claims=claims, citations=citations(claims, selected), answer_evidence=selected)
    if computation:
        result['calculations'] = [computation]
    return result


__all__ = [
    "ChatCompletionError",
    "ChatModel",
    "OpenAICompatibleChat",
    "SYSTEM_PROMPT",
    "answer_with_evidence",
    "build_chat_model",
    "build_evidence",
    "build_messages",
]
