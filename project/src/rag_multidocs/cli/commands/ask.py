"""``rag-cli ask`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ...qa.llm import answer_with_evidence, build_chat_model
from ..render import print_hits
from ..runtime import build_runtime


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.provider:
        settings.chat_provider = args.provider
    if args.model:
        settings.chat_model = args.model
    if args.base_url:
        settings.chat_base_url = args.base_url
    top_k = args.top_k or settings.rag_top_k

    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    hits = runtime.search.search(args.query, top_k=top_k, doc_ids=args.doc_id)

    if args.no_generate:
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        else:
            print_hits(hits)
        return 0

    if not hits:
        if args.json:
            print(json.dumps(
                {"question": args.query, "answer": "没有检索到相关依据。",
                 "citations": [], "hits": []},
                ensure_ascii=False, indent=2,
            ))
        else:
            print("没有检索到相关依据。")
        return 0

    if not settings.chat_provider:
        print(
            "[CONFIG_ERROR] 未配置 chat provider。运行 `rag-cli setup` 选择模型，"
            "或用 --provider 指定；API Key 可用 `rag-cli keys set --provider "
            "deepseek` 保存到本项目，也支持 export DEEPSEEK_API_KEY=... 临时注入。",
            file=sys.stderr,
        )
        print_hits(hits)
        return 2

    stream = bool(getattr(args, "stream", False)) and not args.json
    on_token = None
    if stream:
        print("答案：")
        sys.stdout.flush()
        on_token = lambda token: print(token, end="", flush=True)  # noqa: E731

    try:
        chat = build_chat_model(
            settings.chat_provider,
            model=settings.chat_model,
            base_url=settings.chat_base_url,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
        )
        result = answer_with_evidence(args.query, hits, chat, on_token=on_token)
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\n[LLM_ERROR] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            {
                "question": args.query,
                "provider": settings.chat_provider,
                "model": chat.model,
                "index_backend": runtime.backend,
                **result,
                "hits": hits,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if stream:
        print()
    else:
        print("答案：")
        print(result["answer"])
    print("\n来源：")
    for citation in result["citations"]:
        section = f" §{citation['section_path']}" if citation["section_path"] else ""
        print(
            f"[{citation['index']}] {citation['source']}"
            f" p{citation['page']}{section}"
        )
    return 0


__all__ = ["cmd_ask"]
