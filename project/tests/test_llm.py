from rag_multidocs.llm import answer_with_evidence, build_evidence, build_messages


class FakeChat:
    def __init__(self):
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "检索器负责取回段落 [1]。"


def _hits():
    return [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source": "notes.txt",
            "page": 2,
            "section_path": "检索 > 训练",
            "text": "检索器负责从外部知识库取回相关段落。",
            "score": 0.9,
        }
    ]


def test_evidence_contains_source_and_section():
    evidence = build_evidence(_hits())
    assert "[1] notes.txt p2 §检索 > 训练" in evidence
    assert "检索器" in evidence


def test_answer_is_grounded_and_has_citations():
    chat = FakeChat()
    result = answer_with_evidence("检索器做什么", _hits(), chat)

    assert result["used_llm"] is True
    assert "[1]" in result["answer"]
    assert result["citations"][0]["source"] == "notes.txt"
    assert result["citations"][0]["page"] == 2
    assert chat.messages is not None
    assert "证据" in chat.messages[1]["content"]
    assert "notes.txt" in chat.messages[1]["content"]


def test_no_hits_does_not_call_model():
    class ExplodingChat:
        def complete(self, messages):
            raise AssertionError("没有证据时不应调用模型")

    result = answer_with_evidence("不存在的问题", [], ExplodingChat())
    assert result["used_llm"] is False
    assert result["citations"] == []
    assert "没有检索到" in result["answer"]


def test_messages_include_numbered_evidence():
    messages = build_messages("问题", _hits())
    assert messages[0]["role"] == "system"
    assert "[1]" in messages[1]["content"]
