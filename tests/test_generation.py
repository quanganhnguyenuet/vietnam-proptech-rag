import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.generator import AnswerGenerator, AnswerClaim, GroundedAnswer, GenerationError, LLMConfig, RetrievalQuery
from src.rag.service import LegalChatService


def evidence():
    return {"agent_context": {"text": "[S1] Điều 1. Quy định\nNội dung luật.", "sources": [
        {"label": "S1", "chunk_id": "doc_1", "citation": "Luật Demo, Điều 1", "source_url": "https://vbpl.vn/demo"}
    ]}}


def generator(answer=None, **response_overrides):
    client = MagicMock()
    parsed = answer or GroundedAnswer(status="answered", claims=[AnswerClaim(text="Nội dung trả lời.", source_labels=["S1"])], missing_information=[])
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed,
                                                        model="test-model", usage=None, **response_overrides)
    return AnswerGenerator(LLMConfig(model="test-model"), client)


def test_generator_uses_structured_output_and_maps_trusted_citations():
    service = generator()
    result = service.generate("Câu hỏi", evidence())
    assert result["answer"] == "Nội dung trả lời. [S1]"
    assert result["citations"][0]["source_url"] == "https://vbpl.vn/demo"
    kwargs = service.client.responses.parse.call_args.kwargs
    assert kwargs["store"] is False
    assert kwargs["text_format"] is GroundedAnswer
    assert kwargs["max_output_tokens"] == 1600


def test_no_evidence_never_calls_paid_api():
    service = generator()
    assert service.generate("q", {})["status"] == "insufficient_evidence"
    service.client.responses.parse.assert_not_called()


def test_evidence_labels_must_exist_in_the_text_sent_to_the_model():
    context = evidence()
    context["agent_context"]["text"] = "Nội dung không có nhãn nguồn."
    service = generator()
    with pytest.raises(GenerationError, match="do not match"):
        service.generate("q", context)
    service.client.responses.parse.assert_not_called()


def test_request_can_explicitly_disable_reranker_for_a_live_api_test():
    retriever, model = MagicMock(), MagicMock()
    retriever.retrieve.return_value = evidence()
    model.generate.return_value = {"answer": "text"}
    LegalChatService(retriever, model).answer("Câu hỏi", use_reranker=False)
    retriever.retrieve.assert_called_once_with("Câu hỏi", 5, use_reranker=False)


@pytest.mark.parametrize("claim", [AnswerClaim(text="Sai nguồn", source_labels=["G99"]),
                                  AnswerClaim(text="Không nguồn", source_labels=[]),
                                  AnswerClaim(text="Tự gắn [S99]", source_labels=["S1"])])
def test_generator_rejects_ungrounded_citation_labels(claim):
    parsed = GroundedAnswer(status="answered", claims=[claim], missing_information=[])
    with pytest.raises(GenerationError):
        generator(parsed).generate("q", evidence())


def test_generator_does_not_return_truncated_or_refused_output():
    service = generator()
    service.client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    with pytest.raises(GenerationError, match="completed"):
        service.generate("q", evidence())


def test_insufficient_evidence_can_request_missing_information():
    parsed = GroundedAnswer(status="insufficient_evidence", claims=[], missing_information=["Cần văn bản sửa đổi."])
    result = generator(parsed).generate("q", evidence())
    assert result["status"] == "insufficient_evidence"
    assert not result["citations"]
    assert "Cần văn bản sửa đổi" in result["answer"]


@pytest.mark.parametrize("missing", [[], ["Thông tin về mục đích cần chứng minh thuộc hộ khẩu."]])
def test_ambiguous_question_asks_one_clarification_without_generic_refusal(missing):
    question = "Bạn muốn xác nhận thông tin cư trú hay quyền sử dụng đất của hộ gia đình?"
    parsed = GroundedAnswer(status="insufficient_evidence", claims=[],
                            missing_information=missing, clarification_question=question)
    result = generator(parsed).generate("Không có tên trong sổ đỏ, chứng minh thuộc hộ khẩu?", evidence())
    assert result["answer"] == question
    assert result["clarification_question"] == question
    assert result["missing_information"] == []
    assert result["citations"] == []


@pytest.mark.parametrize("fields", [
    {"clarification_question": "Bạn làm thủ tục gì? Bạn có giấy tờ nào?"},
    {"missing_information": ["Bạn có giấy tờ cư trú nào?"]},
    {"missing_information": ["Thiếu thời điểm sử dụng đất.", "Thiếu văn bản liên quan."]},
    {"claims": [{"text": "Bạn cần xác nhận gì?", "source_labels": ["S1"]}]},
])
def test_answer_schema_prevents_question_lists(fields):
    with pytest.raises(ValidationError):
        GroundedAnswer(**{"status": "insufficient_evidence", "claims": [],
                          "missing_information": [], **fields})


def test_supported_partial_answer_is_kept_before_clarification():
    question = "Bạn muốn xác nhận thông tin cư trú hay quyền sử dụng đất?"
    parsed = GroundedAnswer(status="answered", claims=[AnswerClaim(text="Phần có căn cứ.", source_labels=["S1"])],
                            missing_information=["Chưa đủ dữ kiện để kết luận trường hợp cụ thể."],
                            clarification_question=question)
    result = generator(parsed).generate("q", evidence())
    assert result["answer"].startswith("Phần có căn cứ. [S1]")
    assert result["answer"].endswith(question)
    assert result["answer"].count("?") == 1


def test_chat_service_retrieves_before_generation_and_can_omit_raw_evidence():
    retriever, model = MagicMock(), MagicMock()
    retriever.retrieve.return_value = evidence()
    model.generate.return_value = {"answer": "text"}
    result = LegalChatService(retriever, model).answer("  Câu hỏi  ", 3, include_evidence=False)
    retriever.retrieve.assert_called_once_with("Câu hỏi", 3)
    model.generate.assert_called_once_with("Câu hỏi", evidence())
    assert "evidence" not in result


def test_followup_retrieves_standalone_query_but_answers_latest_user_message():
    retriever, model = MagicMock(), MagicMock()
    history = [{"role": "user", "content": "Đất không có giấy tờ, có được cấp sổ không?"},
               {"role": "assistant", "content": "Bạn sử dụng đất từ năm nào?"}]
    model.rewrite_query.return_value = "Cấp sổ đỏ cho đất sử dụng từ năm 1995 không có giấy tờ"
    retriever.retrieve.return_value = evidence()
    model.generate.return_value = {"answer": "text"}
    result = LegalChatService(retriever, model).answer("Năm 1995", history=history, use_reranker=False)
    retriever.retrieve.assert_called_once_with(model.rewrite_query.return_value, 5, use_reranker=False)
    model.generate.assert_called_once_with("Năm 1995", evidence(), history=history,
                                           retrieval_query=model.rewrite_query.return_value)
    assert result["retrieval_query"] == model.rewrite_query.return_value


def test_generator_includes_history_as_context_but_only_maps_current_sources():
    history = [{"role": "assistant", "content": "Câu trả lời cũ [S99]"}]
    service = generator()
    result = service.generate("Câu hỏi nối tiếp", evidence(), history=history, retrieval_query="Truy vấn độc lập")
    payload = json.loads(service.client.responses.parse.call_args.kwargs["input"])
    assert payload["conversation_history"] == history
    assert payload["retrieval_query"] == "Truy vấn độc lập"
    assert payload["allowed_source_labels"] == ["S1"]
    assert result["citations"][0]["label"] == "S1"


def test_rewrite_preserves_user_context_without_answering_and_rejects_truncation():
    service = generator()
    history = [{"role": "user", "content": "Đất không có giấy tờ"}]
    service.client.responses.parse.return_value.output_parsed = RetrievalQuery(query="  Đất không có giấy tờ dùng từ năm 1995  ")
    assert service.rewrite_query("Năm 1995", history) == "Đất không có giấy tờ dùng từ năm 1995"
    kwargs = service.client.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is RetrievalQuery and kwargs["store"] is False
    assert json.loads(kwargs["input"])["latest_question"] == "Năm 1995"
    service.client.responses.parse.return_value.status = "incomplete"
    with pytest.raises(GenerationError, match="retrieval query"):
        service.rewrite_query("Năm 1995", history)


def test_chat_endpoint_returns_grounded_answer_and_validates_query():
    store, retriever = MagicMock(), MagicMock()
    retriever.retrieve.return_value = evidence()
    with patch("src.api.main.ChromaVectorStore.from_config", return_value=store), \
         patch("src.api.main.LegalRetriever.from_config", return_value=retriever), \
         patch("src.api.main.AnswerGenerator", return_value=generator()):
        with TestClient(app) as client:
            response = client.post("/chat", json={"query": "Câu hỏi", "include_evidence": False})
            assert response.status_code == 200
            assert "[S1]" in response.json()["answer"]
            assert client.post("/chat", json={"query": " "}).status_code == 422
