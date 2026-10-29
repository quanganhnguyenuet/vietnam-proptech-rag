import json
from unittest.mock import MagicMock

import pytest

from chat import run_chat_loop
from src.rag.generator import GenerationError
from src.rag.session import ChatSession


def test_session_carries_successful_turns_and_does_not_keep_failed_attempts():
    service = MagicMock()
    service.answer.side_effect = [{"answer": "Bạn sử dụng đất từ năm nào?"},
                                  GenerationError("temporary failure"), {"answer": "Phần trả lời."}]
    session = ChatSession(service)
    session.answer("  Tôi muốn cấp sổ đỏ  ")
    expected = [{"role": "user", "content": "Tôi muốn cấp sổ đỏ"},
                {"role": "assistant", "content": "Bạn sử dụng đất từ năm nào?"}]
    with pytest.raises(GenerationError):
        session.answer("Năm 1995")
    assert session.history == expected
    session.answer("Năm 1995")
    assert service.answer.call_args.kwargs["history"] == expected
    assert len(session.history) == 4
    assert ChatSession(service).history == []


def test_session_bounds_memory_by_complete_turns_and_characters():
    service = MagicMock()
    service.answer.return_value = {"answer": "abcd"}
    session = ChatSession(service, max_turns=2, max_history_chars=12)
    for question in ("q1", "q2", "q3"):
        session.answer(question)
    assert [row["content"] for row in session.history] == ["q2", "abcd", "q3", "abcd"]
    session.answer("longer")
    assert [row["content"] for row in session.history] == ["longer", "abcd"]


def test_loop_skips_blank_input_remembers_context_and_stops_on_end(monkeypatch, tmp_path, capsys):
    service = MagicMock()
    service.answer.side_effect = [{"answer": "Bạn sử dụng đất từ năm nào?"}, {"answer": "Đã nhận năm 1995."}]
    inputs = iter(["  ", "Tôi muốn cấp sổ đỏ", "Năm 1995", " EnD ", "Không được xử lý"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    output = tmp_path / "conversation.json"
    run_chat_loop(service, output=output)
    assert service.answer.call_count == 2
    assert service.answer.call_args.kwargs["history"][0]["content"] == "Tôi muốn cấp sổ đỏ"
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [turn["query"] for turn in saved["turns"]] == ["Tôi muốn cấp sổ đỏ", "Năm 1995"]
    assert "Đã kết thúc" in capsys.readouterr().out


def test_loop_keeps_running_after_model_failure(monkeypatch):
    service = MagicMock()
    service.answer.side_effect = [GenerationError("retry"), {"answer": "Thành công."}]
    inputs = iter(["retry", "end"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    run_chat_loop(service, initial_query="first")
    assert service.answer.call_count == 2
    assert service.answer.call_args.kwargs["history"] == []


@pytest.mark.parametrize("termination", [EOFError, KeyboardInterrupt])
def test_loop_can_exit_without_a_paid_request(monkeypatch, termination):
    service = MagicMock()
    def stop(_):
        raise termination
    monkeypatch.setattr("builtins.input", stop)
    run_chat_loop(service)
    service.answer.assert_not_called()
