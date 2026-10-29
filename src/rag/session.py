"""In-memory conversation for one terminal session, isolated from API requests."""
from src.rag.service import LegalChatService
from src.retrieval.query_normalizer import normalize_query


class ChatSession:
    def __init__(self, service: LegalChatService, *, max_turns: int = 6, max_history_chars: int = 12000):
        if max_turns < 1 or max_history_chars < 1:
            raise ValueError("Conversation limits must be positive")
        self.service = service
        self.max_turns = max_turns
        self.max_history_chars = max_history_chars
        self._turns: list[tuple[str, str]] = []

    @property
    def history(self) -> list[dict]:
        return [message for question, answer in self._turns for message in (
            {"role": "user", "content": question}, {"role": "assistant", "content": answer})]

    def answer(self, query: str, top_k: int = 5, **options) -> dict:
        question = normalize_query(query).normalized
        result = self.service.answer(question, top_k, history=self.history, **options)
        # Only completed turns enter memory; failed attempts can be retried.
        self._turns.append((question, result["answer"]))
        while len(self._turns) > self.max_turns or sum(len(q) + len(a) for q, a in self._turns) > self.max_history_chars:
            self._turns.pop(0)
        return result
