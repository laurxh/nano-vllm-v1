from __future__ import annotations

from collections import defaultdict
from threading import Lock

from server.schemas import ChatMessage


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[ChatMessage]] = defaultdict(list)
        self._lock = Lock()

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        with self._lock:
            self._sessions[session_id].extend(messages)
