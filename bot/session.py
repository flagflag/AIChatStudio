import time

from providers.base import Message


class Session:
    """单个用户的会话上下文"""

    def __init__(self, max_history: int):
        self.messages: list[Message] = []
        self.max_history = max_history
        self.last_active = time.time()

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()
        self.last_active = time.time()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()
        self.last_active = time.time()

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self):
        self.messages.clear()

    def _trim(self):
        # 保留最近 max_history * 2 条消息（每轮 = 1 user + 1 assistant）
        max_count = self.max_history * 2
        if len(self.messages) > max_count:
            self.messages = self.messages[-max_count:]


class SessionManager:
    """管理所有用户的会话"""

    def __init__(self, max_history: int = 20, timeout_minutes: int = 30):
        self.max_history = max_history
        self.timeout_seconds = timeout_minutes * 60
        self._sessions: dict[str, Session] = {}

    def _make_key(self, chat_id: str, user_id: str) -> str:
        return f"{chat_id}:{user_id}"

    def get(self, chat_id: str, user_id: str) -> Session:
        key = self._make_key(chat_id, user_id)
        self._cleanup_expired()
        if key not in self._sessions:
            self._sessions[key] = Session(self.max_history)
        return self._sessions[key]

    def clear(self, chat_id: str, user_id: str):
        key = self._make_key(chat_id, user_id)
        if key in self._sessions:
            self._sessions[key].clear()

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            k for k, s in self._sessions.items()
            if now - s.last_active > self.timeout_seconds
        ]
        for k in expired:
            del self._sessions[k]
