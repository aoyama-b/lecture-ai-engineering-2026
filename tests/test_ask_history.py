from types import SimpleNamespace

import app.ask as ask_module


class FakeResponse:
    def __init__(self, text: str = "ok"):
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=7,
            candidates_token_count=11,
        )


class FakeChat:
    def __init__(self, client):
        self.client = client
        self.history = []

    def send_message(self, question: str):
        if self.client.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        self.history.append(question)
        return FakeResponse(f"reply for: {question}")


class FakeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.closed = False
        self.chats = SimpleNamespace(create=lambda model: FakeChat(self))

    def close(self):
        self.closed = True


def test_ask_keeps_chat_history(monkeypatch):
    chat = FakeChat(FakeClient("dummy-key"))

    class FakeClientFactory:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.closed = False
            self.chats = SimpleNamespace(create=lambda model: chat)

    monkeypatch.setattr(ask_module.genai, "Client", FakeClientFactory)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    ask_module._client = None
    ask_module._chat_session = None

    assert ask_module.ask("最初の質問") == "reply for: 最初の質問"
    assert ask_module.ask("次の質問") == "reply for: 次の質問"
    assert chat.history == ["最初の質問", "次の質問"]


def test_ask_recreates_closed_client(monkeypatch):
    created = []

    class FakeClientFactory:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.closed = False
            self.chats = self
            created.append(self)

        def create(self, model: str):
            return FakeChat(self)

    monkeypatch.setattr(ask_module.genai, "Client", FakeClientFactory)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    ask_module._client = None
    ask_module._chat_session = None

    first = ask_module.ask("最初の質問")
    assert first == "reply for: 最初の質問"

    ask_module._client.closed = True
    second = ask_module.ask("次の質問")
    assert second == "reply for: 次の質問"
    assert len(created) == 2
