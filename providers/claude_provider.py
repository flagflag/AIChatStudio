import anthropic

from .base import AIProvider, Message, ProviderFactory


class ClaudeProvider(AIProvider):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Claude ({self.model})"

    def chat(self, messages: list[Message]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return response.content[0].text


ProviderFactory.register("claude", ClaudeProvider)
