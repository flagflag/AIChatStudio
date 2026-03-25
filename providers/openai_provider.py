import openai

from .base import AIProvider, Message, ProviderFactory


class OpenAIProvider(AIProvider):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def chat(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return response.choices[0].message.content


ProviderFactory.register("openai", OpenAIProvider)
