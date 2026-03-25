from google import genai
from google.genai import types

from .base import AIProvider, Message, ProviderFactory


class GeminiProvider(AIProvider):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def chat(self, messages: list[Message]) -> str:
        # 将通用 messages 格式转为 Gemini contents 格式
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        return response.text


ProviderFactory.register("gemini", GeminiProvider)
