from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str  # "user" or "assistant"
    content: str


class AIProvider(ABC):
    """AI 提供商抽象基类"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """发送对话消息，返回 AI 回复文本"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """提供商显示名称"""
        ...


class ProviderFactory:
    """Provider 工厂，根据名称创建对应的 AI Provider 实例"""

    _registry: dict[str, type[AIProvider]] = {}

    @classmethod
    def register(cls, key: str, provider_cls: type[AIProvider]):
        cls._registry[key] = provider_cls

    @classmethod
    def create(cls, key: str, api_key: str, model: str) -> AIProvider:
        if key not in cls._registry:
            raise ValueError(f"未知的 AI 提供商: {key}，可选: {list(cls._registry.keys())}")
        return cls._registry[key](api_key=api_key, model=model)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
