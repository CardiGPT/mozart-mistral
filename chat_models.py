from abc import ABC, abstractmethod
from typing import List, Optional, Any
from langchain.chat_models.base import BaseChatModel, BaseMessage, ChatResult, AIMessage, ChatGeneration, CallbackManagerForLLMRun

class AzureChatOpenAI(BaseChatModel, ABC):
    """
    Azure Chat OpenAI Model.
    """

    def __init__(self, deployment_name, openai_api_version, temperature, frequency_penalty, presence_penalty, top_p):
        self.deployment_name = deployment_name
        self.openai_api_version = openai_api_version
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p

    @abstractmethod
    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Simpler interface."""


class ChatOpenAI(BaseChatModel, ABC):
    """
    Chat OpenAI Model.
    """

    def __init__(self, model_name, temperature, frequency_penalty, presence_penalty, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p

    @abstractmethod
    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Simpler interface."""
