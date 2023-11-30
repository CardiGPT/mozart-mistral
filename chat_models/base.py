from abc import ABC, abstractmethod
from typing import List, Optional, Any
from pydantic import BaseModel

class BaseMessage(BaseModel):
    """
    Base Message Model.
    """
    content: str


class AIMessage(BaseModel):
    """
    AI Message Model.
    """
    content: str


class ChatGeneration(BaseModel):
    """
    Chat Generation Model.
    """
    message: AIMessage


class ChatResult(BaseModel):
    """
    Chat Result Model.
    """
    generations: List[ChatGeneration]


class CallbackManagerForLLMRun:
    """
    Callback Manager for LLM Run.
    """
    pass


class BaseChatModel(ABC):
    """
    Base Chat Model.
    """

    @abstractmethod
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate method."""

    @abstractmethod
    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Simpler interface."""
