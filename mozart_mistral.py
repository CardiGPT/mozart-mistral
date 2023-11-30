from abc import ABC
from typing import List, Optional, Any
from langchain.chat_models.base import BaseChatModel, BaseMessage, ChatResult, AIMessage, ChatGeneration, CallbackManagerForLLMRun
import requests

class MozartMistral(BaseChatModel, ABC):
    """
    Override _generate method with api call
    to custom model endpoint (modal gpu)
    """

    def __init__(self, api_url):
        self.api_url = api_url

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        data = {"messages": [message.to_dict() for message in messages], "stop": stop, **kwargs}
        response = requests.post(self.api_url, json=data)
        if response.status_code != 200:
            raise Exception(f"Request to model endpoint failed with status {response.status_code}")
        return response.text
