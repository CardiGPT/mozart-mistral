### Mozart Mistral Project: Python-based Language Model Interaction

This project is a Python-based application that uses the modal library to interact with a language model. The main functionality of the project is to generate text based on the input provided.
Dependencies

The project uses several Python libraries, including modal, torch, pydantic, sentence-transformers, and others. The specific versions of these dependencies can be found in the pyproject.toml file.

```
[tool.poetry]
name = "mozart-mistral"
version = "0.1.0"
description = "Call modal API"
authors = ["CardiGPT <150696547+CardiGPT@users.noreply.github.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
langchain = "^0.0.343"
modal = "^0.55.4147"
torch = "^1.10.0"
pydantic = "^2.5.2"
abc123abc = "^0.0.0"
jupyter = "^1.0.0"
python-dotenv = "*"
openai = "^1.3.4"
huggingface_hub = "*"
passlib = "*"
datasets = "*"
unstructured = "*"
pandas = "*"
duckduckgo-search = "*"
nltk = "*"
python-iso639 = "*"
requests = "*"
langdetect = "*"
python-magic = "*"
dataclasses-json = "*"
tabulate = "*"
beautifulsoup4 = "*"
chardet = "*"
emoji = "*"
numpy = "*"
lxml = "*"
filetype = "*"
jsonpatch = "<2.0,>=1.33"
langsmith = "<0.1.0,>=0.0.40"
anyio = "<4.0"
aiohttp = "<4.0.0,>=3.8.3"
fsspec = "*"
tiktoken = "*"
sentence-transformers = "^2.2.2"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Main Application

The main application is contained in app.py. It includes a FastAPI application with an endpoint for generating embeddings from a list of texts. The application also includes a MozartMistral class that uses the vllm library to generate text based on the input messages.

```
import os
import toml
from dotenv import load_dotenv
from typing import List
import modal
from modal import Stub, Image, Secret
from fastapi.responses import JSONResponse
import torch
import random
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
# from langchain_core.models import BaseChatModel
from typing import Optional, Any

load_dotenv()

NAME = 'embedding'
MODEL_NAME = "mozart-ai/BAAI__bge-small-en-v1.5__Mozart_Fine_Tuned-10"
MODEL_PATH = "/my-model"

def download_models():
    SentenceTransformer(
        MODEL_NAME, use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    ).save(MODEL_PATH)

stub = modal.Stub(
    name=NAME,
    image=Image.debian_slim().pip_install(
            "sentence-transformers",
            force_build=True
        ).run_function(
            download_models,
            secrets=[Secret.from_name("my-huggingface-secret")]
        )
)

if stub.is_inside():
    model = SentenceTransformer(MODEL_PATH)
    model.to('cuda')

class Texts(BaseModel):
    texts: List[str] = Field(..., min_items=1)

app = FastAPI()

@app.post("/embed")
def main(request_data: Texts):
    texts = request_data.texts
    _num_texts = len(texts)
    assert _num_texts > 0, "Must provide at least 1 texts in list to embed"

    try:
        if random.random() < 0.3:
            torch.cuda.empty_cache()
        embeddings = model.encode(texts).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
``§


Chat Models

The project includes several chat models, defined in chat_models.py and mozart_mistral.py. These models are used to generate text based on the input messages.
```
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
```

```
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
```

Running the Application

To run the application, you can use the local_main function in app.py. This function sends a request to the main endpoint with a sample text and prints the result.

```
@stub.local_entrypoint()
def local_main():
    request_data = {"texts": ["Wu-Tang Clan climbing Mount Everest"]}
    print(main(request_data))

if __name__ == "__main__":
    local_main()
```# mozart-mistral
