import os
import toml
from dotenv import load_dotenv
import modal
from modal import Image, Secret
from typing import List
from modal import web_endpoint
from fastapi.responses import JSONResponse
import torch
import random
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from chat_models.base import BaseMessage
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

@app.get("/health")
def health():
    return {"status": "healthy"}

@stub.local_entrypoint()
def local_main():
    request_data = {"texts": ["Wu-Tang Clan climbing Mount Everest"]}
    print(main(request_data))

if __name__ == "__main__":
    local_main()

class MozartMistral(BaseChatModel, ABC):
    def __init__(self):
        self.llm = LLM(MODEL_PATH, secret=Secret.from_name("mozart-secret"))

    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        prompts = [message.content for message in messages]
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        return result