from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from vllm import LLM
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams

app = FastAPI()
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_control_vector=True,
    max_control_vectors=16,
    enable_lora=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.5,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ControlVectorData(BaseModel):
    name: str
    path: str
    scale: float
    target_degree: float
    keep_norm: bool


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 256
    control_vector: Optional[ControlVectorData] = None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        cv_request = None
        if request.control_vector:
            cv = request.control_vector
            cv_request = ControlVectorRequest(
                cv.name,
                abs(hash((cv.name, cv.target_degree))) % 999999,
                cv.path,
                cv.scale,
                cv.target_degree,
                cv.keep_norm,
            )

        result = llm.chat(
            [{"role": m.role, "content": m.content} for m in request.messages],
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        if not result or not result[0].outputs:
            raise HTTPException(status_code=500, detail="Model returned no output")

        return {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result[0].outputs[0].text,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        print(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
