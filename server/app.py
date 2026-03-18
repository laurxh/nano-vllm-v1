from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from server.schemas import ChatCompletionRequest, ChatCompletionResponse, SimpleChatResponse
from server.service import ChatService, build_service_from_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = build_service_from_env()
    app.state.chat_service = service
    try:
        yield
    finally:
        service.llm.exit()


app = FastAPI(title="nano-vLLM OpenAI-compatible server", lifespan=lifespan)


def get_chat_service() -> ChatService:
    return app.state.chat_service


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    service = get_chat_service()
    if request.stream:
        return service.create_streaming_completion(request)
    return service.create_completion(request)


@app.post("/chat", response_model=SimpleChatResponse)
async def create_simple_chat(request: ChatCompletionRequest) -> SimpleChatResponse:
    service = get_chat_service()
    return service.create_simple_chat(request)
