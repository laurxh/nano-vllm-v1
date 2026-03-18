from __future__ import annotations

import os
import json
import logging
import re
import time
import uuid
from threading import Lock

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from nanovllm import LLM, SamplingParams
from server.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatMessage,
    SimpleChatResponse,
    UsageInfo,
)
from server.session_store import SessionStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user directly and naturally in the same language as the user. "
    "Do not output <think> tags."
)

SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
)

STREAM_MIN_PENDING_CHARS = 12
STREAM_SAFE_BOUNDARY_CHARS = " \n\t，。！？；：,.!?;:"

logger = logging.getLogger("nanovllm.server")


class ChatService:
    def __init__(self, model: str, **engine_kwargs) -> None:
        self.model = model
        self.llm = LLM(model, **engine_kwargs)
        self.session_store = SessionStore()
        self._generate_lock = Lock()

    def build_messages(self, request: ChatCompletionRequest) -> list[ChatMessage]:
        base_messages = list(request.messages)
        if not base_messages or base_messages[0].role != "system":
            base_messages = [ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT)] + base_messages
        if request.session_id is None:
            return self._truncate_messages(base_messages, request.max_tokens)
        history = self.session_store.get_messages(request.session_id)
        history = [message for message in history if message.role != "system"]
        return self._truncate_messages(history + base_messages, request.max_tokens)

    def build_prompt(self, messages: list[ChatMessage]) -> str:
        payload = [message.model_dump() for message in messages]
        tokenizer = self.llm.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )

        lines = []
        for message in messages:
            lines.append(f"{message.role}: {message.content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def generate_text(self, prompt: str, request: ChatCompletionRequest) -> str:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        logger.info(
            "non_stream_request session_id=%s max_tokens=%s temperature=%s prompt=%r",
            request.session_id,
            request.max_tokens,
            request.temperature,
            prompt,
        )
        with self._generate_lock:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        raw_text = outputs[0]["text"]
        output_text = self.postprocess_output(raw_text)
        logger.info("non_stream_response raw=%r cleaned=%r", raw_text, output_text)
        return output_text

    def postprocess_output(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^<think>\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("</think>", "")
        for token in SPECIAL_TOKENS:
            text = text.replace(token, "")
        text = text.strip()
        return text

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenizer.encode(text))

    def _token_count_for_messages(self, messages: list[ChatMessage]) -> int:
        prompt = self.build_prompt(messages)
        return self.count_tokens(prompt)

    def _truncate_messages(self, messages: list[ChatMessage], max_tokens: int) -> list[ChatMessage]:
        if not messages:
            return messages

        system_message = messages[0] if messages[0].role == "system" else None
        non_system_messages = messages[1:] if system_message is not None else list(messages)
        max_model_len = self.llm.scheduler.max_model_len
        token_budget = max(max_model_len - max_tokens - 1, 1)

        kept_messages: list[ChatMessage] = []
        for message in reversed(non_system_messages):
            candidate_messages = ([] if system_message is None else [system_message]) + list(reversed([message] + kept_messages))
            if self._token_count_for_messages(candidate_messages) > token_budget:
                break
            kept_messages.append(message)

        kept_messages.reverse()
        truncated = ([] if system_message is None else [system_message]) + kept_messages

        if not truncated:
            return messages[-1:]
        return truncated

    def _stable_flush_length(self, pending_text: str) -> int:
        if len(pending_text) < STREAM_MIN_PENDING_CHARS:
            return 0
        last_boundary = -1
        for index, char in enumerate(pending_text):
            if char in STREAM_SAFE_BOUNDARY_CHARS:
                last_boundary = index + 1
        return last_boundary

    def _persist_session_response(self, request: ChatCompletionRequest, output_text: str) -> None:
        if request.session_id is None:
            return
        persisted_messages = list(request.messages)
        if not persisted_messages or persisted_messages[0].role != "system":
            persisted_messages = [ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT)] + persisted_messages
        self.session_store.append_messages(request.session_id, persisted_messages)
        self.session_store.append_messages(
            request.session_id,
            [ChatMessage(role="assistant", content=output_text)],
        )

    def create_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Only non-streaming chat completions are supported.")
        if not request.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty.")

        all_messages = self.build_messages(request)
        prompt = self.build_prompt(all_messages)
        output_text = self.generate_text(prompt, request)

        self._persist_session_response(request, output_text)

        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(output_text)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or self.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            session_id=request.session_id,
        )

    def create_simple_chat(self, request: ChatCompletionRequest) -> SimpleChatResponse:
        completion = self.create_completion(request)
        return SimpleChatResponse(
            content=completion.choices[0].message.content,
            session_id=completion.session_id,
        )

    def create_streaming_completion(self, request: ChatCompletionRequest) -> StreamingResponse:
        if not request.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty.")

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = request.model or self.model
        created = int(time.time())
        all_messages = self.build_messages(request)
        prompt = self.build_prompt(all_messages)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        def event_stream():
            with self._generate_lock:
                seq_id = self.llm.add_request(prompt, sampling_params)
                emitted_text = ""
                flushed_text = ""
                final_text = ""
                logger.info(
                    "stream_request seq_id=%s session_id=%s max_tokens=%s temperature=%s prompt=%r",
                    seq_id,
                    request.session_id,
                    request.max_tokens,
                    request.temperature,
                    prompt,
                )

                first_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                while True:
                    _, _, events = self.llm.step_with_events()
                    target_event = next((event for event in events if event["seq_id"] == seq_id), None)
                    if target_event is None:
                        continue

                    current_text = self.postprocess_output(
                        self.llm.tokenizer.decode(target_event["token_ids"])
                    )
                    final_text = current_text
                    if current_text.startswith(flushed_text):
                        pending_text = current_text[len(flushed_text):]
                    else:
                        pending_text = current_text
                        flushed_text = ""
                    emitted_text = current_text

                    flush_len = self._stable_flush_length(pending_text)
                    delta_text = pending_text[:flush_len] if flush_len > 0 else ""

                    if delta_text:
                        flushed_text += delta_text
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if target_event["is_finished"]:
                        remaining_text = final_text[len(flushed_text):] if final_text.startswith(flushed_text) else final_text
                        if remaining_text:
                            flushed_text = final_text
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": remaining_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        logger.info(
                            "stream_response seq_id=%s emitted=%r final=%r",
                            seq_id,
                            flushed_text if remaining_text == "" else final_text,
                            final_text,
                        )
                        self._persist_session_response(request, final_text)
                        final_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

        return StreamingResponse(event_stream(), media_type="text/event-stream")


def build_service_from_env() -> ChatService:
    model = os.getenv("NANOVLLM_MODEL")
    if not model:
        raise RuntimeError("NANOVLLM_MODEL environment variable is required.")

    tensor_parallel_size = int(os.getenv("NANOVLLM_TP_SIZE", "1"))
    max_num_batched_tokens = int(os.getenv("NANOVLLM_MAX_BATCHED_TOKENS", "2048"))
    max_num_seqs = int(os.getenv("NANOVLLM_MAX_NUM_SEQS", "256"))
    chunked_prefill = os.getenv("NANOVLLM_CHUNKED_PREFILL", "true").lower() == "true"
    enforce_eager = os.getenv("NANOVLLM_ENFORCE_EAGER", "false").lower() == "true"

    return ChatService(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        chunked_prefill=chunked_prefill,
        enforce_eager=enforce_eager,
    )
