# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi[standard]",
#   "boto3",
# ]
# ///
import re
import os
import time
import uuid
import json
import asyncio
import logging
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Union
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

app = FastAPI(title="Nova Grounding → Perplexity Wrapper")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("nova_wrapper")

_API_KEY = os.getenv("API_KEY")  # optional; if set, all requests must supply Bearer token

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _auth(request: Request, call_next):
    if _API_KEY and request.url.path not in ("/health",):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_API_KEY}":
            return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key", "type": "invalid_request_error"}})
    return await call_next(request)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    config=Config(read_timeout=300, connect_timeout=10)
)

_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
log.info("semaphore initialized max_concurrent=%d", _MAX_CONCURRENT)


async def _try_acquire() -> bool:
    """Acquire semaphore slot; return False immediately if at capacity."""
    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=0.05)
        return True
    except asyncio.TimeoutError:
        return False

# ---------------------------------------------------------------------------
# Model config — all aliases default to Nova 2 Lite
# Pass us.amazon.* CRIS IDs directly, or amazon.* (us. prefix added automatically)
# ---------------------------------------------------------------------------
_NOVA_PREMIER = "us.amazon.nova-premier-v1:0"
_NOVA_PRO     = "us.amazon.nova-pro-v1:0"
_NOVA_LITE    = "us.amazon.nova-lite-v1:0"
_NOVA_MICRO   = "us.amazon.nova-micro-v1:0"
_NOVA_2_LITE  = "us.amazon.nova-2-lite-v1:0"

MODEL_MAP: dict[str, str] = {
    "nova-premier-web-grounding": _NOVA_2_LITE,
    "sonar-pro":                  _NOVA_2_LITE,
    "sonar-pro-online":           _NOVA_2_LITE,
    "sonar-reasoning-pro":        _NOVA_2_LITE,
    "sonar-deep-research":        _NOVA_2_LITE,
    "sonar":                      _NOVA_2_LITE,
    "sonar-online":               _NOVA_2_LITE,
    "sonar-reasoning":            _NOVA_2_LITE,
    "sonar-turbo":                _NOVA_2_LITE,
    "sonar-mini":                 _NOVA_2_LITE,
    "nova-premier-grounding":     _NOVA_2_LITE,
    "nova-pro-grounding":         _NOVA_2_LITE,
    "nova-lite-grounding":        _NOVA_2_LITE,
    "nova-micro-grounding":       _NOVA_2_LITE,
    "nova-2-lite-grounding":      _NOVA_2_LITE,
}
DEFAULT_MODEL_ID = _NOVA_2_LITE

# ---------------------------------------------------------------------------
# Bedrock / inference config
# ---------------------------------------------------------------------------
TOOL_CONFIG = {"tools": [{"systemTool": {"name": "nova_grounding"}}]}
THINKING_RE = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text or "" for p in self.content if p.type == "text")


class ChatRequest(BaseModel):
    model: str = "nova-premier-grounding"
    messages: List[Message]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # OpenAI alias for max_tokens
    stream: Optional[bool] = False
    stream_options: Optional[dict] = None  # accepted, usage always included
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    logprobs: Optional[bool] = None  # accepted (no-op)
    top_logprobs: Optional[int] = None  # accepted (no-op)
    response_format: Optional[dict] = None  # accepted (no-op)
    # OpenAI-compatible no-ops
    n: Optional[int] = 1
    seed: Optional[int] = None
    user: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    # Perplexity-compatible fields (accepted, not enforced by Nova)
    search_domain_filter: Optional[List[str]] = None
    return_images: Optional[bool] = False
    return_related_questions: Optional[bool] = False
    search_recency_filter: Optional[str] = None


class ThinkingStripper:
    """Stream-safe stripper for <thinking>...</thinking> blocks."""

    def __init__(self):
        self._buf = ""
        self._skipping = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while self._buf:
            if self._skipping:
                idx = self._buf.find("</thinking>")
                if idx == -1:
                    # Keep tail in case closing tag spans chunk boundary
                    self._buf = self._buf[-10:]
                    break
                self._buf = self._buf[idx + 11:]
                self._skipping = False
            else:
                idx = self._buf.find("<thinking>")
                if idx == -1:
                    safe = max(0, len(self._buf) - 9)
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + 10:]
                self._skipping = True
        return "".join(out)

    def flush(self) -> str:
        result = "" if self._skipping else self._buf
        self._buf = ""
        return result


def _build_kwargs(req: ChatRequest) -> dict:
    if req.model.startswith("us.amazon."):
        model_id = req.model
    elif req.model.startswith("amazon."):
        model_id = "us." + req.model
    else:
        model_id = MODEL_MAP.get(req.model, DEFAULT_MODEL_ID)
    system, messages = [], []
    for m in req.messages:
        if m.role == "system":
            system.append({"text": m.text()})
        else:
            messages.append({"role": m.role, "content": [{"text": m.text()}]})

    inference_config = {"maxTokens": req.max_completion_tokens or req.max_tokens or 4096}
    if req.temperature is not None:
        inference_config["temperature"] = req.temperature
    if req.top_p is not None:
        inference_config["topP"] = req.top_p
    if req.stop:
        inference_config["stopSequences"] = [req.stop] if isinstance(req.stop, str) else req.stop

    kwargs = dict(
        modelId=model_id,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        inferenceConfig=inference_config,
    )
    if system:
        kwargs["system"] = system
    return kwargs


def _extract(content_list: list) -> tuple[str, list[str]]:
    text_parts, citations = [], []
    for block in content_list:
        if "text" in block:
            text_parts.append(block["text"])
        if "citationsContent" in block:
            for c in block["citationsContent"].get("citations", []):
                url = c.get("location", {}).get("web", {}).get("url", "")
                if url and url not in citations:
                    citations.append(url)
    return THINKING_RE.sub("", "".join(text_parts)).strip(), citations


# ---------------------------------------------------------------------------
# Response / error mapping
# ---------------------------------------------------------------------------
_STOP_REASON = {
    "end_turn":      "stop",
    "stop_sequence": "stop",
    "max_tokens":    "length",
    "tool_use":      "tool_calls",
}


def _finish_reason(nova_reason: str) -> str:
    return _STOP_REASON.get(nova_reason, "stop")


def _openai_error(status: int, message: str, err_type: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


_AWS_STATUS = {
    "ThrottlingException": 429,
    "ValidationException": 400,
    "AccessDeniedException": 403,
    "ResourceNotFoundException": 404,
    "ModelNotReadyException": 503,
    "ServiceUnavailableException": 503,
}


def _bedrock_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, ClientError):
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        return _openai_error(_AWS_STATUS.get(code, 500), msg, err_type=code)
    return _openai_error(500, str(exc))


def _exc_to_error_dict(exc: Exception) -> dict:
    if isinstance(exc, ClientError):
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        return {"message": msg, "type": code, "code": _AWS_STATUS.get(code, 500)}
    return {"message": str(exc), "type": "server_error", "code": 500}


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return _openai_error(500, str(exc))


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    if not await _try_acquire():
        log.warning("rate limit: at capacity (%d concurrent)", _MAX_CONCURRENT)
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "5"},
            content={"error": {"message": f"Too many concurrent requests (max {_MAX_CONCURRENT})", "type": "rate_limit_error"}},
        )

    kwargs = _build_kwargs(req)
    log.info("request model=%s messages=%d stream=%s", req.model, len(req.messages), req.stream)

    if req.stream:
        # semaphore released inside _stream's finally block
        return StreamingResponse(_stream(req, kwargs), media_type="text/event-stream")

    try:
        resp = await asyncio.to_thread(bedrock.converse, **kwargs)
    except Exception as e:
        log.error("bedrock error: %s", e)
        return _bedrock_error(e)
    finally:
        _semaphore.release()

    text, citations = _extract(resp["output"]["message"]["content"])
    usage = resp.get("usage", {})
    finish = _finish_reason(resp.get("stopReason", "end_turn"))
    log.info("done tokens=%d citations=%d finish=%s", usage.get("totalTokens", 0), len(citations), finish)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": "nova-premier-v1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
            "citation_tokens": len(citations) * 80,
            "num_search_queries": 1,
        },
        "citations": citations,
    }


async def _stream(req: ChatRequest, kwargs: dict):
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    stripper = ThinkingStripper()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            resp = bedrock.converse_stream(**kwargs)
            for event in resp["stream"]:
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=_producer, daemon=True).start()
    log.info("stream start model=%s", req.model)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        yield sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        })

        usage = {}
        stop_reason = "end_turn"
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                err = _exc_to_error_dict(payload)
                log.error("stream error: %s", err["message"])
                yield sse({"error": err})
                yield "data: [DONE]\n\n"
                return
            event = payload
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"].get("delta", {}).get("text", "")
                if text:
                    clean = stripper.feed(text)
                    if clean:
                        yield sse({
                            "id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {"content": clean}, "finish_reason": None}],
                        })
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})

        tail = stripper.flush()
        if tail:
            yield sse({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
            })

        yield sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(stop_reason)}],
            "usage": {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            },
        })
        log.info("stream done tokens=%d", usage.get("totalTokens", 0))
        yield "data: [DONE]\n\n"
    finally:
        _semaphore.release()


@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "amazon", "created": 0}
            for k in MODEL_MAP
        ],
    }


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
def get_model(model_id: str):
    # Unknown aliases fall back to nova-premier (consistent with _build_kwargs)
    return {"id": model_id, "object": "model", "owned_by": "amazon", "created": 0,
            "root": DEFAULT_MODEL_ID}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "7000")))


def run():
    """Entry point for uvx / uv tool install."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "7000")))
