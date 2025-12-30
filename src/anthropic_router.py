from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_api_password
from log import log
from src.credential_manager import get_credential_manager

from .anthropic_converter import convert_anthropic_request_to_antigravity_components
from .anthropic_streaming import gemini_sse_to_anthropic_sse
from .gcli_chat_api import send_gemini_request
from .token_estimator import estimate_input_tokens


router = APIRouter()
security = HTTPBearer(auto_error=False)


def _sse_event(event: str, data: Dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _anthropic_error(
    *,
    status_code: int,
    message: str,
    error_type: str = "api_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _anthropic_error_stream(
    *,
    status_code: int,
    message: str,
    error_type: str = "api_error",
) -> StreamingResponse:
    async def error_stream():
        yield _sse_event(
            "error", {"type": "error", "error": {"type": error_type, "message": message}}
        )

    return StreamingResponse(
        error_stream(), media_type="text/event-stream", status_code=status_code
    )


def _extract_api_token(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials]
) -> Optional[str]:
    if credentials and credentials.credentials:
        return credentials.credentials

    authorization = request.headers.get("authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key.strip()

    return None


def _remove_nulls_for_tool_input(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            cleaned[k] = _remove_nulls_for_tool_input(v)
        return cleaned

    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            if item is None:
                continue
            cleaned_list.append(_remove_nulls_for_tool_input(item))
        return cleaned_list

    return value


def _pick_usage_metadata_from_gemini_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    response_usage = response_data.get("usageMetadata", {}) or {}
    if not isinstance(response_usage, dict):
        response_usage = {}

    candidate = (response_data.get("candidates", []) or [{}])[0] or {}
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_usage = candidate.get("usageMetadata", {}) or {}
    if not isinstance(candidate_usage, dict):
        candidate_usage = {}

    fields = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")

    def score(d: Dict[str, Any]) -> int:
        s = 0
        for f in fields:
            if f in d and d.get(f) is not None:
                s += 1
        return s

    if score(candidate_usage) > score(response_usage):
        return candidate_usage
    return response_usage


def _convert_gemini_response_to_anthropic_message(
    response_data: Dict[str, Any],
    *,
    model: str,
    message_id: str,
    fallback_input_tokens: int = 0,
) -> Dict[str, Any]:
    candidate = response_data.get("candidates", [{}])[0] or {}
    parts = candidate.get("content", {}).get("parts", []) or []
    usage_metadata = _pick_usage_metadata_from_gemini_response(response_data)

    content = []
    has_tool_use = False

    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("thought") is True:
            block: Dict[str, Any] = {"type": "thinking", "thinking": part.get("text", "")}
            signature = part.get("thoughtSignature")
            if signature:
                block["signature"] = signature
            content.append(block)
            continue

        if "text" in part:
            content.append({"type": "text", "text": part.get("text", "")})
            continue

        if "functionCall" in part:
            has_tool_use = True
            fc = part.get("functionCall", {}) or {}
            content.append(
                {
                    "type": "tool_use",
                    "id": fc.get("id") or f"toolu_{uuid.uuid4().hex}",
                    "name": fc.get("name") or "",
                    "input": _remove_nulls_for_tool_input(fc.get("args", {}) or {}),
                }
            )
            continue

        if "inlineData" in part:
            inline = part.get("inlineData", {}) or {}
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": inline.get("mimeType", "image/png"),
                        "data": inline.get("data", ""),
                    },
                }
            )
            continue

    finish_reason = candidate.get("finishReason")
    stop_reason = "tool_use" if has_tool_use else "end_turn"
    if finish_reason == "MAX_TOKENS" and not has_tool_use:
        stop_reason = "max_tokens"

    input_tokens_present = isinstance(usage_metadata, dict) and "promptTokenCount" in usage_metadata
    output_tokens_present = isinstance(usage_metadata, dict) and "candidatesTokenCount" in usage_metadata

    input_tokens = usage_metadata.get("promptTokenCount", 0) if isinstance(usage_metadata, dict) else 0
    output_tokens = usage_metadata.get("candidatesTokenCount", 0) if isinstance(usage_metadata, dict) else 0

    if not input_tokens_present:
        input_tokens = max(0, int(fallback_input_tokens or 0))
    if not output_tokens_present:
        output_tokens = 0

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        },
    }


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    password = await get_api_password()
    token = _extract_api_token(request, credentials)
    if token != password:
        return _anthropic_error(
            status_code=403, message="Invalid authentication credentials", error_type="authentication_error"
        )

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"Invalid JSON: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="Request body must be a JSON object", error_type="invalid_request_error"
        )

    model = payload.get("model")
    max_tokens = payload.get("max_tokens")
    messages = payload.get("messages")
    stream = bool(payload.get("stream", False))

    if not model or max_tokens is None or not isinstance(messages, list):
        return _anthropic_error(
            status_code=400,
            message="Missing required fields: model / max_tokens / messages",
            error_type="invalid_request_error",
        )

    if len(messages) == 1 and messages[0].get("role") == "user" and messages[0].get("content") == "Hi":
        return JSONResponse(
            content={
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": str(model),
                "content": [{"type": "text", "text": "gcli2api anthropic messages ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )

    cred_mgr = await get_credential_manager()

    try:
        components = convert_anthropic_request_to_antigravity_components(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] request conversion failed: {e}")
        return _anthropic_error(
            status_code=400, message="Request conversion failed", error_type="invalid_request_error"
        )

    log.info(f"[ANTHROPIC] /messages model mapping: upstream={model} -> downstream={components['model']}")

    if not str(components.get("model", "")).startswith("gemini-"):
        log.info(
            f"[ANTHROPIC] non-gemini model requested for Gemini backend, falling back to gemini-2.5-pro: {components.get('model')}"
        )
        components["model"] = "gemini-2.5-pro"

    if not (components.get("contents") or []):
        return _anthropic_error(
            status_code=400,
            message="messages cannot be empty; text blocks must contain non-whitespace text",
            error_type="invalid_request_error",
        )

    estimated_tokens = 0
    try:
        estimated_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.debug(f"[ANTHROPIC] token estimation failed: {e}")

    request_data: Dict[str, Any] = {
        "contents": components["contents"],
        "generationConfig": components["generation_config"],
    }
    if components.get("system_instruction"):
        request_data["systemInstruction"] = components["system_instruction"]
    if components.get("tools"):
        request_data["tools"] = components["tools"]

    api_payload = {"model": components["model"], "request": request_data}

    if stream:
        message_id = f"msg_{uuid.uuid4().hex}"
        response = await send_gemini_request(api_payload, True, cred_mgr)
        if getattr(response, "status_code", 200) != 200:
            return _anthropic_error_stream(
                status_code=getattr(response, "status_code", 500),
                message="Downstream request failed",
            )

        async def line_iter():
            if hasattr(response, "body_iterator"):
                async for chunk in response.body_iterator:
                    if not chunk:
                        continue
                    text = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
                    for line in text.splitlines():
                        if line:
                            yield line
            else:
                body = getattr(response, "body", None) or getattr(response, "content", None) or ""
                text = body.decode("utf-8", errors="ignore") if isinstance(body, bytes) else str(body)
                for line in text.splitlines():
                    if line:
                        yield line

        async def stream_generator():
            async for chunk in gemini_sse_to_anthropic_sse(
                line_iter(),
                model=str(model),
                message_id=message_id,
                initial_input_tokens=estimated_tokens,
            ):
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    response = await send_gemini_request(api_payload, False, cred_mgr)
    if getattr(response, "status_code", 200) != 200:
        message = "Downstream request failed"
        try:
            body = getattr(response, "body", None) or getattr(response, "content", None)
            if body:
                raw = body.decode("utf-8", errors="ignore") if isinstance(body, bytes) else str(body)
                err_json = json.loads(raw)
                if isinstance(err_json, dict):
                    message = err_json.get("error", {}).get("message", message)
        except Exception:
            pass
        return _anthropic_error(status_code=getattr(response, "status_code", 500), message=message)

    request_id = f"msg_{int(time.time() * 1000)}"
    try:
        if hasattr(response, "body"):
            response_data = json.loads(
                response.body.decode() if isinstance(response.body, bytes) else response.body
            )
        else:
            response_data = json.loads(
                response.content.decode() if isinstance(response.content, bytes) else response.content
            )
    except Exception as e:
        log.error(f"[ANTHROPIC] response parse failed: {e}")
        return _anthropic_error(status_code=500, message="Response conversion failed")

    anthropic_response = _convert_gemini_response_to_anthropic_message(
        response_data,
        model=str(model),
        message_id=request_id,
        fallback_input_tokens=estimated_tokens,
    )
    return JSONResponse(content=anthropic_response)


@router.post("/v1/messages/count_tokens")
async def anthropic_messages_count_tokens(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    password = await get_api_password()
    token = _extract_api_token(request, credentials)
    if token != password:
        return _anthropic_error(
            status_code=403, message="Invalid authentication credentials", error_type="authentication_error"
        )

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"Invalid JSON: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="Request body must be a JSON object", error_type="invalid_request_error"
        )

    if not payload.get("model") or not isinstance(payload.get("messages"), list):
        return _anthropic_error(
            status_code=400,
            message="Missing required fields: model / messages",
            error_type="invalid_request_error",
        )

    input_tokens = 0
    try:
        input_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] token estimation failed: {e}")

    return JSONResponse(content={"input_tokens": input_tokens})
