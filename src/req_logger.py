"""
统一请求日志模块 - 为所有 router 提供一致的请求日志记录
"""

from typing import Any, Dict, Optional

from fastapi import Request

from log import log


def _mask_api_key(api_key: Optional[str], visible_chars: int = 8) -> str:
    """脱敏 API Key，仅显示前几位"""
    if not api_key:
        return "None"
    if len(api_key) <= visible_chars:
        return api_key
    return f"{api_key[:visible_chars]}..."


def _get_client_info(request: Request) -> tuple[str, str]:
    """获取客户端信息"""
    try:
        client_host = request.client.host if request.client else "unknown"
        client_port = str(request.client.port) if request.client else "unknown"
    except Exception:
        client_host = "unknown"
        client_port = "unknown"
    return client_host, client_port


async def log_incoming_request(
    router_name: str,
    request: Request,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
) -> None:
    """
    统一记录入站请求日志

    Args:
        router_name: 路由名称，如 "OpenAI", "Gemini", "Antigravity"
        request: FastAPI Request 对象
        api_key: API 密钥（会自动脱敏）
        model: 请求的模型名称
        body: 请求体（用于提取额外信息）
    """
    client_host, client_port = _get_client_info(request)
    user_agent = request.headers.get("user-agent", "")
    masked_key = _mask_api_key(api_key)

    # 提取流式标志
    stream = False
    if body:
        stream = body.get("stream", False)

    # 提取消息数量
    msg_count = 0
    if body:
        if "messages" in body:
            msg_count = len(body.get("messages", []))
        elif "contents" in body:
            msg_count = len(body.get("contents", []))

    log.info(
        f"[{router_name}] 收到请求: "
        f"client={client_host}:{client_port}, "
        f"model={model}, "
        f"stream={stream}, "
        f"messages={msg_count}, "
        f"key={masked_key}, "
        f"ua={user_agent}"
    )
