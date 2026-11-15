"""脱敏与敏感信息防护工具。

用途：
1. 导出/展示前对工具调用参数做长度限制与敏感字段掩码；
2. 对异常消息中的 Authorization / api_key 等做正则清洗，避免密钥进入日志或报告。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

_SECRET_PATTERNS: List[re.Pattern] = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{6,})", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(
        r"((?:api[_-]?key|authorization|token|secret|password)\s*[:=]\s*)(\S+)",
        re.IGNORECASE,
    ),
]

_SENSITIVE_ARG_KEYS = {
    "api_key",
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "credential",
    "credentials",
    "private_key",
}

_MAX_ARG_LENGTH = 500


def redact_text(text: str, max_length: int | None = 1000) -> str:
    """清洗文本中的密钥痕迹，并限制长度。"""
    if not isinstance(text, str):
        return ""
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            cleaned = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)
    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "…(truncated)"
    return cleaned


def redact_arguments(
    arguments: Mapping[str, Any], max_length: int | None = _MAX_ARG_LENGTH
) -> Dict[str, Any]:
    """对工具调用参数做脱敏：敏感键掩码、超长值截断。"""
    result: Dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        lowered = str(key).lower()
        if lowered in _SENSITIVE_ARG_KEYS:
            result[key] = "[REDACTED]"
            continue
        if isinstance(value, str):
            result[key] = redact_text(value, max_length=max_length)
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, dict)):
            text = str(value)
            result[key] = redact_text(text, max_length=max_length)
        else:
            result[key] = redact_text(str(value), max_length=max_length)
    return result


def safe_error(message: str) -> str:
    """生成可安全展示/落盘的错误信息。"""
    return redact_text(message, max_length=600)


def mask_key(key: str) -> str:
    """返回密钥的安全展示形式。"""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}{'*' * (len(key) - 7)}{key[-4:]}"
