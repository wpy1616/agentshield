"""模拟邮件工具：search_emails 与 send_email（均为模拟，不联网）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import ArgSpec, ToolResult, ToolSpec
from .datasets import (
    MOCK_EMAILS,
    is_internal_recipient,
)

_TOKEN_RE = re.compile(r"[\w一-鿿]+")

# 模拟待发送邮件列表（仅存在于内存，绝不发起网络请求）
_OUTBOX: List[Dict[str, str]] = []


def get_outbox() -> List[Dict[str, str]]:
    """返回模拟待发件列表的副本。"""
    return [dict(item) for item in _OUTBOX]


def reset_outbox() -> None:
    """清空模拟待发件列表（测试用）。"""
    _OUTBOX.clear()


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text or "") if len(t) >= 2]


def _match(record: Dict[str, str], query: str) -> bool:
    haystack = " ".join(
        str(record.get(k, "")) for k in ("subject", "body", "sender", "date")
    )
    query = (query or "").strip()
    if not query:
        return False
    if query in haystack:
        return True
    return any(tok in haystack for tok in _tokens(query))


# --------------------------------------------------------------------------- #
# search_emails
# --------------------------------------------------------------------------- #
SEARCH_EMAILS_SPEC = ToolSpec(
    name="search_emails",
    description="在内置模拟邮箱中搜索邮件摘要（仅虚构数据，不访问真实邮箱）。",
    args=(
        ArgSpec("query", description="搜索关键词，例如 会议、进度、预算", max_length=100),
        ArgSpec(
            "sender", required=False, description="可选：按发件人过滤，如 li.wei@example-corp.com"
        ),
    ),
    risk_level="low",
    data_scope="emails",
)


def search_emails(arguments: Dict[str, Any]) -> ToolResult:
    query = str(arguments.get("query", ""))
    sender = arguments.get("sender")
    hits: List[Dict[str, str]] = []
    for mail in MOCK_EMAILS:
        if sender and sender.lower() not in mail["sender"].lower():
            continue
        # IPI 分区隔离：未命中分区主题词时，带毒邮件不可见
        if _match(mail, query):
            hits.append(
                {
                    "id": mail["id"],
                    "sender": mail["sender"],
                    "subject": mail["subject"],
                    "date": mail["date"],
                    "snippet": mail["body"][:80],
                }
            )
    return ToolResult(
        success=True,
        data={"count": len(hits), "emails": hits[:5]},
        summary=f"模拟邮件搜索命中 {len(hits)} 条（返回前 {min(len(hits), 5)} 条）",
    )


# --------------------------------------------------------------------------- #
# read_email（读取模拟邮件全文）
# --------------------------------------------------------------------------- #
READ_EMAIL_SPEC = ToolSpec(
    name="read_email",
    description="按 mail_id 读取内置模拟邮件的完整正文（仅虚构数据，不访问真实邮箱）。",
    args=(ArgSpec("mail_id", description="邮件 ID，例如 mail-001", max_length=60),),
    risk_level="low",
    data_scope="emails",
)


def read_email(arguments: Dict[str, Any]) -> ToolResult:
    mail_id = str(arguments.get("mail_id", ""))
    mail = next((m for m in MOCK_EMAILS if m["id"] == mail_id), None)
    if mail is None:
        return ToolResult(success=False, error=f"未知的 mail_id: {mail_id}", summary="读取失败")
    return ToolResult(
        success=True,
        data={
            "id": mail["id"],
            "sender": mail["sender"],
            "subject": mail["subject"],
            "date": mail["date"],
            "body": mail["body"],
        },
        summary=f"模拟读取邮件 {mail_id}：{mail['subject']}",
    )


# --------------------------------------------------------------------------- #
# send_email（模拟）
# --------------------------------------------------------------------------- #
SEND_EMAIL_SPEC = ToolSpec(
    name="send_email",
    description="模拟发送邮件：仅写入内存中的模拟待发件列表，不会真正发送任何邮件。",
    args=(
        ArgSpec("to", description="收件人地址（模拟）", max_length=120),
        ArgSpec("subject", description="邮件主题", max_length=200),
        ArgSpec("body", description="邮件正文", max_length=1000),
    ),
    risk_level="high",
    side_effect=True,
    data_scope="emails",
)


def send_email(arguments: Dict[str, Any]) -> ToolResult:
    to = str(arguments.get("to", ""))
    subject = str(arguments.get("subject", ""))
    body = str(arguments.get("body", ""))
    record = {
        "to": to,
        "subject": subject,
        "body": body,
        "internal_recipient": is_internal_recipient(to),
    }
    _OUTBOX.append(record)
    return ToolResult(
        success=True,
        data={"queued": True, "outbox_size": len(_OUTBOX), **record},
        summary=f"[模拟] 已写入模拟待发件列表，收件人={to}（不会真正发送邮件）",
    )
