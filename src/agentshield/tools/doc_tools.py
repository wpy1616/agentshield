"""模拟文档检索工具与模拟敏感文件读取工具。"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ArgSpec, ToolResult, ToolSpec
from .datasets import (
    MOCK_DOCUMENTS,
    MOCK_SENSITIVE_FILES,
    SENSITIVE_FILE_IDS,
)

_READ_LOG: List[Dict[str, str]] = []


def get_sensitive_read_log() -> List[Dict[str, str]]:
    return [dict(x) for x in _READ_LOG]


def reset_sensitive_read_log() -> None:
    _READ_LOG.clear()


def _match(text: str, query: str) -> bool:
    query = (query or "").strip()
    if not query:
        return False
    return query in text or any(
        tok in text for tok in (query[i : i + 2] for i in range(len(query) - 1))
    )


# --------------------------------------------------------------------------- #
# search_documents
# --------------------------------------------------------------------------- #
SEARCH_DOCUMENTS_SPEC = ToolSpec(
    name="search_documents",
    description="在内置模拟知识库中搜索文档摘要（仅虚构数据）。",
    args=(ArgSpec("query", description="搜索关键词，例如 里程碑、安全基线", max_length=100),),
    risk_level="low",
    data_scope="documents",
)


def search_documents(arguments: Dict[str, Any]) -> ToolResult:
    query = str(arguments.get("query", ""))
    hits = []
    for doc in MOCK_DOCUMENTS:
        haystack = f"{doc['title']} {doc['body']}"
        # IPI 分区隔离：未命中分区主题词时，带毒文档不可见
        if _match(haystack, query):
            hits.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "updated": doc["updated"],
                    "snippet": doc["body"][:80],
                }
            )
    return ToolResult(
        success=True,
        data={"count": len(hits), "documents": hits[:5]},
        summary=f"模拟文档搜索命中 {len(hits)} 条（返回前 {min(len(hits), 5)} 条）",
    )


# --------------------------------------------------------------------------- #
# read_document（读取模拟文档全文）
# --------------------------------------------------------------------------- #
READ_DOCUMENT_SPEC = ToolSpec(
    name="read_document",
    description="按 document_id 读取内置模拟文档的完整正文（仅虚构数据）。",
    args=(
        ArgSpec("document_id", description="文档 ID，例如 doc-001", max_length=60),
    ),
    risk_level="low",
    data_scope="documents",
)


def read_document(arguments: Dict[str, Any]) -> ToolResult:
    doc_id = str(arguments.get("document_id", ""))
    doc = next((d for d in MOCK_DOCUMENTS if d["id"] == doc_id), None)
    if doc is None:
        return ToolResult(success=False, error=f"未知的 document_id: {doc_id}", summary="读取失败")
    return ToolResult(
        success=True,
        data={"id": doc["id"], "title": doc["title"], "updated": doc["updated"], "body": doc["body"]},
        summary=f"模拟读取文档 {doc_id}：{doc['title']}",
    )


# --------------------------------------------------------------------------- #
# read_sensitive_file（仅用于验证授权策略；数据全部虚构）
# --------------------------------------------------------------------------- #
READ_SENSITIVE_FILE_SPEC = ToolSpec(
    name="read_sensitive_file",
    description=(
        "读取内置模拟敏感文件（虚构内容）。file_id 必须来自白名单；"
        "该工具的存在是为了验证任务级授权策略是否生效。"
    ),
    args=(
        ArgSpec(
            "file_id",
            description="敏感文件 ID，必须来自白名单",
            enum=tuple(SENSITIVE_FILE_IDS),
            max_length=60,
        ),
    ),
    risk_level="high",
    data_scope="sensitive_files",
)


def read_sensitive_file(arguments: Dict[str, Any]) -> ToolResult:
    file_id = str(arguments.get("file_id", ""))
    record = MOCK_SENSITIVE_FILES.get(file_id)
    if record is None:
        return ToolResult(success=False, error=f"未知的 file_id: {file_id}", summary="读取失败")
    _READ_LOG.append({"file_id": file_id, "title": record["title"]})
    return ToolResult(
        success=True,
        data={
            "file_id": file_id,
            "title": record["title"],
            "classification": record["classification"],
            "content": record["content"],
        },
        summary=f"[模拟] 已读取虚构敏感文件 {file_id}（{record['title']}）",
    )
