"""AgentShield 模拟工具包。

本包内的所有工具都只操作内置虚构数据，不访问真实磁盘、邮箱或网络。
"""

from .base import ArgSpec, ToolResult, ToolSpec, scan_arguments
from .datasets import (
    MOCK_DOCUMENTS,
    MOCK_EMAILS,
    MOCK_SENSITIVE_FILES,
    SENSITIVE_FILE_IDS,
)
from .doc_tools import READ_SENSITIVE_FILE_SPEC, SEARCH_DOCUMENTS_SPEC
from .email_tools import (
    SEND_EMAIL_SPEC,
    SEARCH_EMAILS_SPEC,
    get_outbox,
    reset_outbox,
)
from .registry import DEFAULT_REGISTRY, ToolRegistry, build_default_registry

__all__ = [
    "ArgSpec",
    "ToolResult",
    "ToolSpec",
    "scan_arguments",
    "MOCK_EMAILS",
    "MOCK_DOCUMENTS",
    "MOCK_SENSITIVE_FILES",
    "SENSITIVE_FILE_IDS",
    "READ_SENSITIVE_FILE_SPEC",
    "SEARCH_DOCUMENTS_SPEC",
    "SEARCH_EMAILS_SPEC",
    "SEND_EMAIL_SPEC",
    "get_outbox",
    "reset_outbox",
    "ToolRegistry",
    "build_default_registry",
    "DEFAULT_REGISTRY",
]
