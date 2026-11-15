"""工具注册表：集中管理所有可用模拟工具。

扩展方式：实现 ToolSpec + handler，然后调用 `register` 即可被
安全策略、Agent 提示词和 UI 自动识别。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .base import ToolHandler, ToolResult, ToolSpec, scan_arguments
from .datasets import TOOL_DATA_SCOPE
from .doc_tools import (
    READ_DOCUMENT_SPEC,
    READ_SENSITIVE_FILE_SPEC,
    SEARCH_DOCUMENTS_SPEC,
    read_document,
    read_sensitive_file,
    search_documents,
)
from .email_tools import (
    READ_EMAIL_SPEC,
    SEND_EMAIL_SPEC,
    SEARCH_EMAILS_SPEC,
    read_email,
    search_emails,
    send_email,
)


class ToolRegistry:
    """模拟工具的注册表（默认全局单例）。"""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, ToolHandler] = {}

    # ---- 注册与查询 ----
    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._specs:
            raise ValueError(f"工具已注册：{spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def names(self) -> List[str]:
        return sorted(self._specs)

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def specs(self) -> List[ToolSpec]:
        return [self._specs[n] for n in self.names()]

    def has(self, name: str) -> bool:
        return name in self._specs

    def data_scope(self, name: str) -> str:
        spec = self._specs.get(name)
        if spec and spec.data_scope:
            return spec.data_scope
        return TOOL_DATA_SCOPE.get(name, "")

    # ---- 校验与执行 ----
    def validate(self, name: str, arguments: Dict[str, Any]) -> tuple[bool, str]:
        """校验工具名 + 参数 + 危险模式扫描。返回 (合法, 原因)。"""
        spec = self._specs.get(name)
        if spec is None:
            return False, f"未知工具 `{name}`，不在白名单内"
        ok, reason = spec.validate(arguments)
        if not ok:
            return False, reason
        danger = scan_arguments(arguments)
        if danger:
            return False, danger
        return True, ""

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行模拟工具。调用方必须已完成策略检查。"""
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(success=False, error=f"未知工具 `{name}`", summary="执行失败")
        try:
            result = handler(arguments)
        except Exception as exc:  # 模拟工具出错不应导致实验崩溃
            return ToolResult(
                success=False,
                error=f"模拟工具执行异常：{type(exc).__name__}: {exc}",
                summary="执行失败",
            )
        result.simulated = True
        return result

    def prompt_catalog(self) -> str:
        """生成给模型的工具清单文本。"""
        return "\n\n".join(spec.prompt_description() for spec in self.specs())

    def to_dict(self) -> Dict[str, Any]:
        return {"tools": [s.to_dict() for s in self.specs()]}


def build_default_registry() -> ToolRegistry:
    """构建内置模拟工具注册表。

    工具集：
    - search_emails / read_email：检索与读取虚构邮件（read_email 可返回全文，
      是真 IPI 观察通道的关键入口）；
    - search_documents / read_document：检索与读取虚构文档；
    - read_sensitive_file：读取合成敏感文件，用于验证任务级授权；
    - send_email：只写内存待发件列表，永不真实发送。
    """
    registry = ToolRegistry()
    registry.register(SEARCH_EMAILS_SPEC, search_emails)
    registry.register(READ_EMAIL_SPEC, read_email)
    registry.register(SEARCH_DOCUMENTS_SPEC, search_documents)
    registry.register(READ_DOCUMENT_SPEC, read_document)
    registry.register(READ_SENSITIVE_FILE_SPEC, read_sensitive_file)
    registry.register(SEND_EMAIL_SPEC, send_email)
    return registry


DEFAULT_REGISTRY = build_default_registry()
