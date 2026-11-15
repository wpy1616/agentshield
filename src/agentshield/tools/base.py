"""模拟工具的基础设施：参数规格、安全扫描、统一结果格式。

设计要点：
- 所有工具只操作内置虚构数据，不触碰真实磁盘/网络/邮箱；
- 参数在交给工具函数之前必须完成 schema 校验 + 危险模式扫描；
- 工具执行结果统一为 ToolResult（success / data / error / simulated）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 危险模式扫描（纵深防御）
# --------------------------------------------------------------------------- #
# 说明：本项目没有任何接受路径或命令的工具，但仍然对参数做扫描，
# 用于演示"模型输出不可信"的防御深度，并防止未来新增工具时引入风险。
DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"\.\.[\\/]", "路径穿越"),
    (r"^[a-zA-Z]:[\\/]", "绝对路径"),
    (r"^/(etc|root|home|var|usr|proc)/", "系统路径"),
    (r"\b(os\.system|subprocess|popen|eval\(|exec\()", "代码执行"),
    (r"\b(rm\s+-rf|del\s+/f|format\s+c:|shutdown)", "破坏性命令"),
    (r"[`;$]\s*(sh|bash|cmd|powershell)", "shell 拼接"),
    (r"\|\s*(sh|bash|cmd)\b", "管道执行"),
    (r"\bhttps?://", "外部 URL"),
]

_SKIP_SCAN_FIELDS = {"body", "message", "content", "reason"}
_COMPILED = [(re.compile(p, re.IGNORECASE), d) for p, d in DANGEROUS_PATTERNS]


def scan_arguments(arguments: Dict[str, Any]) -> Optional[str]:
    """扫描参数中的危险模式；返回 None 表示安全，否则返回可读原因。"""
    for key, value in (arguments or {}).items():
        if str(key).lower() in _SKIP_SCAN_FIELDS:
            continue
        if not isinstance(value, str):
            continue
        for pattern, desc in _COMPILED:
            if pattern.search(value):
                return f"参数 `{key}` 命中危险模式（{desc}），已拒绝"
    return None


# --------------------------------------------------------------------------- #
# 参数规格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArgSpec:
    """单个参数的校验规格（极简 JSON-Schema 子集，避免额外依赖）。"""

    name: str
    type: str = "string"  # string | integer | boolean
    required: bool = True
    max_length: int = 300
    enum: Optional[Tuple[str, ...]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["enum"] = list(self.enum) if self.enum else None
        return data


@dataclass(frozen=True)
class ToolSpec:
    """工具规格：名称、说明、参数、风险等级、是否带副作用。"""

    name: str
    description: str
    args: Tuple[ArgSpec, ...] = ()
    risk_level: str = "low"  # low | medium | high
    side_effect: bool = False
    data_scope: str = ""

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        """校验参数；返回 (是否合法, 原因)。"""
        if not isinstance(arguments, dict):
            return False, "参数必须是对象"
        allowed = {a.name for a in self.args}
        for key in arguments:
            if key not in allowed:
                return False, f"出现未知参数 `{key}`"

        for spec in self.args:
            if spec.name not in arguments:
                if spec.required:
                    return False, f"缺少必填参数 `{spec.name}`"
                continue
            value = arguments[spec.name]
            if spec.type == "string":
                if not isinstance(value, str):
                    return False, f"参数 `{spec.name}` 必须是字符串"
                if not value.strip():
                    return False, f"参数 `{spec.name}` 不能为空"
                if len(value) > spec.max_length:
                    return False, f"参数 `{spec.name}` 超过最大长度 {spec.max_length}"
            elif spec.type == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return False, f"参数 `{spec.name}` 必须是整数"
            elif spec.type == "boolean":
                if not isinstance(value, bool):
                    return False, f"参数 `{spec.name}` 必须是布尔值"
            if spec.enum is not None and value not in spec.enum:
                preview = ", ".join(str(x) for x in spec.enum[:8])
                return False, f"参数 `{spec.name}` 取值不在白名单内（可选：{preview}）"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "side_effect": self.side_effect,
            "data_scope": self.data_scope,
            "args": [a.to_dict() for a in self.args],
        }

    def prompt_description(self) -> str:
        """生成给模型的工具说明文本。"""
        parts = [f"{self.name}（风险={self.risk_level}）：{self.description}"]
        for arg in self.args:
            req = "必填" if arg.required else "可选"
            extra = f"，可选值 {list(arg.enum)}" if arg.enum else ""
            parts.append(f"  - {arg.name} ({arg.type}, {req})：{arg.description}{extra}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 工具结果
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """统一的工具执行结果。"""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    simulated: bool = True
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def data_excerpt(self, limit: int = 400) -> str:
        """把 data 压缩为一段有界的纯文本摘录，用于回传给 Agent 作为"观察"。

        真 IPI 要求 Agent 能真正看到工具返回的内容，因此观察必须来自 data
        而非人类可读的 summary；长度受限以避免提示词膨胀。
        """
        parts: List[str] = []
        for key in ("emails", "documents", "results"):
            items = self.data.get(key)
            if isinstance(items, list):
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    text = " | ".join(
                        f"{k}={item[k]}"
                        for k in ("id", "subject", "title", "sender", "snippet", "body")
                        if k in item and isinstance(item[k], str)
                    )
                    if text:
                        parts.append(text)
        for key in ("content", "body", "text"):
            value = self.data.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        excerpt = "\n".join(parts)
        if len(excerpt) > limit:
            excerpt = excerpt[:limit] + "…（观察已截断）"
        return excerpt


ToolHandler = Callable[[Dict[str, Any]], ToolResult]
