"""AgentAdapter：可扩展的被测 Agent 统一接入接口。

设计目标
--------
AgentShield 是**可扩展的 Agent 工具调用安全测试平台**。任何被测 Agent
（规则模拟、DeepSeek API、用户自定义 Python Agent、未来的其他模型服务）
都通过实现 `AgentAdapter` 接入，并把各自的原生输出转换成内部的
`AgentResponse` / `ToolCallProposal` 标准格式。这样：

* 安全策略、模拟工具、评估器、指标计算完全与 Agent 实现解耦；
* 不同 Agent 的结果可以使用同一套指标横向比较；
* 新增一种 Agent 只需新增一个文件并在注册表中注册，无需改动 UI 与评估逻辑。

契约（所有适配器必须遵守）
--------------------------
1. 适配器只**提出**工具调用，绝不执行工具；执行由编排层在策略检查后完成；
2. 不得执行模型返回的任何代码（Python / shell / 任意 URL / 任意路径）；
3. 返回的 `tool_calls` 中每条都会被白名单与参数校验，非法项记为 `invalid`；
4. 适配器不得直接访问 `ToolRegistry` 或安全策略，只能看到工具说明文本；
5. 适配器内部异常应包装为 `AdapterError`，由编排层记录为 error 状态。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from ..schemas import RunMode, ToolCallProposal
from ..tools.base import ToolSpec


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class AdapterError(RuntimeError):
    """适配器执行失败（配置缺失、API 错误、解析失败等）。"""

    def __init__(self, message: str, *, kind: str = "adapter_error") -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------- #
# 输入数据结构
# --------------------------------------------------------------------------- #
@dataclass
class AgentRequest:
    """发给被测 Agent 的一次请求。"""

    task: str
    attack_prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    available_tools: List[ToolSpec] = field(default_factory=list)
    round_index: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    defense_hint: str = ""  # 只给模糊提示，不泄露内部策略细节

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 输出数据结构
# --------------------------------------------------------------------------- #
@dataclass
class AgentResponse:
    """被测 Agent 的标准化输出（内部标准事件格式的载体）。"""

    message: str = ""
    tool_calls: List[ToolCallProposal] = field(default_factory=list)
    final_answer: str = ""
    mode: str = RunMode.MOCK.value
    model_name: str = "unknown"
    adapter_id: str = "unknown"
    error: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    raw: Any = None
    # 模型客户端调用前保存的消息快照（已脱敏，不含连接配置或请求头）。
    request_messages: List[Dict[str, str]] = field(default_factory=list)
    prompt_defense: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        data.pop("raw", None)
        return data


# --------------------------------------------------------------------------- #
# 适配器元信息
# --------------------------------------------------------------------------- #
@dataclass
class AdapterInfo:
    """适配器元信息，供 UI 展示与可用性判断。"""

    adapter_id: str
    display_name: str
    description: str
    mode: str
    requires_api_key: bool = False
    supports_llm_attacker: bool = False
    default_model: str = ""
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #
class AgentAdapter(ABC):
    """被测 Agent 的统一接口。"""

    adapter_id: str = "base"
    display_name: str = "Base Adapter"

    def __init__(self, **options: Any) -> None:
        self.options: Dict[str, Any] = dict(options)

    # ---- 元信息 ----
    @abstractmethod
    def info(self) -> AdapterInfo:
        """返回适配器元信息（UI 用它决定是否可用）。"""

    # ---- 健康检查 ----
    def health_check(self) -> tuple[bool, str]:
        """连接/可用性自检。默认实现认为总是可用。"""
        return True, "适配器可用（无需外部依赖）"

    # ---- 主执行 ----
    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse:
        """执行一次 Agent 推理，返回标准化响应。"""

    # ---- 可选：LLM 攻击者改写 ----

    # ---- 工具方法 ----
    def _tools_payload(self, request: AgentRequest) -> str:
        return "\n\n".join(t.prompt_description() for t in request.available_tools)

    def _measure(self) -> Callable[[], int]:
        start = time.perf_counter()
        return lambda: int((time.perf_counter() - start) * 1000)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
class AdapterRegistry:
    """适配器注册表：只有显式注册的适配器才能被 UI 选择。"""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., AgentAdapter]] = {}

    def register(
        self, adapter_id: str, factory: Callable[..., AgentAdapter]
    ) -> None:
        if adapter_id in self._factories:
            raise ValueError(f"适配器已注册：{adapter_id}")
        self._factories[adapter_id] = factory

    def create(self, adapter_id: str, **options: Any) -> AgentAdapter:
        factory = self._factories.get(adapter_id)
        if factory is None:
            raise AdapterError(
                f"未注册的适配器 `{adapter_id}`，可选：{sorted(self._factories)}",
                kind="unknown_adapter",
            )
        return factory(**options)

    def ids(self) -> List[str]:
        return sorted(self._factories)

    def has(self, adapter_id: str) -> bool:
        return adapter_id in self._factories

    def infos(self) -> List[AdapterInfo]:
        infos: List[AdapterInfo] = []
        for aid in self.ids():
            try:
                infos.append(self.create(aid).info())
            except Exception:
                continue
        return infos


REGISTRY = AdapterRegistry()


def register_adapter(adapter_id: str):
    """装饰器：注册一个适配器类。"""

    def _wrap(cls: type) -> type:
        REGISTRY.register(adapter_id, lambda **kw: cls(**kw))
        cls.adapter_id = adapter_id
        return cls

    return _wrap


def get_adapter(adapter_id: str, **options: Any) -> AgentAdapter:
    """按 id 创建适配器实例。"""
    return REGISTRY.create(adapter_id, **options)


def list_adapters() -> List[AdapterInfo]:
    return REGISTRY.infos()
