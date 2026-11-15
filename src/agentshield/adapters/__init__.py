"""Agent 适配器包。

对外暴露统一的 `AgentAdapter` 接口、注册表与内置适配器。
新增适配器只需在本包内新增模块并 import，即可完成注册。
"""

from .base import (
    REGISTRY,
    AdapterError,
    AdapterInfo,
    AdapterRegistry,
    AgentAdapter,
    AgentRequest,
    AgentResponse,
    ToolCallProposal,
    get_adapter,
    list_adapters,
    register_adapter,
)
from .custom_adapter import CallableAgentAdapter
from .deepseek_adapter import DeepSeekAgentAdapter
from .local_llm_adapter import LocalLLMAdapter
from .mock_adapter import MockAgentAdapter  # noqa: F401 - 触发注册

__all__ = [
    "REGISTRY",
    "AdapterError",
    "AdapterInfo",
    "AdapterRegistry",
    "AgentAdapter",
    "AgentRequest",
    "AgentResponse",
    "ToolCallProposal",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "MockAgentAdapter",
    "DeepSeekAgentAdapter",
    "LocalLLMAdapter",
    "CallableAgentAdapter",
]
