"""自定义 Python Agent 适配器（扩展示例）。

用途：让用户把自己写的任意 Python Agent 接入 AgentShield，
只需实现一个函数：

    def my_agent(task: str, attack_prompt: str, tool_descriptions: str) -> dict:
        return {
            "message": "...",
            "tool_calls": [{"tool": "search_emails", "arguments": {"query": "进度"}}],
            "final_answer": "...",
        }

然后：

    from agentshield.adapters.custom_adapter import CallableAgentAdapter
    adapter = CallableAgentAdapter(fn=my_agent, adapter_id="my_agent")

安全约束（强制）：
* 该函数只返回工具调用**提案**，不得自行执行工具；
* 不得执行模型返回的代码、shell 命令或访问任意路径；
* 返回的 tool / arguments 仍会被白名单与参数校验，非法项记为 invalid。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..schemas import RunMode, ToolCallProposal
from .base import AdapterInfo, AgentAdapter, AgentRequest, AgentResponse

AgentFn = Callable[[str, str, str], Dict[str, Any]]


class CallableAgentAdapter(AgentAdapter):
    """把任意 Python 函数包装为 AgentAdapter。"""

    def __init__(
        self,
        fn: AgentFn,
        adapter_id: str = "custom_python",
        display_name: str = "自定义 Python Agent",
        description: str = "",
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if not callable(fn):
            raise TypeError("fn 必须是可调用对象")
        self.fn = fn
        self.adapter_id = adapter_id
        self.display_name = display_name
        self._description = description or f"用户自定义 Python Agent：`{adapter_id}`"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id=self.adapter_id,
            display_name=self.display_name,
            description=self._description,
            mode=RunMode.MOCK.value,
            requires_api_key=False,
            supports_llm_attacker=False,
            default_model="user-python-function",
            options=["fn"],
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "自定义函数适配器：直接调用本地 Python 函数（仍在模拟工具沙箱内执行）"

    def run(self, request: AgentRequest) -> AgentResponse:
        done = self._measure()
        try:
            raw = self.fn(request.task, request.attack_prompt, self._tools_payload(request))
        except Exception as exc:  # noqa: BLE001
            return AgentResponse(
                error=f"自定义 Agent 执行异常：{type(exc).__name__}: {exc}",
                adapter_id=self.adapter_id,
                model_name="user-python-function",
                latency_ms=done(),
            )
        if not isinstance(raw, dict):
            return AgentResponse(
                error="自定义 Agent 必须返回 dict",
                adapter_id=self.adapter_id,
                model_name="user-python-function",
                latency_ms=done(),
            )

        proposals: List[ToolCallProposal] = []
        for item in (raw.get("tool_calls") or [])[:3]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            if not tool:
                continue
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            proposals.append(
                ToolCallProposal(tool=tool, arguments=args, reason=str(item.get("reason", ""))[:200])
            )

        return AgentResponse(
            message=str(raw.get("message", ""))[:500],
            tool_calls=proposals,
            final_answer=str(raw.get("final_answer", "")),
            mode=RunMode.MOCK.value,
            model_name="user-python-function",
            adapter_id=self.adapter_id,
            latency_ms=done(),
            raw=raw,
        )
