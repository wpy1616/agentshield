"""MockAgentAdapter：无需 API Key、完全离线、可复现的模拟被测 Agent。

用于正常办公任务的离线演示和自动化检查。UI 与导出将模拟结果标记为 `mock`。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..schemas import Case, RunMode
from .base import (
    AdapterInfo,
    AgentAdapter,
    AgentRequest,
    AgentResponse,
    ToolCallProposal,
    register_adapter,
)

_TOPIC_KEYWORDS = [
    "预算", "进度", "会议", "客户", "风险", "培训", "测试",
    "分享", "材料", "用例", "反馈", "里程碑", "规范", "合规",
]

_STOPWORDS = {"请", "帮我", "总结", "一下", "关于", "的", "和", "与", "并", "查找", "搜索"}


def _infer_query(text: str, default: str = "项目") -> str:
    for kw in _TOPIC_KEYWORDS:
        if kw in text:
            return kw
    tokens = [t for t in re.findall(r"[\w一-鿿]+", text or "") if t not in _STOPWORDS]
    return tokens[0] if tokens else default


@register_adapter("mock")
class MockAgentAdapter(AgentAdapter):
    """确定性模拟 Agent（离线可用）。"""

    adapter_id = "mock"
    display_name = "Mock 模拟 Agent"

    def __init__(self, seed: int = 42, compliance: float = 0.6, **options: Any) -> None:
        super().__init__(**options)
        self.seed = int(seed)
        self.compliance = max(0.0, min(1.0, float(compliance)))

    # ------------------------------------------------------------------ #
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="mock",
            display_name="Mock 模拟 Agent（离线）",
            description=(
                "完全离线的确定性模拟 Agent，不需要 API Key。"
                "按正常任务生成模拟提案，用于流程验证与课堂演示。"
            ),
            mode=RunMode.MOCK.value,
            requires_api_key=False,
            supports_llm_attacker=False,
            default_model="mock-simulated-v1",
            options=["seed", "compliance"],
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "Mock 适配器无需外部依赖，可直接运行（结果非真实模型输出）"

    # ------------------------------------------------------------------ #
    def run(self, request: AgentRequest) -> AgentResponse:
        done = self._measure()
        case: Optional[Case] = request.params.get("case")
        tool_names = {t.name for t in request.available_tools}

        if case is None:
            # 无案例上下文时的保守行为：只做一次安全检索
            calls = [
                ToolCallProposal(
                    "search_emails",
                    {"query": _infer_query(request.task)},
                    reason="按任务检索相关邮件",
                )
            ]
            return AgentResponse(
                message="模拟 Agent：无案例上下文，执行保守检索。",
                tool_calls=[c for c in calls if c.tool in tool_names],
                final_answer="（模拟）已检索相关资料。",
                mode=RunMode.MOCK.value,
                model_name="mock-simulated-v1",
                adapter_id="mock",
                latency_ms=done(),
            )

        if case.benign_steps:
            calls = self._capability_calls(case, request, tool_names)
            msg = "模拟 Agent：按案例执行已授权的操作。"
        else:
            calls = self._benign_calls(case, tool_names)
            msg = "模拟 Agent：按正常任务执行检索。"
        answer = self._benign_answer(case, request)
        return AgentResponse(
            message=msg,
            tool_calls=calls[:3],
            final_answer=answer,
            mode=RunMode.MOCK.value,
            model_name="mock-simulated-v1",
            adapter_id="mock",
            usage={"mock_calls": 0},
            latency_ms=done(),
            raw={"seed": self.seed, "compliance": self.compliance},
        )

    def _benign_calls(self, case: Case, tool_names: set) -> List[ToolCallProposal]:
        calls: List[ToolCallProposal] = []
        for tool in case.expected_allowed_tools or []:
            if tool in tool_names and tool in ("search_emails", "search_documents"):
                calls.append(
                    ToolCallProposal(
                        tool, {"query": _infer_query(case.task)}, reason=f"按任务检索：{case.task}"
                    )
                )
        return calls[:2]

    def _capability_calls(
        self, case: Case, request: AgentRequest, tool_names: set[str]
    ) -> List[ToolCallProposal]:
        """按顺序模拟已声明的合法操作，每轮最多执行一步。"""
        completed = [
            event
            for record in request.history for event in record.get("events", [])
            if event.get("executed") and not event.get("is_unauthorized")
        ]
        if len(completed) >= len(case.benign_steps):
            return []

        step = case.benign_steps[len(completed)]
        tool = step["tool"]
        if tool not in tool_names or tool not in case.expected_allowed_tools:
            return []
        arguments = dict(step.get("arguments", {}))
        if tool == "send_email":
            arguments.setdefault("subject", "授权资料（模拟）")
            if step.get("body_from_read"):
                read = next(
                    (event for event in reversed(completed) if event["tool_name"] == "read_sensitive_file"),
                    None,
                )
                if read is None or not read.get("observation"):
                    return []
                arguments["body"] = read["observation"]
            else:
                arguments["body"] = "。".join(group[0] for group in step.get("body_keywords", []) if group)
        return [ToolCallProposal(tool, arguments, reason="按案例演示已授权的操作")]

    def _benign_answer(self, case: Case, request: AgentRequest) -> str:
        # 只复述此前确实取得的合法工具结果，不能提前宣称任务已经完成。
        observations = [
            str(event.get("observation", ""))
            for record in request.history or [] for event in record.get("events", [])
            if event.get("executed") and not event.get("is_unauthorized")
            and event.get("observation")
        ]
        return "（模拟资料摘录）" + "\n".join(dict.fromkeys(observations)) if observations else ""

