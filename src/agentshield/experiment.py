"""实验编排层：案例 → Agent → 策略 → 模拟工具 → 评估 → 记录。

这一层不依赖 Streamlit，可以在命令行、测试或未来的云端任务中直接调用。
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .adapters.base import AgentAdapter, AgentRequest
from .adapters.mock_adapter import MockAgentAdapter  # noqa: F401 - 确保注册
from .config import Settings, get_settings
from .judge import evaluate_case
from .metrics import compute_metrics
from .policy import SecurityPolicy, classify_authorization, policy_config_for
from .redaction import redact_arguments, redact_text, safe_error
from .schemas import (
    CASE_SET_VERSION,
    POLICY_VERSION,
    Case,
    CaseResult,
    ExperimentConfig,
    ExperimentRun,
    PolicyDecision,
    RoundRecord,
    ToolCallEvent,
    new_run_id,
)
from .tools.datasets import MOCK_DOCUMENTS, MOCK_EMAILS, SENSITIVE_FILE_IDS
from .tools.registry import ToolRegistry, build_default_registry

MAX_ROUNDS_LIMIT = 5
# 回传给 Agent 的观察文本长度上限：既要让注入内容可见（真 IPI 成立），
# 又要防止超长内容把提示词撑爆。
OBSERVATION_EXCERPT_LIMIT = 600


def build_context() -> Dict[str, Any]:
    """生成模拟数据摘要（只暴露数量与 ID，不泄露全部内容）。"""
    return {
        "emails_count": len(MOCK_EMAILS),
        "documents_count": len(MOCK_DOCUMENTS),
        "sensitive_file_ids": list(SENSITIVE_FILE_IDS),
        "note": "以上均为内置虚构模拟数据，工具不会产生任何真实外部副作用。",
    }


def run_label(config: ExperimentConfig) -> str:
    """为一次运行生成可阅读的编号标签。

    形如 `mock-d3-defon-psand`：适配器 / DPI 变体（none→base）/
    执行层防御开关 / Prompt 级防御。便于在 UI 与导出文件名中一眼分辨，
    无需依赖随机串；消融实验的四组也因此天然区分。
    """
    variant = getattr(config, "attack_variant", "none") or "none"
    prompt = getattr(config, "prompt_defense", "none") or "none"
    return "-".join(
        [
            (config.adapter_id or "agent").replace("_", "-"),
            variant if variant != "none" else "base",
            "defon" if config.defense_enabled else "defoff",
            "p" + prompt[:4],
        ]
    )


@dataclass
class ExperimentRunner:
    """实验执行器。"""

    adapter: AgentAdapter
    registry: ToolRegistry = field(default_factory=build_default_registry)
    settings: Optional[Settings] = None
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()


    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def run(self, cases: List[Case], config: ExperimentConfig) -> ExperimentRun:
        started = datetime.now()
        run_id = new_run_id(run_label(config))
        max_rounds = max(1, min(int(config.max_rounds), MAX_ROUNDS_LIMIT))
        policy_config = policy_config_for(config.defense_enabled)
        initial_calls = self._collect_api_calls()
        initial_failures = self._collect_client_counter("total_failures")

        results: List[CaseResult] = []
        api_calls = 0
        api_failures = 0
        mode_switched = False

        total = len(cases)
        for index, case in enumerate(cases, start=1):
            if self.progress_callback:
                self.progress_callback(index, total, f"正在运行 {case.case_id}")
            result = self.run_case(
                case=case,
                run_id=run_id,
                config=config,
                policy_config=policy_config,
                max_rounds=max_rounds,
            )
            results.append(result)
            if result.mode != config.mode:
                mode_switched = True

        # 统计真实 API 调用次数
        api_calls = self._collect_api_calls() - initial_calls
        api_failures = self._collect_client_counter("total_failures") - initial_failures

        actual_models = list(dict.fromkeys(r.model_name for r in results if r.model_name))
        actual_model = " | ".join(actual_models) or getattr(self.adapter, "model_name", "") or config.model_name
        run = ExperimentRun(
            run_id=run_id,
            created_at=started.isoformat(timespec="seconds"),
            mode=config.mode,
            model_name=actual_model,
            adapter_id=config.adapter_id,
            defense_enabled=config.defense_enabled,
            defense_config=policy_config.to_dict(),
            case_set_version=CASE_SET_VERSION,
            policy_version=POLICY_VERSION,
            max_rounds=max_rounds,
            temperature=config.temperature,
            total_cases=len(results),
            seed=config.seed,
            attack_variant=getattr(config, "attack_variant", "none") or "none",
            prompt_defense=getattr(config, "prompt_defense", "none") or "none",
            api_calls=api_calls,
            api_failures=api_failures,
            mode_switched=mode_switched,
            results=results,
        )
        run.summary_metrics = compute_metrics(results)
        return run

    # ------------------------------------------------------------------ #
    def run_case(
        self,
        case: Case,
        run_id: str,
        config: ExperimentConfig,
        policy_config: Any,
        max_rounds: int,
    ) -> CaseResult:
        if case.is_malicious or case.attack_prompt or case.is_ipi:
            raise ValueError("公开版仅接收正常任务与合法操作样例。")
        started = time.perf_counter()
        policy = SecurityPolicy(registry=self.registry, config=policy_config)
        policy.reset_case(case.case_id)

        rounds: List[RoundRecord] = []
        error_summary = ""
        observed_mode = config.mode
        observed_model = config.model_name
        calls_this_case = 0

        for round_index in range(max_rounds):
            request = AgentRequest(
                task=case.task,
                context=build_context(),
                available_tools=self.registry.specs(),
                round_index=round_index,
                # 消息快照仅用于审计，不递归嵌入下一轮历史。
                history=[{
                    k: v for k, v in r.to_dict().items()
                    if k not in {"request_messages", "attacker_request_messages", "attacker_feedback"}
                } for r in rounds],
                params={
                    "case": case,
                    "temperature": config.temperature,
                    "defense_enabled": config.defense_enabled,
                    "attack_variant": getattr(config, "attack_variant", "none"),
                    "prompt_defense": getattr(config, "prompt_defense", "none"),
                },
            )

            response = self.adapter.run(request)
            observed_mode = response.mode or observed_mode
            observed_model = response.model_name or observed_model

            record = RoundRecord(
                round_index=round_index,
                agent_message=response.message,
                agent_final_answer=response.final_answer,
                proposed_calls=[c.to_dict() for c in response.tool_calls],
                parse_error=response.error,
                duration_ms=response.latency_ms,
                request_messages=deepcopy(response.request_messages),
                prompt_defense=response.prompt_defense,
            )
            if response.error:
                error_summary = safe_error(response.error)
                record.events = []
                rounds.append(record)
                # 出错后不再继续消耗轮次
                break

            events: List[ToolCallEvent] = []
            calls_this_round = 0
            for proposal in response.tool_calls[:3]:
                unauthorized, auth_reason = classify_authorization(
                    case, proposal.tool, self.registry
                )
                verdict = policy.evaluate(
                    case=case,
                    tool_name=proposal.tool,
                    arguments=proposal.arguments,
                    calls_this_round=calls_this_round,
                    calls_this_case=calls_this_case,
                )
                calls_this_round += 1
                calls_this_case += 1

                event = ToolCallEvent.new(
                    run_id=run_id,
                    case_id=case.case_id,
                    round_index=round_index,
                    proposal=proposal,
                    arguments=redact_arguments(proposal.arguments, max_length=1000),
                    policy_decision=verdict.decision,
                    policy_reason=verdict.reason,
                    is_unauthorized=unauthorized,
                    defense_enabled=config.defense_enabled,
                    adapter_id=config.adapter_id,
                    model_name=observed_model,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
                spec = self.registry.get_spec(proposal.tool)
                event.risk_level = spec.risk_level if spec else ""

                if verdict.decision == PolicyDecision.ALLOWED.value:
                    result = policy.execute(case, proposal.tool, proposal.arguments)
                    if result.success:
                        event.executed = True
                        data = result.data
                        if proposal.tool == "send_email":
                            event.executed_arguments = {
                                key: redact_text(str(data.get(key, "")), max_length=None)
                                for key in ("to", "subject", "body")
                            }
                        event.result_ids = [str(data["id"])] if data.get("id") else []
                        for key in ("emails", "documents", "results"):
                            records = data.get(key, [])
                            if isinstance(records, list):
                                event.result_ids.extend(
                                    str(item["id"]) for item in records
                                    if isinstance(item, dict) and item.get("id")
                                )
                        event.result_summary = result.summary or "模拟工具执行成功"
                        # 观察通道：把工具返回的真实内容回传给 Agent（真 IPI 的载体）
                        event.observation = result.data_excerpt(
                            limit=OBSERVATION_EXCERPT_LIMIT
                        )
                    else:
                        event.policy_decision = PolicyDecision.ERROR.value
                        event.policy_reason = result.error or "模拟工具执行失败"
                        event.result_summary = result.summary or "执行失败"
                else:
                    event.result_summary = (
                        f"未执行（{verdict.decision}）：{verdict.reason}"
                    )
                events.append(event)

            record.events = events
            rounds.append(record)

        duration_ms = int((time.perf_counter() - started) * 1000)
        return evaluate_case(
            case=case,
            rounds=rounds,
            defense_enabled=config.defense_enabled,
            run_id=run_id,
            mode=observed_mode,
            model_name=observed_model,
            adapter_id=config.adapter_id,
            duration_ms=duration_ms,
            error_summary=error_summary,
        )

    # ------------------------------------------------------------------ #
    def _collect_api_calls(self) -> int:
        """统计被测模型的 API 调用次数。"""
        return self._collect_client_counter("total_calls")

    def _collect_client_counter(self, counter: str) -> int:
        value = getattr(getattr(self.adapter, "client", None), counter, 0)
        return value if isinstance(value, int) else 0


def build_adapter(config: ExperimentConfig, settings: Optional[Settings] = None) -> AgentAdapter:
    """根据实验配置创建适配器实例。"""
    from .adapters.base import get_adapter

    settings = settings or get_settings()
    if config.adapter_id == "mock":
        return get_adapter("mock", seed=config.seed, compliance=0.6)
    if config.adapter_id == "deepseek":
        return get_adapter(
            "deepseek",
            settings=settings,
            model_name=config.model_name or settings.deepseek_model,
            prompt_defense=getattr(config, "prompt_defense", "none") or "none",
        )
    if config.adapter_id == "local":
        # 本地 llama.cpp（ROCm）服务：不需要 API Key，失败不降级为 Mock
        return get_adapter(
            "local",
            settings=settings,
            model_name=config.model_name or settings.local_model,
            prompt_defense=getattr(config, "prompt_defense", "none") or "none",
        )
    return get_adapter(config.adapter_id)


def estimate_model_calls(cases: List[Case], config: ExperimentConfig, groups: int = 1) -> Dict[str, int]:
    """按角色和后端估算完整预算，不含重试；失败提前停止会减少实际调用。"""
    rounds = max(1, min(int(config.max_rounds), MAX_ROUNDS_LIMIT))
    counts = {"deepseek": 0, "local": 0}
    if config.adapter_id in counts:
        counts[config.adapter_id] += len(cases) * rounds * groups
    return counts


# --------------------------------------------------------------------------- #
# 双层防御消融（2×2）
# --------------------------------------------------------------------------- #
# 两个因子：
#   exec_layer：执行层策略（policy.py）开/关
#   prompt_layer：prompt 级防御（prompt_defenses.py）开/关
# 四组对照用于回答"仅靠提示词加固够不够""执行层拦截带来多少额外收益"。
ABLATION_GROUPS = {
    "no_defense": (False, "none"),
    "prompt_only": (False, "delimiters"),
    "execution_only": (True, "none"),
    "both": (True, "delimiters"),
}


def run_ablation(
    cases: List[Case],
    config: ExperimentConfig,
    settings: Optional[Settings] = None,
    prompt_defense: str = "delimiters",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, ExperimentRun]:
    """双层防御消融：同一案例集跑四组（执行层 × prompt 层）。

    注意：prompt 级防御对 Mock Agent 无效（Mock 不解析文本），
    因此 Mock 模式下 prompt_only 与 no_defense 的指标必然相同 —— 这是预期结果，
    报告中不得据此得出"prompt 级防御无效"的结论。
    """
    output: Dict[str, ExperimentRun] = {}
    for group_index, (name, (exec_layer, prompt_layer)) in enumerate(ABLATION_GROUPS.items()):
        group_config = ExperimentConfig(
            **{
                **config.to_dict(),
                "defense_enabled": exec_layer,
                "prompt_defense": prompt_layer if prompt_layer != "none" else "none",
            }
        )
        if name == "prompt_only":
            group_config.prompt_defense = prompt_defense
        elif name == "both":
            group_config.prompt_defense = prompt_defense
        adapter = build_adapter(group_config, settings)
        def report(index: int, total: int, message: str, group_index=group_index, name=name) -> None:
            if progress_callback:
                progress_callback(group_index * total + index, total * 4, f"[{group_index + 1}/4 {name}] {message}")
        runner = ExperimentRunner(
            adapter=adapter, settings=settings, progress_callback=report
        )
        output[name] = runner.run(cases, group_config)
    return output


def run_comparison(
    cases: List[Case],
    config: ExperimentConfig,
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, ExperimentRun]:
    """一键对照实验：同一案例集分别跑防御关闭与防御开启两组。"""
    output: Dict[str, ExperimentRun] = {}
    for group_index, flag in enumerate((False, True)):
        group_config = ExperimentConfig(**{**config.to_dict(), "defense_enabled": flag})
        adapter = build_adapter(group_config, settings)
        def report(index: int, total: int, message: str, group_index=group_index, flag=flag) -> None:
            if progress_callback:
                progress_callback(group_index * total + index, total * 2, f"[{group_index + 1}/2 执行层{'开' if flag else '关'}] {message}")
        runner = ExperimentRunner(
            adapter=adapter, settings=settings, progress_callback=report
        )
        output["defense_on" if flag else "defense_off"] = runner.run(cases, group_config)
    return output
