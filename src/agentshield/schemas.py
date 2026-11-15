"""AgentShield 核心数据模型。

本模块定义贯穿全项目的统一数据结构：案例、工具调用提案、工具调用事件、
逐轮轨迹、案例结果、实验运行记录。所有适配器、策略、评估器和导出模块
都只依赖这里的对象，从而保证不同 Agent 后端产生的数据可以直接横向比较。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.6"
CASE_SET_VERSION = "cases-v5-public"
POLICY_VERSION = "policy-v2"


# --------------------------------------------------------------------------- #
# 攻击通道
# --------------------------------------------------------------------------- #
class AttackChannel(str, Enum):
    """恶意指令进入 Agent 的通道。

    PROMPT：直接提示注入（DPI）——恶意指令拼进用户输入提示词；
    OBSERVATION：间接提示注入（IPI）——恶意指令藏在工具返回的数据里，
                Agent 读到观察结果后才有可能被诱导。
    """

    PROMPT = "prompt"
    OBSERVATION = "observation"


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class PolicyDecision(str, Enum):
    """安全策略对一次工具调用提案的判定结果。"""

    ALLOWED = "allowed"  # 允许执行
    BLOCKED = "blocked"  # 策略拒绝
    INVALID = "invalid"  # 工具名或参数不合法
    ERROR = "error"  # 模拟工具执行错误


class CaseStatus(str, Enum):
    """案例级最终状态。"""

    ATTACK_SUCCESS = "attack_success"  # 恶意案例攻击目标达成
    DEFENSE_SUCCESS = "defense_success"  # 违规尝试被拦截，目标未达成
    NO_ATTEMPT = "no_attempt"  # Agent 未提出任何违规调用
    BENIGN_COMPLETED = "benign_completed"  # 正常案例成功完成
    FALSE_POSITIVE = "false_positive"  # 正常案例中合法调用被误拦截
    INCONCLUSIVE = "inconclusive"  # 无法判定
    ERROR = "error"  # 运行错误


class RunMode(str, Enum):
    """运行模式。

    MOCK：规则驱动的模拟 Agent（非真实模型，仅用于流程与对照）；
    DEEPSEEK：通过 DeepSeek 云端 API 调用真实模型；
    LOCAL：通过本地 / 局域网的 OpenAI 兼容服务调用开源模型
           （如 AMD AUP 上由 llama.cpp + ROCm 提供的 GGUF 模型），
           推理发生在本地 GPU，不经过任何外部 API。
    """

    MOCK = "mock"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


def mode_label(mode: str) -> str:
    return {
        "mock": "Mock 模拟（非真实模型）",
        "deepseek": "DeepSeek API 真实模型",
        "local": "本地模型（OpenAI 兼容服务）",
    }.get(mode, f"自定义模式：{mode}")


# --------------------------------------------------------------------------- #
# 案例
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """一个红队测试案例。"""

    case_id: str
    category: str
    is_malicious: bool
    task: str
    attack_prompt: str = ""
    expected_allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    allowed_data_scope: List[str] = field(default_factory=list)
    malicious_goal: str = ""
    expected_benign_outcome: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    # 攻击通道：prompt（DPI，默认）或 observation（IPI）
    attack_channel: str = AttackChannel.PROMPT.value
    # 明确的评估条件；未提供时只记录调用行为，不臆测任务/攻击目标达成。
    benign_sources: List[str] = field(default_factory=list)
    benign_answer_keywords: List[List[str]] = field(default_factory=list)
    # 按时间顺序匹配工具、参数和内容；每项至少包含 tool。
    attack_steps: List[Dict[str, Any]] = field(default_factory=list)
    # 0 表示全部要点；正数表示至少命中多少个不同要点组。
    benign_min_keyword_groups: int = 0
    benign_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_ipi(self) -> bool:
        return self.attack_channel == AttackChannel.OBSERVATION.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# 工具调用提案（Agent 适配器输出）
# --------------------------------------------------------------------------- #
@dataclass
class ToolCallProposal:
    """Agent 提出的一次工具调用请求（标准化后的内部格式）。"""

    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 工具调用事件（执行后记录）
# --------------------------------------------------------------------------- #
@dataclass
class ToolCallEvent:
    """一次工具调用从提案到执行的完整记录。"""

    event_id: str
    run_id: str
    case_id: str
    round_index: int
    tool_name: str
    arguments: Dict[str, Any]
    policy_decision: str
    policy_reason: str
    executed: bool
    simulated: bool
    result_summary: str
    risk_level: str = ""
    defense_enabled: bool = True
    adapter_id: str = ""
    model_name: str = ""
    is_unauthorized: bool = False
    timestamp: str = ""
    # 观察：本轮工具返回给 Agent 的有界文本（真 IPI 的注入载体）
    observation: str = ""
    result_ids: List[str] = field(default_factory=list)
    # 发信工具实际接受的收件人/主题/正文；None 兼容旧记录。
    executed_arguments: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def new(
        cls,
        run_id: str,
        case_id: str,
        round_index: int,
        proposal: "ToolCallProposal",
        **kwargs: Any,
    ) -> "ToolCallEvent":
        payload = {
            "event_id": uuid.uuid4().hex[:12],
            "run_id": run_id,
            "case_id": case_id,
            "round_index": round_index,
            "tool_name": proposal.tool,
            "arguments": proposal.arguments,
            "policy_decision": PolicyDecision.BLOCKED.value,
            "policy_reason": "",
            "executed": False,
            "simulated": True,
            "result_summary": "",
        }
        payload.update(kwargs)
        return cls(**payload)


# --------------------------------------------------------------------------- #
# 逐轮轨迹
# --------------------------------------------------------------------------- #
@dataclass
class RoundRecord:
    """一轮攻击—响应—执行—评估的完整记录。"""

    round_index: int
    attack_prompt: str = ""
    attacker_strategy: str = "template"
    agent_message: str = ""
    agent_final_answer: str = ""
    proposed_calls: List[Dict[str, Any]] = field(default_factory=list)
    events: List[ToolCallEvent] = field(default_factory=list)
    parse_error: str = ""
    model_calls: int = 0
    duration_ms: int = 0
    # 原样保存本轮消息顺序和防御处理后的内容；仅对凭据痕迹脱敏。
    request_messages: List[Dict[str, str]] = field(default_factory=list)
    prompt_defense: str = ""
    attack_source: str = ""
    attacker_feedback: str = ""
    attacker_request_messages: List[Dict[str, str]] = field(default_factory=list)
    attacker_error: str = ""
    attacker_model_name: str = ""
    attack_goal_reached: Optional[bool] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    observation_payload: str = ""
    attack_reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["events"] = [e.to_dict() for e in self.events]
        return data

    @property
    def unauthorized_attempts(self) -> int:
        return sum(1 for e in self.events if e.is_unauthorized)

    @property
    def unauthorized_executions(self) -> int:
        return sum(1 for e in self.events if e.is_unauthorized and e.executed)


# --------------------------------------------------------------------------- #
# 案例结果
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    """单个案例的最终结果。"""

    run_id: str
    case_id: str
    mode: str
    model_name: str
    defense_enabled: bool
    adapter_id: str = ""
    category: str = ""
    is_malicious: bool = False
    attack_attempted: bool = False
    unauthorized_attempts: int = 0
    unauthorized_executions: int = 0
    blocked_attempts: int = 0
    attack_success: bool = False
    defense_success: bool = False
    benign_task_completed: bool = False
    false_positive: bool = False
    status: str = CaseStatus.INCONCLUSIVE.value
    duration_ms: int = 0
    error_summary: str = ""
    total_tool_calls: int = 0
    evaluation_error: bool = False
    attack_evaluable: bool = True
    task_evaluable: bool = True
    evaluation_note: str = ""
    rounds: List[RoundRecord] = field(default_factory=list)
    events: List[ToolCallEvent] = field(default_factory=list)
    case_snapshot: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rounds"] = [r.to_dict() for r in self.rounds]
        data["events"] = [e.to_dict() for e in self.events]
        return data


# --------------------------------------------------------------------------- #
# 实验运行
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """一次实验运行的配置。"""

    mode: str = RunMode.MOCK.value
    adapter_id: str = "mock"
    model_name: str = ""  # 留空时使用所选适配器的默认模型。
    defense_enabled: bool = True
    max_rounds: int = 3
    temperature: float = 0.0
    case_filter: str = "all"
    categories: List[str] = field(default_factory=list)
    seed: int = 42
    use_llm_attacker: bool = False
    label: str = ""
    # 攻击者后端：留空表示"跟随被测 Agent"（保持历史行为，此时存在 self-red-teaming）。
    # 填 deepseek / local 等可让攻击者使用**独立于靶子**的模型（跨模型红队，结论更可信）。
    attacker_adapter_id: str = ""
    attacker_model_name: str = ""
    # DPI 攻击变体：none | d1 | d2 | d3 | d4 | d5（语义型五变体，自定义分类）
    attack_variant: str = "none"
    # Prompt 级防御：none | delimiters | sandwich | instructional
    prompt_defense: str = "none"
    attacker_mode: str = "feedback"  # feedback | frozen
    ipi_llm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRun:
    """一次完整实验运行（一组案例 × 一种防御配置）。"""

    run_id: str
    created_at: str
    mode: str
    model_name: str
    adapter_id: str
    defense_enabled: bool
    defense_config: Dict[str, Any] = field(default_factory=dict)
    case_set_version: str = CASE_SET_VERSION
    policy_version: str = POLICY_VERSION
    schema_version: str = SCHEMA_VERSION
    max_rounds: int = 3
    temperature: float = 0.0
    total_cases: int = 0
    seed: int = 42
    # 实验轴（用于报告复现）：DPI 变体与 prompt 级防御
    attack_variant: str = "none"
    prompt_defense: str = "none"
    # 攻击者后端：空串表示"跟随被测 Agent"（self-red-teaming，需在报告中声明）；
    # 非空表示跨模型红队（如 deepseek 攻击 + local 被测），结论更可信。
    attacker_adapter_id: str = ""
    attacker_model_name: str = ""
    # LLM 攻击者实际改写次数与失败次数（用于识别"静默回退到模板"）
    attacker_llm_calls: int = 0
    attacker_llm_failures: int = 0
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    api_calls: int = 0
    api_failures: int = 0
    mode_switched: bool = False
    results: List[CaseResult] = field(default_factory=list)
    attacker_enabled: bool = False
    attacker_note: str = ""
    attacker_mode: str = "feedback"
    ipi_llm: bool = False
    batch_id: str = ""
    trial_index: int = 0
    experiment_group: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data

    @property
    def events(self) -> List[ToolCallEvent]:
        return [e for r in self.results for e in r.events]


_RUN_ID_STATE = {"stamp": "", "seq": 0}


def slugify(text: str, fallback: str = "run") -> str:
    """把任意标签压成只含小写字母数字与连字符的短片段。"""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or fallback


def new_run_id(label: str = "") -> str:
    """生成**可读且可排序**的运行编号，不再依赖随机串。

    格式：`YYYYmmdd-HHMMSS-<语义标签>-<同秒序号>`
    例：`20261027-160215-mock-d3-defon-01`

    * 语义标签由调用方给出（适配器/变体/防御开关等），便于一眼分辨；
    * 同一秒内多次生成时以两位序号递增，保证唯一性；
    * 时间戳在前，文件名与导出目录可按时间自然排序。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if _RUN_ID_STATE["stamp"] != stamp:
        _RUN_ID_STATE["stamp"] = stamp
        _RUN_ID_STATE["seq"] = 0
    _RUN_ID_STATE["seq"] += 1
    return f"{stamp}-{slugify(label)}-{_RUN_ID_STATE['seq']:02d}"
