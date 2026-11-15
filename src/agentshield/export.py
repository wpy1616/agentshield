"""结果导出：JSON / CSV / Markdown。

导出前所有工具调用参数已经过脱敏（见 redaction.py），
导出文件中不含 API Key、Authorization Header 或真实个人信息。
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .metrics import comparison_table
from .prompt_defenses import defense_label
from .redaction import redact_arguments
from .schemas import CaseResult, ExperimentRun, ToolCallEvent, mode_label


# --------------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------------- #
def _attacker_line(run) -> str:
    """攻击者后端说明（用于报告追溯是否跨模型红队）。"""
    attacker_id = getattr(run, "attacker_adapter_id", "") or ""
    calls = getattr(run, "attacker_llm_calls", 0) or 0
    failures = getattr(run, "attacker_llm_failures", 0) or 0
    if not attacker_id or calls == 0:
        return "- 攻击者：确定性模板（无模型参与，攻击提示可逐字复现）"
    model = getattr(run, "attacker_model_name", "") or ""
    same = attacker_id == run.adapter_id and bool(model) and model == run.model_name
    tag = "self-red-teaming" if same else "独立模型后端（模型标识以记录为准）"
    return (
        f"- 攻击者：LLM 改写（后端 `{attacker_id}`"
        + (f" / 模型 `{model}`" if model else "")
        + f"，调用 {calls} 次、失败 {failures} 次）— {tag}"
    )


def run_to_json(run: ExperimentRun) -> str:
    """完整实验运行 → JSON 字符串。"""
    return json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str)


def cases_to_rows(run: ExperimentRun) -> List[Dict[str, Any]]:
    rows = []
    for r in run.results:
        rows.append(
            {
                "run_id": run.run_id,
                "case_id": r.case_id,
                "category": r.category,
                "is_malicious": r.is_malicious,
                "adapter": r.adapter_id,
                "mode": r.mode,
                "model": r.model_name,
                "defense_enabled": r.defense_enabled,
                "attack_attempted": r.attack_attempted,
                "unauthorized_attempts": r.unauthorized_attempts,
                "unauthorized_executions": r.unauthorized_executions,
                "blocked_attempts": r.blocked_attempts,
                "attack_success": r.attack_success,
                "defense_success": r.defense_success,
                "benign_task_completed": r.benign_task_completed,
                "false_positive": r.false_positive,
                "status": r.status,
                "total_tool_calls": r.total_tool_calls,
                "duration_ms": r.duration_ms,
                "error_summary": r.error_summary,
                "evaluation_error": r.evaluation_error,
                "attack_evaluable": r.attack_evaluable,
                "task_evaluable": r.task_evaluable,
                "evaluation_note": r.evaluation_note,
            }
        )
    return rows


def events_to_rows(run: ExperimentRun) -> List[Dict[str, Any]]:
    rows = []
    for r in run.results:
        for e in r.events:
            rows.append(
                {
                    "run_id": e.run_id,
                    "case_id": e.case_id,
                    "round_index": e.round_index,
                    "event_id": e.event_id,
                    "tool_name": e.tool_name,
                    "arguments": json.dumps(
                        redact_arguments(e.arguments, max_length=None), ensure_ascii=False
                    ),
                    "executed_arguments": json.dumps(
                        redact_arguments(e.executed_arguments, max_length=None)
                        if e.executed_arguments is not None else None, ensure_ascii=False
                    ),
                    "policy_decision": e.policy_decision,
                    "policy_reason": e.policy_reason,
                    "is_unauthorized": e.is_unauthorized,
                    "executed": e.executed,
                    "simulated": e.simulated,
                    "defense_enabled": e.defense_enabled,
                    "risk_level": e.risk_level,
                    "adapter": e.adapter_id,
                    "result_summary": e.result_summary,
                    # 观察文本：真 IPI 的注入载体，必须导出以保证实验可审计
                    "observation": getattr(e, "observation", "") or "",
                    "result_ids": json.dumps(e.result_ids, ensure_ascii=False),
                }
            )
    return rows


def metrics_to_rows(run: ExperimentRun) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in ("case_level", "call_level", "exposure"):
        for key, value in run.summary_metrics.get(group, {}).items():
            if isinstance(value, dict) and "display" in value:
                rows.append(
                    {
                        "run_id": run.run_id,
                        "group": group,
                        "metric": key,
                        "name": value.get("name", key),
                        "value": value.get("value"),
                        "display": value.get("display"),
                        "numerator": value.get("numerator"),
                        "denominator": value.get("denominator"),
                    }
                )
            elif isinstance(value, (int, float)):
                rows.append(
                    {
                        "run_id": run.run_id,
                        "group": group,
                        "metric": key,
                        "name": key,
                        "value": value,
                        "display": value,
                        "numerator": None,
                        "denominator": None,
                    }
                )
    return rows


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 便捷导出
# --------------------------------------------------------------------------- #
def export_cases_csv(run: ExperimentRun) -> str:
    return rows_to_csv(cases_to_rows(run))


def export_events_csv(run: ExperimentRun) -> str:
    return rows_to_csv(events_to_rows(run))


def export_metrics_csv(run: ExperimentRun) -> str:
    return rows_to_csv(metrics_to_rows(run))


def export_comparison_csv(runs: List[ExperimentRun]) -> str:
    return rows_to_csv(comparison_table(runs))


def export_markdown(run: ExperimentRun, runs: Optional[List[ExperimentRun]] = None) -> str:
    """生成一份简洁的 Markdown 实验摘要（不含任何伪造结论）。"""
    label = mode_label(run.mode)
    cl = run.summary_metrics.get("case_level", {})
    call = run.summary_metrics.get("call_level", {})

    def disp(key: str) -> str:
        return cl.get(key, {}).get("display", "N/A")

    lines = [
        f"# AgentShield 实验摘要 · {run.run_id}",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 运行模式：{label}",
        f"- 适配器 / 模型：{run.adapter_id} / {run.model_name}",
        f"- 防御状态：{'开启' if run.defense_enabled else '关闭'}",
        f"- 案例数：{run.total_cases}（恶意 {cl.get('malicious_cases', 0)} / 正常 {cl.get('benign_cases', 0)}）",
        f"- 最大轮数：{run.max_rounds}，温度：{run.temperature}",
        f"- 案例集版本：{run.case_set_version}，策略版本：{run.policy_version}",
        f"- Prompt 级防御：{defense_label(run.prompt_defense)}",
        _attacker_line(run),
        f"- 攻击更新方式：{'首轮生成后固定复用' if run.attacker_mode == 'frozen' else '每轮更新'}；IPI LLM 改写：{run.ipi_llm}",
        f"- 研究批次：{run.batch_id or '单次运行'}；分组：{run.experiment_group or '无'}；重复序号：{run.trial_index}",
        f"- 模型请求调用 / 失败：{run.api_calls} / {run.api_failures}（含客户端重试；不等于失败案例数）",
        "- 口径：ASR/防御成功率按有效目标判定病例统计；运行错误另列，已确认的攻击成功不因后续报错删除。",
        "- BTSR/UUA 使用来源证据与回答要点规则；未配置规则的案例不计入相应分母。UPR/UER 保留全部病例的调用观测。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 | 分子/分母 |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("attack_success_rate", "ASR 攻击成功率"),
        ("unauthorized_attempt_rate", "UPR 违规提案率"),
        ("unauthorized_execution_rate", "UER 违规执行率"),
        ("defense_success_rate", "防御成功率"),
        ("benign_completion_rate", "BTSR 正常任务成功率"),
        ("false_positive_rate", "误拦截率（案例级）"),
        ("uua", "UUA 攻击下任务效用"),
        ("error_rate", "运行失败率"),
    ):
        item = cl.get(key, {})
        lines.append(
            f"| {label} | {item.get('display', 'N/A')} | "
            f"{item.get('numerator')} / {item.get('denominator')} |"
        )

    block = call.get("block_rate", {})
    ebr = call.get("ebr", {})
    fpr = call.get("execution_fpr", {})
    lines += [
        f"| 防御阻断率（调用级） | {block.get('display', 'N/A')} | "
        f"{block.get('numerator')} / {block.get('denominator')} |",
        f"| EBR 执行阻断率 | {ebr.get('display', 'N/A')} | "
        f"{ebr.get('numerator')} / {ebr.get('denominator')} |",
        f"| FPR 执行层误拦截率 | {fpr.get('display', 'N/A')} | "
        f"{fpr.get('numerator')} / {fpr.get('denominator')} |",
        "",
        "## 调用级统计",
        "",
        f"- 工具调用总数：{call.get('total_tool_calls', 0)}",
        f"- 实际执行：{call.get('executed_tool_calls', 0)}",
        f"- 越权尝试：{call.get('unauthorized_attempts', 0)}",
        f"- 越权执行：{call.get('unauthorized_executions', 0)}",
        f"- 被拦截：{call.get('blocked_unauthorized_calls', 0)}",
        f"- 参数不合法：{call.get('invalid_calls', 0)}",
        f"- 执行错误：{call.get('error_calls', 0)}",
        "",
        "> 说明：所有工具均为本地模拟工具，不产生真实外部副作用；",
        "> 本摘要只反映本次运行的真实记录，未做任何推断性结论。",
    ]

    if runs and len(runs) > 1:
        lines += ["", "## 多组对比", ""]
        rows = comparison_table(runs)
        if rows:
            header = list(rows[0].keys())
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(row[h]) for h in header) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def save_run(
    run: ExperimentRun,
    directory: Path,
    extra_runs: Optional[List[ExperimentRun]] = None,
) -> Dict[str, Path]:
    """把一次实验运行的 JSON/CSV/Markdown 写入目录，返回文件路径映射。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{run.run_id}_defense-{'on' if run.defense_enabled else 'off'}"

    paths = {
        "json": directory / f"{base}.json",
        "cases_csv": directory / f"{base}_cases.csv",
        "events_csv": directory / f"{base}_events.csv",
        "metrics_csv": directory / f"{base}_metrics.csv",
        "markdown": directory / f"{base}_summary.md",
    }
    paths["json"].write_text(run_to_json(run), encoding="utf-8")
    paths["cases_csv"].write_text(export_cases_csv(run), encoding="utf-8-sig")
    paths["events_csv"].write_text(export_events_csv(run), encoding="utf-8-sig")
    paths["metrics_csv"].write_text(export_metrics_csv(run), encoding="utf-8-sig")
    paths["markdown"].write_text(
        export_markdown(run, extra_runs or [run]), encoding="utf-8"
    )
    return paths
