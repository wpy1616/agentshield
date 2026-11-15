"""AgentShield Streamlit 应用入口。

页面：概览 Dashboard / 运行实验 / 轨迹分析 / 实验对比 / 项目说明

UI 只负责配置与展示：业务逻辑全部在 src/agentshield 包内，
新增 Agent 适配器、工具或策略都不需要改动本文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentshield import __version__  # noqa: E402
from agentshield.adapters.base import list_adapters  # noqa: E402
from agentshield.config import get_settings  # noqa: E402
from agentshield.experiment import (  # noqa: E402
    ExperimentRunner,
    build_adapter,
    run_comparison,
)
from agentshield.export import (  # noqa: E402
    export_cases_csv,
    export_comparison_csv,
    export_events_csv,
    export_markdown,
    export_metrics_csv,
    run_to_json,
)
from agentshield.judge import status_label  # noqa: E402
from agentshield.metrics import comparison_table  # noqa: E402
from agentshield.policy import POLICY_VERSION  # noqa: E402
from agentshield.prompt_defenses import DEFENSE_CHOICES, defense_label  # noqa: E402
from agentshield.sample_cases import (  # noqa: E402
    ALL_CASES,
    CAPABILITY_CASES,
    CASE_SET_VERSION,
    get_cases,
)
from agentshield.schemas import ExperimentConfig  # noqa: E402

APP_TITLE = "Agent 工具调用安全测试平台"

# 运行模式的中文标签（mode 值 → 展示文案）
MODE_LABELS = {
    "mock": "Mock 模拟（非真实模型结果）",
    "deepseek": "DeepSeek API",
    "local": "本地 llama.cpp 模型（GPU 本地推理）",
}

st.set_page_config(page_title="Agent 工具调用安全测试平台", page_icon="🛡️", layout="wide")
st.title(APP_TITLE)
st.caption(
    "通过模拟任务，观察 AI 助手的操作表现，比较不同设置下的测试结果。"
    "邮件、文档和工具操作均为模拟，不会读取真实邮箱或文件，也不会真正发送邮件。"
)

settings = get_settings()


def _width_kwargs() -> dict:
    """兼容不同 Streamlit 版本的宽度参数（新版用 width='stretch'）。"""
    import inspect

    try:
        if "width" in inspect.signature(st.dataframe).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return {"use_container_width": True}


WIDTH = _width_kwargs()


# --------------------------------------------------------------------------- #
# 会话状态
# --------------------------------------------------------------------------- #
def init_state() -> None:
    st.session_state.setdefault("runs", [])  # List[ExperimentRun]
    st.session_state.setdefault("api_status", None)


init_state()


def remember_run(run) -> None:
    st.session_state.runs.append(run)
    st.session_state.runs = st.session_state.runs[-20:]


# --------------------------------------------------------------------------- #
# 侧边栏：Agent / 模型配置
# --------------------------------------------------------------------------- #
def sidebar() -> ExperimentConfig:
    st.sidebar.header("⚙️ 运行配置")

    adapters = list_adapters()
    adapter_map = {a.adapter_id: a for a in adapters}
    # 展示顺序：Mock 优先（保证无密钥时开箱可用），其余按 id 排序
    preferred = ["mock", "deepseek", "local"]
    adapter_ids = [a for a in preferred if a in adapter_map]

    st.sidebar.subheader("1. 被测 Agent（适配器）")
    # 默认选择：配置要求 deepseek 且已配置 Key 时才默认 DeepSeek，否则一律 Mock
    default_id = "mock"
    if settings.default_mode == "deepseek" and settings.has_api_key and "deepseek" in adapter_ids:
        default_id = "deepseek"
    elif settings.default_mode == "local" and "local" in adapter_ids:
        default_id = "local"
    adapter_id = st.sidebar.selectbox(
        "Agent 类型 / 适配器",
        options=adapter_ids,
        index=adapter_ids.index(default_id) if default_id in adapter_ids else 0,
        format_func=lambda x: f"{adapter_map[x].display_name} ({x})",
        help="不同 Agent 通过统一的 AgentAdapter 接口接入，输出会转换为同一套标准事件格式。",
    )
    info = adapter_map[adapter_id]
    st.sidebar.caption(info.description)

    # ---- 模型与模式 ----
    st.sidebar.subheader("2. 模型配置")

    def _run_health_check(label_adapter_id: str, label_model: str) -> None:
        with st.spinner("正在测试连接（不会执行任何业务操作）…"):
            try:
                adapter = build_adapter(
                    ExperimentConfig(adapter_id=label_adapter_id, model_name=label_model),
                    settings,
                )
                ok, message = adapter.health_check()
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"连接测试异常：{type(exc).__name__}"
            st.session_state.api_status = (ok, message)

    if adapter_id == "deepseek":
        model_name = st.sidebar.text_input(
            "模型名", value=settings.deepseek_model or "deepseek-chat"
        )
        mode = "deepseek"
        st.sidebar.markdown(f"API Key 状态：**{settings.key_fingerprint()}**")
        st.sidebar.caption(f"Base URL：{settings.deepseek_base_url}（密钥不会显示）")
        if not settings.has_api_key:
            st.sidebar.warning(
                "未检测到 DEEPSEEK_API_KEY。请在项目根目录 `.env` 中填写后重启应用；"
                "当前只能使用 Mock 模式。"
            )
        if st.sidebar.button("🔌 测试 API 连接", key="health_deepseek"):
            _run_health_check("deepseek", model_name)
    elif adapter_id == "local":
        model_name = st.sidebar.text_input(
            "模型名（留空则自动发现）",
            value=settings.local_model,
            placeholder="留空时读取服务端 /v1/models",
        )
        mode = "local"
        st.sidebar.caption(f"服务地址：`{settings.local_base_url}`（LOCAL_LLM_BASE_URL）")
        st.sidebar.caption(
            f"模型：`{settings.local_model_display}` · "
            f"关闭 thinking：`{settings.local_disable_thinking}`"
            "（Qwen3 需要关闭，否则可能只输出思考内容）"
        )
        st.sidebar.info(
            "被测模型使用本地 llama.cpp（ROCm）+ GGUF 推理。"
            "小参数量模型的 JSON 稳定性有限，解析失败会如实记为格式错误。"
        )
        if st.sidebar.button("🔌 测试本地服务连接", key="health_local"):
            _run_health_check("local", model_name)
    else:
        model_name = "mock-simulated-v1"
        mode = "mock"
        st.sidebar.info("当前为 **Mock 模拟模式**：无需 API Key，结果为模拟输出，**不是真实模型结果**。")

    if st.session_state.api_status:
        ok, message = st.session_state.api_status
        (st.sidebar.success if ok else st.sidebar.error)(message)

    temperature = st.sidebar.slider(
        "温度 temperature", 0.0, 1.0, 0.0, 0.1,
        help="仅对真实 API 生效；Mock 模式忽略该参数。建议保持较低温度以提升可复现性。",
    )

    # ---- 实验参数 ----
    st.sidebar.subheader("3. 实验参数")
    max_rounds = st.sidebar.slider("最大运行轮数", 1, 5, 3)
    seed = st.sidebar.number_input("随机种子（Mock 模式）", min_value=0, value=42, step=1)
    prompt_defense = st.sidebar.selectbox(
        "Prompt 级防御", DEFENSE_CHOICES, format_func=defense_label,
        help="控制被测模型的输入格式。Mock 模式不解析提示词。",
    )
    defense_on = st.sidebar.checkbox("执行层防御", value=True)
    st.sidebar.caption(f"公开版 v{__version__} · {CASE_SET_VERSION} · {POLICY_VERSION}")

    return ExperimentConfig(
        mode=mode, adapter_id=adapter_id, model_name=model_name, defense_enabled=defense_on,
        max_rounds=max_rounds, temperature=temperature, seed=int(seed), prompt_defense=prompt_defense,
    )


config = sidebar()
tabs = st.tabs(["📊 概览", "▶️ 运行实验", "🔍 轨迹分析", "⚖️ 实验对比", "ℹ️ 项目说明"])
runs = st.session_state.runs

with tabs[0]:
    st.header("概览")
    total, normal, capability = st.columns(3)
    total.metric("内置案例", len(ALL_CASES))
    normal.metric("正常办公任务", len(ALL_CASES) - len(CAPABILITY_CASES))
    capability.metric("合法操作验证", len(CAPABILITY_CASES))
    st.dataframe(
        pd.DataFrame([{"案例": case.case_id, "任务": case.task,
                       "授权工具": "、".join(case.expected_allowed_tools)} for case in ALL_CASES]),
        hide_index=True, **WIDTH,
    )

with tabs[1]:
    st.header("运行实验")
    choices = {"all": "全部样例", "benign": "正常办公任务", "capability": "合法操作能力"}
    suite = st.selectbox("案例集", list(choices), format_func=choices.get, index=1)
    cases = get_cases(suite)
    if config.mode == "mock" and suite != "benign":
        st.caption("Mock 按案例演示授权步骤。选择真实模型适配器可以观察模型如何完成这些操作。")

    run_single, run_both = st.columns(2)
    single = run_single.button("开始实验", key="run_single", type="primary")
    compare = run_both.button("一键运行对照实验", key="run_comparison")
    if single or compare:
        try:
            with st.spinner(f"正在运行 {len(cases)} 个案例…"):
                if single:
                    adapter = build_adapter(config, settings)
                    runner = ExperimentRunner(adapter, settings=settings)
                    fresh = [runner.run(cases, config)]
                else:
                    fresh = list(run_comparison(cases, config, settings).values())
        except (ValueError, RuntimeError) as exc:
            st.error(f"运行失败：{type(exc).__name__}")
        else:
            for run in fresh:
                remember_run(run)
            st.success(f"已完成 {len(fresh)} 组实验。可以在「轨迹分析」查看并下载结果。")
    runs = st.session_state.runs
    if runs:
        latest = runs[-1]
        summary = latest.summary_metrics
        evaluated = summary["case_level"]
        calls = summary["call_level"]
        a, b, c = st.columns(3)
        a.metric("任务完成（可判定）", evaluated["btsr"]["display"])
        b.metric("越权提案次数", calls["unauthorized_attempts"])
        c.metric("执行层拦截次数", calls["blocked_unauthorized_calls"])
        st.caption(f"最近一次运行 {latest.run_id} · 解析或运行出错 {evaluated['error_cases']}/{latest.total_cases}")

with tabs[2]:
    st.header("轨迹分析")
    if not runs:
        st.info("请先在「运行实验」完成一组测试。")
    else:
        current = st.selectbox(
            "运行记录", runs, index=len(runs) - 1, format_func=lambda run: run.run_id,
        )
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.download_button("运行 JSON", run_to_json(current), f"{current.run_id}.json", "application/json")
        d2.download_button("案例 CSV", export_cases_csv(current), f"{current.run_id}-cases.csv", "text/csv")
        d3.download_button("调用 CSV", export_events_csv(current), f"{current.run_id}-events.csv", "text/csv")
        d4.download_button("指标 CSV", export_metrics_csv(current), f"{current.run_id}-metrics.csv", "text/csv")
        d5.download_button("运行摘要", export_markdown(current), f"{current.run_id}.md", "text/markdown")
        for case in current.results:
            title = f"{case.case_id} · {status_label(case.status)}"
            with st.expander(title):
                st.write(case.case_snapshot.get("task", ""))
                if case.error_summary:
                    st.warning(case.error_summary)
                for rd in case.rounds:
                    st.markdown(f"**第 {rd.round_index + 1} 轮**")
                    if rd.agent_final_answer:
                        st.write(rd.agent_final_answer)
                    if rd.events:
                        st.dataframe(pd.DataFrame([{
                            "工具": event.tool_name, "参数": event.arguments,
                            "策略": event.policy_decision, "原因": event.policy_reason,
                            "已执行": event.executed, "结果": event.result_summary,
                        } for event in rd.events]), **WIDTH, hide_index=True)
                    if rd.request_messages and st.checkbox(
                        "显示本轮目标模型输入",
                        key=f"input-{current.run_id}-{case.case_id}-{rd.round_index}",
                    ):
                        st.json(rd.request_messages)

with tabs[3]:
    st.header("实验对比")
    if not runs:
        st.info("运行一键对照实验后，可以在此比较两个分组。")
    else:
        st.dataframe(pd.DataFrame(comparison_table(runs)), **WIDTH, hide_index=True)
        st.download_button("下载对比 CSV", export_comparison_csv(runs), "agentshield-comparison.csv", "text/csv")

with tabs[4]:
    st.header("项目说明")
    st.markdown("""
公开版提供正常任务评估、任务授权策略对照、轨迹查看和报告导出。默认使用免密钥的
Mock 适配器。需要测量真实模型行为时，可将 DeepSeek API 或本地模型作为被测 Agent。

邮件、文档、预算文件和外发操作都经过模拟。任务级授权策略在工具执行前检查工具权限、
数据范围和参数。关闭执行层防御时仍有工具白名单、参数校验和调用限制。

仓库中的样例数据用于现场运行。课程报告的历史实验记录与公开代码分别归档。
""")
