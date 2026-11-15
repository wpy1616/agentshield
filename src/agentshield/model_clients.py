"""模型客户端：统一 `generate_json` 协议，供 Attacker / Target Agent 复用。

实现选择：使用 OpenAI 官方 Python SDK 的自定义 `base_url` 方式调用 DeepSeek
（DeepSeek 提供 OpenAI 兼容接口）。全项目只维护这一条真实调用链路，
另外提供一个完全离线的 `MockLLMClient` 用于流程测试与降级演示。
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .redaction import safe_error


# --------------------------------------------------------------------------- #
# 异常与响应
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    """模型调用失败。kind 用于 UI 给出针对性提示。"""

    def __init__(self, message: str, kind: str = "api_error") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class LLMResponse:
    content: str = ""
    data: Optional[Dict[str, Any]] = None
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    attempts: int = 1
    # 诊断信息（例如"只返回了 reasoning_content"），不参与指标计算
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.data is not None


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取第一个 JSON 对象（容忍代码块围栏与前后说明文字）。"""
    if not text:
        return None
    candidates = [text]
    fence = _JSON_FENCE.search(text)
    if fence:
        candidates.insert(0, fence.group(1))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # 尝试截取第一个 {...} 区块
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #
class BaseLLMClient(ABC):
    """模型客户端统一协议。"""

    model_name: str = "unknown"

    @abstractmethod
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """调用模型并尝试解析为 JSON 对象。"""

    def test_connection(self) -> tuple[bool, str]:
        return False, "该客户端未提供连接测试"


# --------------------------------------------------------------------------- #
# Mock 客户端（离线）
# --------------------------------------------------------------------------- #
class MockLLMClient(BaseLLMClient):
    """离线的确定性"伪模型"，用于流程测试与降级。

    它是规则驱动的，不代表任何真实模型能力；结果在 UI 中始终标记为 mock。
    """

    def __init__(
        self,
        model_name: str = "mock-llm-v1",
        responder: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        fail: bool = False,
    ) -> None:
        self.model_name = model_name
        self._responder = responder
        self._fail = fail
        self.calls = 0

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        self.calls += 1
        if self._fail:
            raise LLMError("（模拟）API 调用失败，用于测试错误处理路径", kind="simulated")
        start = time.perf_counter()
        if self._responder is not None:
            content = self._responder(messages)
        else:
            content = self._default_response(messages)
        return LLMResponse(
            content=content,
            data=extract_json_object(content),
            model=self.model_name,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def test_connection(self) -> tuple[bool, str]:
        return True, "Mock 模型客户端无需网络（非真实 API 结果）"

    @staticmethod
    def _default_response(messages: List[Dict[str, str]]) -> str:
        last = messages[-1]["content"] if messages else ""
        allowed = re.findall(r"`(search_emails|search_documents|read_sensitive_file|send_email)`", last)
        tool = allowed[0] if allowed else "search_emails"
        query = "进度"
        for kw in ("预算", "进度", "会议", "客户", "风险", "里程碑", "合规"):
            if kw in last:
                query = kw
                break
        payload = {
            "message": "（模拟模型）按任务检索相关信息。",
            "tool_calls": [{"tool": tool, "arguments": {"query": query}, "reason": "检索任务相关信息"}],
            "final_answer": "（模拟模型）已完成检索与总结。",
        }
        if tool == "read_sensitive_file":
            payload["tool_calls"][0]["arguments"] = {"file_id": "budget_2026Q1"}
        if tool == "send_email":
            payload["tool_calls"][0]["arguments"] = {
                "to": "external.partner@example.com",
                "subject": "项目资料（模拟）",
                "body": "（模拟）内容摘要。",
            }
        return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# DeepSeek 客户端（OpenAI 兼容接口）
# --------------------------------------------------------------------------- #
class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端（通过 OpenAI 兼容 SDK）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 45.0,
        max_retries: int = 2,
        max_output_tokens: int = 1024,
        extra_body: Optional[Dict[str, Any]] = None,
        require_api_key: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url
        self.model = model
        self.model_name = model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.max_output_tokens = int(max_output_tokens)
        # 透传给服务端的非标准字段（如 llama.cpp 的 chat_template_kwargs）
        self.extra_body: Dict[str, Any] = dict(extra_body or {})
        # 本地服务通常不需要密钥；为 False 时用占位值避免 SDK 抛错
        self.require_api_key = bool(require_api_key)
        self._client: Any = None
        self.total_calls = 0
        self.total_failures = 0

    # ---- 内部 ----
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            if self.require_api_key:
                raise LLMError(
                    "未配置 DEEPSEEK_API_KEY，请填写项目根目录 .env 后重启应用"
                    "（不会在界面显示密钥）",
                    kind="missing_key",
                )
            # 本地 OpenAI 兼容服务：用占位密钥，服务端一般不校验
            self.api_key = "no-key"
        try:
            from openai import OpenAI  # 延迟导入，避免无 SDK 时启动失败
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "缺少 openai SDK，请执行 pip install -r requirements.txt",
                kind="sdk_missing",
            ) from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self._client

    @staticmethod
    def _classify(exc: Exception) -> LLMError:
        name = type(exc).__name__.lower()
        message = safe_error(str(exc))
        if "authentication" in name or "401" in message:
            return LLMError(
                "API 认证失败：请检查 DEEPSEEK_API_KEY 是否正确、是否过期（不会显示密钥）。",
                kind="auth",
            )
        if "ratelimit" in name or "429" in message:
            return LLMError("API 触发限流：请稍后重试或减少并发案例数。", kind="rate_limit")
        if "insufficient" in message or "balance" in message or "quota" in message:
            return LLMError("API 余额不足或配额受限：请在 DeepSeek 控制台确认账户状态。", kind="quota")
        if "timeout" in name or "timed out" in message.lower():
            return LLMError(f"API 请求超时：{message}", kind="timeout")
        if "apiconnection" in name or "connection" in message.lower():
            return LLMError(f"无法连接 API 服务：{message}", kind="network")
        if "badrequest" in name or "400" in message:
            return LLMError(f"API 请求参数有误：{message}", kind="bad_request")
        if "notfound" in name or "404" in message:
            return LLMError(f"模型或接口不存在：{message}", kind="not_found")
        return LLMError(f"API 调用失败（{type(exc).__name__}）：{message}", kind="api_error")

    # ---- 主调用 ----
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        attempt = 0
        last_error: Optional[LLMError] = None
        max_attempts = self.max_retries + 1

        while attempt < max_attempts:
            attempt += 1
            self.total_calls += 1
            start = time.perf_counter()
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or self.max_output_tokens,
                }
                if self.extra_body:
                    # llama.cpp 等服务的非标准字段（如 chat_template_kwargs）
                    kwargs["extra_body"] = dict(self.extra_body)
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 统一转换 SDK 异常
                self.total_failures += 1
                last_error = self._classify(exc)
                if last_error.kind in ("auth", "quota", "bad_request", "not_found", "missing_key", "sdk_missing"):
                    raise last_error
                if attempt >= max_attempts:
                    raise last_error
                time.sleep(min(2.0, 0.5 * attempt))
                continue

            latency = int((time.perf_counter() - start) * 1000)
            message_obj = None
            content = ""
            try:
                message_obj = response.choices[0].message
                content = message_obj.content or ""
            except Exception:  # pragma: no cover
                content = ""
            note = ""
            if not content.strip() and message_obj is not None:
                # Qwen3 等思考型模型在 thinking 未关闭时只会返回 reasoning_content
                reasoning = str(getattr(message_obj, "reasoning_content", "") or "").strip()
                if reasoning:
                    note = (
                        "模型只返回了 reasoning_content（思考内容），未产生可解析正文；"
                        "请确认服务端已接受 chat_template_kwargs.enable_thinking=false"
                    )
            usage: Dict[str, int] = {}
            try:
                raw_usage = getattr(response, "usage", None)
                if raw_usage is not None:
                    usage = {
                        "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                    }
            except Exception:  # pragma: no cover
                usage = {}
            return LLMResponse(
                content=content,
                data=extract_json_object(content),
                model=getattr(response, "model", self.model) or self.model,
                usage=usage,
                latency_ms=latency,
                attempts=attempt,
                note=note,
            )

        assert last_error is not None
        raise last_error

    def test_connection(self) -> tuple[bool, str]:
        """发送一次极小请求验证连通性，不做任何破坏性操作。"""
        try:
            self._ensure_client()
        except LLMError as exc:
            return False, str(exc)
        try:
            response = self.generate_json(
                messages=[
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": '请输出 JSON：{"ok": true}'},
                ],
                temperature=0.0,
                max_tokens=32,
            )
        except LLMError as exc:
            return False, str(exc)
        if response.data is None:
            return False, f"API 返回内容无法解析为 JSON（模型：{response.model}）"
        return True, f"连接成功：模型 {response.model}，延迟约 {response.latency_ms} ms"


class LocalLLMClient(DeepSeekClient):
    """本地 / 局域网 OpenAI 兼容服务客户端（llama.cpp + ROCm 提供的 GGUF 模型）。

    与 `DeepSeekClient` 的三点差异：

    1. 请求发往本地服务（默认 `http://127.0.0.1:8080/v1`），不需要真实 API Key；
    2. 通过 `extra_body` 透传 llama.cpp 特有字段 `chat_template_kwargs`，
       用于关闭 Qwen3 等模型的 thinking，避免只输出 `reasoning_content`；
    3. 模型标识留空时通过 `GET /v1/models` 自动发现服务端实际加载的模型，
       不臆造模型名。

    推理完全发生在本地 GPU（AMD AUP），不经过任何外部 API。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "",
        api_key: str = "",
        disable_thinking: bool = True,
        timeout: float = 120.0,
        max_retries: int = 1,
        max_output_tokens: int = 1024,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        body: Dict[str, Any] = {}
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            body.update(extra_body)
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            max_output_tokens=max_output_tokens,
            extra_body=body,
            require_api_key=False,
        )
        self.disable_thinking = bool(disable_thinking)

    # ---- 模型标识自动发现 ----
    def _ensure_client(self) -> Any:
        client = super()._ensure_client()
        if not self.model:
            self.model = self._discover_model(client)
            self.model_name = self.model
        return client

    def _discover_model(self, client: Any) -> str:
        try:
            listed = client.models.list()
            data = getattr(listed, "data", None) or []
            if data:
                return str(data[0].id)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"无法从本地服务获取模型列表（{type(exc).__name__}）："
                f"请确认 {self.base_url} 可访问，或在 .env 中显式设置 LOCAL_LLM_MODEL",
                kind="local_unavailable",
            ) from exc
        raise LLMError(
            f"本地服务 {self.base_url} 未返回任何模型，请在 .env 中设置 LOCAL_LLM_MODEL",
            kind="local_unavailable",
        )

    # ---- 连接自检 ----
    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self.generate_json(
                messages=[
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": '请输出 JSON：{"ok": true}'},
                ],
                temperature=0.0,
                max_tokens=64,
            )
        except LLMError as exc:
            return False, f"{self.base_url} 连接失败：{exc}"
        if response.note:
            return False, f"{self.base_url} 返回异常：{response.note}"
        if response.data is None:
            preview = response.content.strip()[:80].replace("\n", " ")
            return False, (
                f"{self.base_url} 可达，但返回内容无法解析为 JSON"
                f"（模型 {response.model}；返回预览：{preview or '空'}）。"
                "小参数量模型的 JSON 稳定性有限，实验中会记为格式错误。"
            )
        return True, (
            f"连接成功：{self.base_url} · 模型 {response.model} · "
            f"延迟约 {response.latency_ms} ms"
            + ("（thinking 已关闭）" if self.disable_thinking else "")
        )


def build_client(
    mode: str,
    settings: Any,
    model_name: Optional[str] = None,
) -> BaseLLMClient:
    """根据模式构建模型客户端。

    mode='deepseek' 要求有 API Key；
    mode='local' 指向本地 / 局域网 OpenAI 兼容服务（llama.cpp + ROCm），不需要 Key；
    其余情况返回离线 Mock 客户端。
    """
    if mode == "deepseek":
        return DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=model_name or settings.deepseek_model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            max_output_tokens=settings.max_output_tokens,
        )
    if mode == "local":
        return LocalLLMClient(
            base_url=settings.local_base_url,
            model=model_name or settings.local_model,
            api_key=settings.local_api_key,
            disable_thinking=settings.local_disable_thinking,
            timeout=settings.local_timeout,
            max_output_tokens=settings.local_max_output_tokens,
        )
    return MockLLMClient()
