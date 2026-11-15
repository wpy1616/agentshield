"""LocalLLMAdapter：通过本地 / 局域网 OpenAI 兼容服务驱动的被测 Agent。

典型后端是 AMD AUP Learning Cloud 上 **llama.cpp（ROCm 构建）+ GGUF 量化模型**
（例如 Qwen3-0.6B）提供的 `/v1` HTTP 服务：推理发生在本地 GPU，不经过外部 API。

与 DeepSeek 适配器共享同一套提示词构造、JSON 解析、工具调用标准化逻辑，
差别只在模型客户端：

* `base_url` 指向本地服务（默认 `http://127.0.0.1:8080/v1`），通常不需要 API Key；
* 通过 `extra_body` 传 `chat_template_kwargs={"enable_thinking": False}`，
  避免 Qwen3 只输出 `reasoning_content` 而没有可解析正文；
* 模型标识留空时由客户端通过 `GET /v1/models` 自动发现，不臆造模型名。

安全约束与 DeepSeek 适配器完全一致：只解析 JSON、不执行任何模型返回的代码，
工具名必须落在白名单内，非法项由编排层记为 `invalid`。
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import Settings, get_settings
from ..model_clients import BaseLLMClient, LocalLLMClient
from ..schemas import RunMode
from .base import AdapterInfo, register_adapter
from .deepseek_adapter import DeepSeekAgentAdapter


@register_adapter("local")
class LocalLLMAdapter(DeepSeekAgentAdapter):
    """本地 llama.cpp（ROCm）OpenAI 兼容服务上的被测 Agent。"""

    adapter_id = "local"
    display_name = "本地 llama.cpp 模型（OpenAI 兼容 API）"
    run_mode = RunMode.LOCAL.value
    settings_model_field = "local_model"

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        model_name: str = "",
        settings: Optional[Settings] = None,
        base_url: str = "",
        disable_thinking: Optional[bool] = None,
        prompt_defense: str = "none",
        **options: Any,
    ) -> None:
        super().__init__(
            client=client,
            model_name=model_name,
            settings=settings,
            prompt_defense=prompt_defense,
            **options,
        )
        self.base_url = (base_url or self.settings.local_base_url).strip()
        self.disable_thinking = (
            self.settings.local_disable_thinking
            if disable_thinking is None
            else bool(disable_thinking)
        )
        # 本地服务不做 Mock 降级：失败就如实记录为错误，避免污染真实结果
        self.allow_mock_fallback = False

    # ------------------------------------------------------------------ #
    def _build_client(self) -> BaseLLMClient:
        return LocalLLMClient(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.settings.local_api_key,
            disable_thinking=self.disable_thinking,
            timeout=self.settings.local_timeout,
            max_output_tokens=self.settings.local_max_output_tokens,
        )

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id=self.adapter_id,
            display_name=self.display_name,
            description=(
                "调用本地 / 局域网的 OpenAI 兼容服务（llama.cpp + ROCm + GGUF 模型）"
                "作为被测 Agent，推理在本地 GPU 上完成，不需要 API Key。"
                "默认透传 chat_template_kwargs 关闭 thinking；"
                "小参数量模型的 JSON 与指令跟随稳定性有限，实验中会如实记为格式错误。"
            ),
            mode=self.run_mode,
            requires_api_key=False,
            supports_llm_attacker=False,
            default_model=self.settings.local_model_display,
            options=["model_name", "base_url", "temperature", "disable_thinking"],
        )

    def health_check(self) -> tuple[bool, str]:
        return self.client.test_connection()
