"""配置加载与密钥安全管理。

配置优先级：环境变量 > 项目根目录 `.env` > 代码内置默认值。
本模块绝不打印完整 API Key；对外只提供掩码指纹。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .schemas import CASE_SET_VERSION

try:  # python-dotenv 是可选依赖，缺失时退化到手动解析
    from dotenv import load_dotenv  # type: ignore

    _HAS_DOTENV = True
except Exception:  # pragma: no cover
    _HAS_DOTENV = False


def _project_root() -> Path:
    """返回项目根目录（src/agentshield/config.py 的上三级）。"""
    return Path(__file__).resolve().parents[2]


def _manual_load_dotenv(path: Path) -> None:
    """极简 .env 解析，仅用于 python-dotenv 不可用时的兜底。"""
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except OSError:
        pass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 本地 / 局域网 OpenAI 兼容服务（AMD AUP 上为 llama.cpp + ROCm 提供的 GGUF 模型）
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
# 模型标识留空：运行时通过 GET /v1/models 自动发现服务端实际加载的模型，
# 避免写死一个并不存在的模型名（不同 GGUF 文件名/alias 会不同）。


@dataclass
class Settings:
    """全局运行配置。"""

    root: Path = field(default_factory=_project_root)
    deepseek_api_key: str = ""
    deepseek_base_url: str = DEFAULT_BASE_URL
    deepseek_model: str = DEFAULT_MODEL
    default_mode: str = "mock"
    request_timeout: float = 45.0
    max_retries: int = 2
    max_output_tokens: int = 1024
    # 默认值从 schemas 派生（单一来源），避免与实际案例集版本脱节
    case_set_version: str = CASE_SET_VERSION

    # ---- 本地 / 局域网 OpenAI 兼容服务（llama.cpp + ROCm）----
    local_base_url: str = DEFAULT_LOCAL_BASE_URL
    # 留空表示运行时自动发现（GET /v1/models）
    local_model: str = ""
    # 本地服务通常不校验密钥；留空时客户端内部使用占位值 "no-key"
    local_api_key: str = ""
    # Qwen3 等思考型模型需要 enable_thinking=false，否则可能只输出 reasoning_content
    local_disable_thinking: bool = True
    local_timeout: float = 120.0
    local_max_output_tokens: int = 1024

    # ---- 派生路径 ----
    @property
    def data_dir(self) -> Path:
        p = self.root / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = self.root / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- 密钥 ----
    @property
    def has_api_key(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key.strip())

    def key_fingerprint(self) -> str:
        """返回安全的密钥指纹，绝不泄露完整密钥。"""
        key = (self.deepseek_api_key or "").strip()
        if not key:
            return "未配置"
        tail = key[-4:] if len(key) >= 4 else "****"
        return f"已配置（长度 {len(key)}，尾号 …{tail}）"

    @property
    def local_model_display(self) -> str:
        """本地模型标识的展示值（未配置时说明会自动发现）。"""
        return self.local_model.strip() or "自动发现（GET /v1/models）"


_SETTINGS: Optional[Settings] = None


def load_settings(root: Optional[Path] = None, reload: bool = False) -> Settings:
    """加载配置；默认缓存，可通过 reload=True 强制刷新。"""
    global _SETTINGS
    if _SETTINGS is not None and not reload and root is None:
        return _SETTINGS

    base = Path(root) if root else _project_root()
    env_path = base / ".env"
    if _HAS_DOTENV:
        load_dotenv(env_path, override=False)
    else:  # pragma: no cover
        _manual_load_dotenv(env_path)

    def _get(name: str, default: str) -> str:
        value = os.environ.get(name, "")
        return value.strip() if value.strip() else default

    raw_timeout = os.environ.get("DEEPSEEK_TIMEOUT", "")
    raw_retries = os.environ.get("DEEPSEEK_MAX_RETRIES", "")
    raw_tokens = os.environ.get("AGENTSHIELD_MAX_OUTPUT_TOKENS", "")

    def _get_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes", "on")

    settings = Settings(
        root=base,
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=_get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        deepseek_model=_get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        default_mode=_get("AGENTSHIELD_DEFAULT_MODE", "mock").lower(),
        request_timeout=float(raw_timeout) if raw_timeout else 45.0,
        max_retries=int(raw_retries) if raw_retries else 2,
        max_output_tokens=int(raw_tokens) if raw_tokens else 1024,
        case_set_version=_get("AGENTSHIELD_CASE_SET_VERSION", CASE_SET_VERSION),
        local_base_url=_get("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
        local_model=os.environ.get("LOCAL_LLM_MODEL", "").strip(),
        local_api_key=os.environ.get("LOCAL_LLM_API_KEY", "").strip(),
        local_disable_thinking=_get_bool("LOCAL_LLM_DISABLE_THINKING", True),
        local_timeout=float(os.environ.get("LOCAL_LLM_TIMEOUT", "") or 120.0),
        local_max_output_tokens=int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "") or 1024),
    )
    _SETTINGS = settings
    return settings


def get_settings() -> Settings:
    if _SETTINGS is None:
        return load_settings()
    return _SETTINGS
