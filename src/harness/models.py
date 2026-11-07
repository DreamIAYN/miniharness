"""Model catalog: direct OpenAI-compatible providers plus OpenCode Zen.

A single harness instance can expose several models.  Callers select one with
``RunRequest.model``; unknown aliases return ``400 model_not_found``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import LLMNotConfigured, ModelNotFound


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    provider: str
    model_id: str
    base_url: str
    api_key: str
    context_window: int = 1_000_000
    max_output_tokens: int = 393_216
    supports_tools: bool = True
    supports_vision: bool = False
    # "enabled" / "disabled" / "omit" / None (use global LLM_THINKING).
    thinking: str | None = None
    label: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "id": self.alias,
            "provider": self.provider,
            "model": self.model_id,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "label": self.label or self.alias,
        }


# Metadata for models verified against OpenCode Zen.  Unknown model IDs get
# conservative defaults so they can still be enabled explicitly.
ZEN_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek-v4.1-flash": {
        "context_window": 1_000_000,
        "max_output_tokens": 393_216,
        "supports_vision": True,
        "thinking": None,
        "label": "DeepSeek V4.1 Flash",
    },
    "deepseek-v4-pro": {
        "context_window": 1_000_000,
        "max_output_tokens": 393_216,
        "supports_vision": False,
        "thinking": None,
        "label": "DeepSeek V4 Pro",
    },
    "glm-5.3-flash": {
        "context_window": 200_000,
        "max_output_tokens": 131_072,
        "supports_vision": True,
        "thinking": "omit",
        "label": "GLM 5.3 Flash",
    },
    "glm-5.3": {
        "context_window": 200_000,
        "max_output_tokens": 131_072,
        "supports_vision": True,
        "thinking": "omit",
        "label": "GLM 5.3",
    },
    "minimax-m3": {
        "context_window": 1_000_000,
        "max_output_tokens": 131_072,
        "supports_vision": True,
        "thinking": "omit",
        "label": "MiniMax M3",
    },
}

ZEN_DEFAULTS: dict[str, Any] = {
    "context_window": 131_072,
    "max_output_tokens": 32_768,
    "supports_vision": False,
    "thinking": "omit",
}


class ModelCatalog:
    def __init__(self, specs: dict[str, ModelSpec], default_alias: str | None) -> None:
        self._specs = specs
        self.default_alias = default_alias

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelCatalog:
        specs: dict[str, ModelSpec] = {}

        if settings.llm_api_key and settings.llm_base_url and settings.llm_model:
            direct_alias = settings.llm_model
            specs[direct_alias] = ModelSpec(
                alias=direct_alias,
                provider="direct",
                model_id=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                context_window=settings.agent_context_window_tokens,
                max_output_tokens=settings.agent_max_output_tokens,
                supports_vision=True,
                thinking=None,
                label=f"Direct {settings.llm_model}",
            )

        if settings.opencode_zen_api_key:
            for model_id in settings.opencode_zen_models:
                if not model_id or model_id in specs:
                    continue
                metadata = {**ZEN_DEFAULTS, **ZEN_PRESETS.get(model_id, {})}
                specs[model_id] = ModelSpec(
                    alias=model_id,
                    provider="opencode_zen",
                    model_id=model_id,
                    base_url=settings.opencode_zen_base_url,
                    api_key=settings.opencode_zen_api_key,
                    **metadata,
                )

        preferred = settings.llm_default_model or settings.llm_model
        default_alias = preferred if preferred in specs else next(iter(specs), None)
        return cls(specs, default_alias)

    def resolve(self, alias: str | None = None) -> ModelSpec:
        if not self._specs:
            raise LLMNotConfigured()
        name = alias or self.default_alias
        spec = self._specs.get(name) if name else None
        if spec is None:
            raise ModelNotFound(str(name or ""), sorted(self._specs))
        return spec

    def list(self) -> list[ModelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.alias)

    def aliases(self) -> list[str]:
        return sorted(self._specs)
