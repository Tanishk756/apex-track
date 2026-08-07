"""
Config Manager
==============
Loads, merges, and provides live access to ApexConfig.

Merge order (later overrides earlier):
    1. configs/default.yaml          (bundled defaults)
    2. configs/<hw_profile>.yaml     (hardware-specific overrides)
    3. User-specified --config file  (operator overrides)
    4. Environment variables         (APEX_* prefix, e.g. APEX_DETECTOR__PLUGIN=yolo11)
    5. Runtime API updates           (hot-reload, no restart needed)

Environment variable convention:
    APEX_DETECTOR__PLUGIN=yolo11       → config.detector.plugin = "yolo11"
    APEX_HARDWARE__FORCE_CPU=true      → config.hardware.force_cpu = True
    APEX_TELEMETRY__CONNECTION=...     → config.telemetry.connection = ...
    (double underscore = nested path separator)
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml
from pydantic import ValidationError

from apex.engine.config.schema import ApexConfig

log = structlog.get_logger(__name__)

# Bundled default config location (relative to this file)
_BUILTIN_DEFAULTS = Path(__file__).parent.parent.parent.parent.parent / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Returns empty dict if file not found."""
    if not path.exists():
        log.debug("config_file_not_found", path=str(path))
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        log.debug("config_file_loaded", path=str(path))
        return data
    except Exception as exc:
        log.error("config_file_error", path=str(path), error=str(exc))
        return {}


def _apply_env_vars(data: dict) -> dict:
    """
    Apply APEX_* environment variables to the config dict.
    APEX_DETECTOR__PLUGIN=yolo11 → data["detector"]["plugin"] = "yolo11"
    """
    for key, value in os.environ.items():
        if not key.startswith("APEX_"):
            continue
        # Strip prefix and split on double underscore
        path_str = key[5:]  # remove APEX_
        parts = [p.lower() for p in path_str.split("__")]
        if not parts:
            continue

        # Navigate to the nested dict and set the value
        target = data
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        # Type-coerce booleans and numbers
        coerced: Any = value
        if value.lower() in ("true", "1", "yes"):
            coerced = True
        elif value.lower() in ("false", "0", "no"):
            coerced = False
        else:
            try:
                coerced = int(value)
            except ValueError:
                try:
                    coerced = float(value)
                except ValueError:
                    pass  # keep as string

        target[parts[-1]] = coerced
        log.debug("env_var_applied", key=key, path=".".join(parts))

    return data


class ConfigManager:
    """
    Singleton configuration manager with hot-reload support.

    Usage::
        cfg_mgr = ConfigManager.instance()
        cfg_mgr.load(hw_profile_name="desktop_rtx", user_config=Path("my.yaml"))
        config = cfg_mgr.config   # ApexConfig instance

        # Hot-reload a sub-section at runtime (e.g. from API)
        cfg_mgr.update({"detector": {"confidence_threshold": 0.3}})
    """

    _instance: "ConfigManager | None" = None

    def __init__(self) -> None:
        self._raw: dict = {}
        self._config: Optional[ApexConfig] = None
        self._listeners: list = []

    @classmethod
    def instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def load(
        self,
        hw_profile_name: str = "cpu_only",
        user_config: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
    ) -> "ApexConfig":
        """
        Load and merge the full configuration stack.
        Returns the validated ApexConfig.
        """
        base_dir = configs_dir or _BUILTIN_DEFAULTS.parent

        # 1. Built-in defaults
        raw = _load_yaml(_BUILTIN_DEFAULTS)

        # 2. Hardware profile overrides
        hw_config_path = base_dir / f"{hw_profile_name}.yaml"
        raw = _deep_merge(raw, _load_yaml(hw_config_path))

        # 3. User config file
        if user_config:
            raw = _deep_merge(raw, _load_yaml(user_config))

        # 4. Environment variables
        raw = _apply_env_vars(raw)

        self._raw = raw
        self._config = self._validate(raw)
        log.info("config_loaded", hw_profile=hw_profile_name)
        return self._config

    def update(self, partial: dict) -> "ApexConfig":
        """
        Hot-reload: merge partial config into current config and re-validate.
        Notifies listeners on success.
        """
        if self._raw is None:
            raise RuntimeError("ConfigManager.load() must be called before update()")
        self._raw = _deep_merge(self._raw, partial)
        self._config = self._validate(self._raw)
        for fn in self._listeners:
            try:
                fn(self._config)
            except Exception as exc:
                log.warning("config_listener_error", error=str(exc))
        log.info("config_hot_reloaded", keys=list(partial.keys()))
        return self._config

    def _validate(self, raw: dict) -> ApexConfig:
        try:
            return ApexConfig.model_validate(raw)
        except ValidationError as exc:
            log.error("config_validation_failed", errors=exc.errors())
            raise

    @property
    def config(self) -> ApexConfig:
        if self._config is None:
            raise RuntimeError("ConfigManager not initialized. Call load() first.")
        return self._config

    def on_update(self, fn) -> None:
        """Register a callback invoked after every hot-reload."""
        self._listeners.append(fn)

    def as_dict(self) -> dict:
        return self._config.model_dump() if self._config else {}
