"""
APEX-Track Security & Credential Masking Manager
=================================================
Secures API keys and sensitive credentials by reading from environment variables
or local .env files, and masking sensitive strings in logs and REST outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any
import structlog

log = structlog.get_logger(__name__)

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parents[3] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def mask_key(key: str, visible_prefix_len: int = 6) -> str:
    """Masks API keys e.g. 'sk-proj-abc...' -> 'sk-proj-***MASKED***'."""
    if not key:
        return "NOT_CONFIGURED"
    if len(key) <= visible_prefix_len:
        return "***MASKED***"
    return f"{key[:visible_prefix_len]}...***MASKED***"


class SecurityManager:
    """Centralized Security & API Key Provider."""

    _instance: SecurityManager | None = None

    def __init__(self) -> None:
        self.roboflow_key = os.environ.get("ROBOFLOW_API_KEY", "")
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.weather_key = os.environ.get("OPENWEATHER_API_KEY", "")

    @classmethod
    def instance(cls) -> SecurityManager:
        if cls._instance is None:
            cls._instance = SecurityManager()
        return cls._instance

    def get_security_audit(self) -> Dict[str, Any]:
        """Returns sanitized security audit report with all keys masked."""
        return {
            "security_level": "DEFENSE_HIGH",
            "env_file_detected": (Path(__file__).parents[3] / ".env").exists(),
            "roboflow_status": "ACTIVE" if self.roboflow_key else "INACTIVE",
            "roboflow_key_masked": mask_key(self.roboflow_key),
            "openai_status": "ACTIVE" if self.openai_key else "INACTIVE",
            "openai_key_masked": mask_key(self.openai_key),
            "weather_status": "ACTIVE" if self.weather_key else "INACTIVE",
            "weather_key_masked": mask_key(self.weather_key),
        }
