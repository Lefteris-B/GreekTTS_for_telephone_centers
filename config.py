"""
Persistent configuration for the Greek TTS app (Moira edition).

A few generation hyperparameters are exposed because Orpheus uses sampled
generation (non-deterministic). Most users won't touch these, but power
users might want to nudge temperature/top_p if output is unstable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


CONFIG_FILENAME = "config.json"


@dataclass
class AppConfig:
    """User-facing settings."""

    # Default telephony output format
    default_format: str = "alaw_8k"

    # Orpheus generation hyperparameters (defaults from the GreekTTS-1.5 model card)
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1200    # ~8–10 seconds of audio

    # Optional fixed seed for reproducible generations. None = random each time.
    seed: int | None = None

    # ----- persistence -----

    @classmethod
    def load(cls, app_dir: Path) -> "AppConfig":
        path = app_dir / CONFIG_FILENAME
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def save(self, app_dir: Path) -> None:
        path = app_dir / CONFIG_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
