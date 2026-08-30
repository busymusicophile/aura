"""
AURA — central configuration.

Every path and tunable lives here so no module hardcodes a location.

Layout rationale: code lives in the OneDrive project folder (versioned, synced,
small), while data and models live under C:\\AURA (large, churny, and in Phase 12
encrypted at rest). Nothing private is ever written into the synced folder.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------

AURA_HOME = Path(os.environ.get("AURA_HOME", r"C:\AURA"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = AURA_HOME / "data"
LOG_DIR = DATA_DIR / "logs"
MEMORY_DIR = DATA_DIR / "memory"
PERSONA_DIR = DATA_DIR / "persona"
FACES_DIR = DATA_DIR / "faces"
AUDIT_LOG = DATA_DIR / "audit" / "audit.jsonl"

MODELS_DIR = AURA_HOME / "models"
TOOLS_DIR = AURA_HOME / "tools"

PIPER_EXE = TOOLS_DIR / "piper" / "piper.exe"
PIPER_VOICE = MODELS_DIR / "piper" / "en_GB-alan-medium.onnx"
PIPER_VOICE_BYTES = 63_201_294

PERSONA_FILE = PERSONA_DIR / "persona_profile.yaml"
PERSONA_TEMPLATE = PROJECT_ROOT / "aura" / "persona" / "persona_template.yaml"

ALL_DIRS = [
    DATA_DIR,
    LOG_DIR,
    MEMORY_DIR,
    PERSONA_DIR,
    FACES_DIR,
    AUDIT_LOG.parent,
]


def ensure_dirs() -> None:
    """Create every directory AURA writes to. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMSettings:
    model: str = "qwen3:8b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    # qwen3 is a hybrid reasoning model; AURA is conversational, so thinking is
    # off by default to keep latency low. Enable per-call for hard reasoning.
    think: bool = False
    context_tokens: int = 8192
    # Hard ceiling on one reply. Without this, a small model under a long or
    # conflicting system prompt can spiral into open-ended or repetitive
    # generation with nothing to stop it short of filling num_ctx entirely -
    # observed in practice as a 10+ minute hang on a one-line question.
    max_reply_tokens: int = 600
    # Discourages the token-by-token repetition loops ("no no no no...") that
    # are the other half of the same failure mode. 1.1 is llama.cpp's own
    # default; ollama does not set it for you.
    repeat_penalty: float = 1.1
    repeat_last_n: int = 256


@dataclass(frozen=True)
class MemorySettings:
    collection: str = "aura_memory"
    # Small, fast, runs on CPU so it never competes with the LLM for the 6GB VRAM.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    retrieve_k: int = 6
    # Corrections are the highest-signal memories; they get a relevance boost so
    # an explicit "no, I meant X" outranks a dozen casual mentions of X.
    correction_boost: float = 0.25


@dataclass(frozen=True)
class SpeechSettings:
    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute: str = "float16"
    language: str | None = None  # None = autodetect; she mixes en/hi/te
    sample_rate: int = 16_000
    vad_threshold: float = 0.5
    silence_ms: int = 700
    min_speech_ms: int = 250


@dataclass(frozen=True)
class VisionSettings:
    camera_index: int = 0
    det_size: tuple[int, int] = (640, 640)
    # Cosine similarity on 512-d ArcFace embeddings. 0.38 is a deliberately
    # conservative threshold: a false "that's Keerthana" is a privacy breach,
    # while a false "unknown" only costs a re-greeting.
    match_threshold: float = 0.38
    frame_interval: float = 0.5
    absent_after: float = 8.0


@dataclass(frozen=True)
class VoiceSettings:
    length_scale: float = 1.15
    noise_scale: float = 0.667
    sentence_silence: float = 0.2


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)

    primary_user: str = "Keerthana"
    autonomy_enabled: bool = False  # Phase 8 ships opt-in, never opt-out.


SETTINGS = Settings()


def venv_python() -> Path:
    """Path to the interpreter AURA runs under."""
    return Path(sys.executable)
