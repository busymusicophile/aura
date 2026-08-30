"""
AURA — speech output (Phase 3).

Wraps the standalone Piper binary. Piper is used as an executable rather than via
its Python package because the pip route is unreliable on Windows, and the binary
is fast enough that it does not matter: measured real-time factor on this machine
is 0.059, i.e. 2.6 seconds of speech synthesised in 0.15 seconds.

Long replies are synthesised and played one sentence at a time. Waiting for a
whole paragraph before the first word is audible makes AURA feel sluggish even
though total synthesis time is identical.
"""

from __future__ import annotations

import queue
import re
import subprocess
import tempfile
import threading
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from aura import config
from aura.safety import redaction

# Split on sentence enders, keeping the punctuation.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Emoji, dingbats and other pictographic symbols have no spoken form. Piper's
# phonemiser (espeak-ng) either drops them silently or garbles the surrounding
# words trying to render them - and on Windows, piping one through subprocess
# without an explicit encoding crashes outright, because Python defaults stdin
# pipes to the console codepage (cp1252 here), which cannot represent emoji at
# all. Stripped before synthesis; the original text is kept for chat, memory
# and the audit log, which all handle Unicode natively.
_UNSPEAKABLE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoticons, supplemental
    "\U00002600-\U000027BF"  # misc symbols and dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flag letters)
    "\U0000FE0F"              # variation selector-16
    "\U0000200D"              # zero-width joiner (emoji sequences)
    "]+"
)


@dataclass
class SpokenClip:
    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str


class PiperUnavailable(RuntimeError):
    pass


class Speaker:
    """Piper text-to-speech with sentence-level streaming playback."""

    def __init__(self, voice: Path | None = None, exe: Path | None = None) -> None:
        self.exe = exe or config.PIPER_EXE
        self.voice = voice or config.PIPER_VOICE

        if not self.exe.exists():
            raise PiperUnavailable(f"piper.exe not found at {self.exe}")
        if not self.voice.exists():
            raise PiperUnavailable(f"voice model not found at {self.voice}")

        # A truncated model makes piper exit 0 with no audio and no error. That
        # happened during setup and cost real debugging time; check it up front.
        size = self.voice.stat().st_size
        if size != config.PIPER_VOICE_BYTES:
            logger.warning(
                "voice model is {} bytes, expected {} - it may be truncated, "
                "which makes piper fail silently",
                size, config.PIPER_VOICE_BYTES,
            )

        self._play_q: queue.Queue[SpokenClip | None] = queue.Queue()
        self._player: threading.Thread | None = None
        self._stop = threading.Event()
        self.speaking = threading.Event()
        # Counts clips queued but not yet finished playing. A bare "is the queue
        # empty" check races: the player can have pulled the last clip and still
        # be mid-playback, so wait() would return while audio is running.
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        # Called with True when speech starts, False when it ends. Used to mute
        # the microphone and drive the orb.
        self.on_speaking_changed = None

    # ------------------------------------------------------------- synthesis
    def synthesize(self, text: str) -> SpokenClip | None:
        """Render text to audio. Returns None for empty or unusable text."""
        clean = redaction.redact(text)
        if not clean.is_clean:
            logger.warning("redacted before speaking aloud: {}", clean.summary())
        body = clean.text.strip()
        if not body:
            return None

        # Emoji cannot be spoken and, on Windows, piping one to a subprocess
        # without a forced encoding crashes outright - see _UNSPEAKABLE above.
        # This was a live bug: a persona built to use emoji in its replies
        # crashed AURA's voice the moment one appeared in a sentence.
        speakable = _UNSPEAKABLE.sub("", body).strip()
        speakable = re.sub(r"\s{2,}", " ", speakable)
        if not speakable:
            return None

        cfg = config.SETTINGS.voice
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clip.wav"
            proc = subprocess.run(
                [
                    str(self.exe),
                    "--model", str(self.voice),
                    "--output_file", str(out),
                    "--length_scale", str(cfg.length_scale),
                    "--noise_scale", str(cfg.noise_scale),
                    "--sentence_silence", str(cfg.sentence_silence),
                ],
                input=speakable,
                capture_output=True,
                text=True,
                encoding="utf-8",  # do not let this default to the console codepage
                errors="replace",
                timeout=120,
                check=False,
                cwd=str(self.exe.parent),  # piper needs its espeak-ng-data
            )
            if not out.exists() or out.stat().st_size < 128:
                logger.error(
                    "piper produced no audio (exit {}): {}",
                    proc.returncode, (proc.stderr or "").strip()[:200],
                )
                return None

            with wave.open(str(out), "rb") as wav:
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return SpokenClip(
            audio=audio, sample_rate=rate, duration=len(audio) / rate, text=body
        )

    def to_file(self, text: str, path: Path) -> Path | None:
        clip = self.synthesize(text)
        if clip is None:
            return None
        data = (clip.audio * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(clip.sample_rate)
            wav.writeframes(data.tobytes())
        return path

    # -------------------------------------------------------------- playback
    def _notify(self, speaking: bool) -> None:
        if speaking:
            self.speaking.set()
        else:
            self.speaking.clear()
        if self.on_speaking_changed:
            try:
                self.on_speaking_changed(speaking)
            except Exception:  # noqa: BLE001
                logger.exception("speaking callback failed")

    def _enqueue(self, clip: SpokenClip) -> None:
        """Hand one clip to the player thread and count it as outstanding."""
        with self._pending_lock:
            self._pending += 1
            self._idle.clear()
        self._play_q.put(clip)

    def _finished_one(self) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()
                self._notify(False)

    def _play_loop(self) -> None:
        import sounddevice as sd

        while not self._stop.is_set():
            clip = self._play_q.get()
            if clip is None:
                break
            self._notify(True)
            try:
                sd.play(clip.audio, clip.sample_rate)
                sd.wait()
            except Exception as exc:  # noqa: BLE001
                logger.error("playback failed: {}", exc)
            finally:
                self._finished_one()

    def _ensure_player(self) -> None:
        if self._player and self._player.is_alive():
            return
        self._stop.clear()
        self._player = threading.Thread(target=self._play_loop, name="aura-tts", daemon=True)
        self._player.start()

    def say(self, text: str, block: bool = True) -> None:
        """Speak text, one sentence at a time so the first word starts sooner."""
        sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
        if not sentences:
            return

        self._ensure_player()
        for sentence in sentences:
            clip = self.synthesize(sentence)
            if clip is not None:
                self._enqueue(clip)

        if block:
            self.wait()

    def say_stream(self, chunks: Iterator[str]) -> str:
        """Speak an LLM stream as sentences complete. Returns the full text."""
        self._ensure_player()
        buffer = ""
        spoken: list[str] = []

        for chunk in chunks:
            buffer += chunk
            while True:
                match = _SENTENCE.search(buffer)
                if not match:
                    break
                sentence, buffer = buffer[: match.start()].strip(), buffer[match.end() :]
                if sentence:
                    clip = self.synthesize(sentence)
                    if clip is not None:
                        self._enqueue(clip)
                    spoken.append(sentence)

        tail = buffer.strip()
        if tail:
            clip = self.synthesize(tail)
            if clip is not None:
                self._enqueue(clip)
            spoken.append(tail)

        self.wait()
        return " ".join(spoken)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has finished playing."""
        return self._idle.wait(timeout)

    def stop(self) -> None:
        """Cut playback immediately and drop anything still queued."""
        import sounddevice as sd

        self._stop.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        while not self._play_q.empty():
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                break
        with self._pending_lock:
            self._pending = 0
            self._idle.set()
        self._play_q.put(None)
        self._notify(False)


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA speech output (Phase 3)")
    parser.add_argument("text", nargs="*", default=["Aura voice output is working."])
    parser.add_argument("--out", type=Path, help="write to a WAV file instead of playing")
    args = parser.parse_args()

    bootstrap("tts")
    speaker = Speaker()
    text = " ".join(args.text)

    if args.out:
        result = speaker.to_file(text, args.out)
        print(f"wrote {result}" if result else "synthesis failed")
        return 0 if result else 1

    print(f"speaking: {text}")
    speaker.say(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
