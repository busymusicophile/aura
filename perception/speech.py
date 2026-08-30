"""
AURA — speech to text (Phase 2).

Continuous microphone capture, gated by voice-activity detection, transcribed
locally with faster-whisper on the GPU.

The VAD gate is the whole point. Running Whisper continuously would keep the GPU
busy transcribing silence and room noise, and would produce a stream of
hallucinated text from nothing - Whisper is notorious for inventing speech in
quiet audio. Silero VAD is tiny and runs on CPU, so it decides cheaply when
something was actually said and only then wakes Whisper.

Language is left on autodetect because she mixes English, Hindi and Telugu; qwen3
handles the mixed transcript downstream.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from loguru import logger

from aura import config

# Silero VAD operates on fixed 512-sample frames at 16 kHz (32 ms).
VAD_FRAME = 512


@dataclass
class Utterance:
    text: str
    duration: float
    language: str
    confidence: float
    timestamp: str


class VoiceActivityGate:
    """Decides when speech starts and stops."""

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        import torch

        cfg = config.SETTINGS.speech
        self._torch = torch
        self._model = load_silero_vad()
        self.threshold = cfg.vad_threshold
        self.silence_frames = int(cfg.silence_ms / 32)
        self.min_speech_frames = int(cfg.min_speech_ms / 32)

        self._speaking = False
        self._silent_run = 0
        self._speech_run = 0

    def reset(self) -> None:
        self._model.reset_states()
        self._speaking = False
        self._silent_run = 0
        self._speech_run = 0

    def push(self, frame: np.ndarray) -> str:
        """Feed one 512-sample frame. Returns 'start', 'end', or ''."""
        tensor = self._torch.from_numpy(frame)
        with self._torch.no_grad():
            prob = float(self._model(tensor, config.SETTINGS.speech.sample_rate).item())

        is_speech = prob >= self.threshold

        if is_speech:
            self._speech_run += 1
            self._silent_run = 0
            if not self._speaking and self._speech_run >= self.min_speech_frames:
                self._speaking = True
                return "start"
        else:
            self._silent_run += 1
            if self._speaking and self._silent_run >= self.silence_frames:
                self._speaking = False
                self._speech_run = 0
                return "end"
            if not self._speaking:
                self._speech_run = 0
        return ""

    @property
    def speaking(self) -> bool:
        return self._speaking


class Transcriber:
    """faster-whisper on the GPU."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        cfg = config.SETTINGS.speech
        started = time.perf_counter()
        try:
            self._model = WhisperModel(
                cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute
            )
            self.device = cfg.whisper_device
        except Exception as exc:  # noqa: BLE001
            # The 6GB card may already be full of qwen3; CPU is slow but correct.
            logger.warning("whisper failed on {} ({}) - falling back to CPU", cfg.whisper_device, exc)
            self._model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")
            self.device = "cpu"
        logger.info(
            "whisper '{}' ready on {} in {:.1f}s",
            cfg.whisper_model, self.device, time.perf_counter() - started,
        )

    def transcribe(self, audio: np.ndarray) -> Utterance | None:
        cfg = config.SETTINGS.speech
        duration = len(audio) / cfg.sample_rate
        if duration < 0.3:
            return None

        segments, info = self._model.transcribe(
            audio,
            language=cfg.language,
            beam_size=5,
            vad_filter=False,  # already gated upstream
            condition_on_previous_text=False,  # stops runaway repetition
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return None

        return Utterance(
            text=text,
            duration=duration,
            language=getattr(info, "language", "") or "",
            confidence=float(getattr(info, "language_probability", 0.0) or 0.0),
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        )


class SpeechListener:
    """Background microphone listener that emits complete utterances."""

    def __init__(
        self,
        on_utterance: Callable[[Utterance], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        barge_in: bool = False,
    ) -> None:
        self.on_utterance = on_utterance
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._utterances: queue.Queue[Utterance] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gate: VoiceActivityGate | None = None
        self._transcriber: Transcriber | None = None
        self.is_speaking = False
        # Set while AURA is talking, so it does not transcribe its own voice.
        self.muted = threading.Event()

        # Barge-in: keep listening while AURA speaks so she can interrupt.
        #
        # Off by default, and that is not timidity. Without acoustic echo
        # cancellation the microphone hears AURA's own voice through the laptop
        # speakers, so a live gate during playback will interrupt itself mid
        # sentence, every sentence. The raised threshold below makes that less
        # likely but cannot rule it out - it is a volume heuristic, not echo
        # cancellation. Enable it with headphones, or once an AEC path exists.
        self.barge_in_enabled = barge_in
        self.on_barge_in = on_barge_in
        self._barge_gate: VoiceActivityGate | None = None
        self.barge_in_threshold = 0.92

    # ------------------------------------------------------------------ loop
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("audio status: {}", status)
        frame = indata[:, 0].copy()

        if not self.muted.is_set():
            self._audio_q.put(frame)
        elif self.barge_in_enabled:
            self._check_barge_in(frame)

    def _check_barge_in(self, frame: np.ndarray) -> None:
        """Watch for her talking over AURA, at a deliberately high threshold."""
        if self._barge_gate is None or self.on_barge_in is None:
            return
        if len(frame) != VAD_FRAME:
            return
        try:
            if self._barge_gate.push(frame) == "start":
                logger.info("barge-in detected - stopping playback")
                self.on_barge_in()
        except Exception:  # noqa: BLE001
            logger.exception("barge-in check failed")

    def _run(self) -> None:
        import sounddevice as sd

        cfg = config.SETTINGS.speech
        self._gate = VoiceActivityGate()
        self._transcriber = Transcriber()

        if self.barge_in_enabled:
            # A separate gate, so interrupting does not disturb the state of the
            # gate used for ordinary transcription.
            self._barge_gate = VoiceActivityGate()
            self._barge_gate.threshold = self.barge_in_threshold
            logger.info("barge-in enabled (threshold {:.2f})", self.barge_in_threshold)

        buffer = np.zeros(0, dtype=np.float32)
        speech: list[np.ndarray] = []
        # Keep a little audio from before the trigger; the first syllable is
        # otherwise clipped, which changes what the model hears.
        preroll: list[np.ndarray] = []
        preroll_frames = 8

        with sd.InputStream(
            samplerate=cfg.sample_rate, channels=1, dtype="float32",
            blocksize=VAD_FRAME, callback=self._callback,
        ):
            logger.info("listening (VAD-gated)")
            while not self._stop.is_set():
                try:
                    chunk = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])
                while len(buffer) >= VAD_FRAME:
                    frame, buffer = buffer[:VAD_FRAME], buffer[VAD_FRAME:]
                    event = self._gate.push(frame)

                    if self._gate.speaking:
                        speech.append(frame)
                    else:
                        preroll.append(frame)
                        if len(preroll) > preroll_frames:
                            preroll.pop(0)

                    if event == "start":
                        self.is_speaking = True
                        speech = list(preroll) + speech
                        preroll.clear()
                    elif event == "end":
                        self.is_speaking = False
                        audio = np.concatenate(speech) if speech else np.zeros(0, np.float32)
                        speech = []
                        if len(audio):
                            self._handle(audio)

    def _handle(self, audio: np.ndarray) -> None:
        assert self._transcriber is not None
        try:
            utterance = self._transcriber.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            logger.error("transcription failed: {}", exc)
            return
        if utterance is None:
            return
        logger.info("heard [{}]: {}", utterance.language, utterance.text)
        self._utterances.put(utterance)
        if self.on_utterance:
            try:
                self.on_utterance(utterance)
            except Exception:  # noqa: BLE001
                logger.exception("utterance callback failed")

    # -------------------------------------------------------------------- api
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aura-speech", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def next(self, timeout: float | None = None) -> Utterance | None:
        try:
            return self._utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Utterance]:
        out = []
        while not self._utterances.empty():
            out.append(self._utterances.get())
        return out


def main() -> int:
    from aura.runtime import bootstrap

    bootstrap("speech")
    listener = SpeechListener()
    listener.start()
    print("speak into the mic; ctrl-c to stop")
    try:
        while True:
            utterance = listener.next(timeout=1.0)
            if utterance:
                print(f"[{utterance.duration:.1f}s {utterance.language}] {utterance.text}")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
