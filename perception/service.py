"""
AURA — perception service (Phase 2 deliverable).

Runs face recognition and speech recognition in the background and reports, at
any moment, exactly who is present by name, what was last said, and what is on
screen.

Presence is debounced. A face that vanishes for a frame - someone turns their
head, or blinks at the wrong moment - must not read as "they left the room",
because Phase 4 switches access tiers off this signal and a flickering tier would
be both annoying and unsafe. A person stays "present" until unseen for
`absent_after` seconds.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from aura import config
from aura.perception import screen
from aura.perception.faces import UNKNOWN, Detection, FaceRecognizer
from aura.perception.speech import SpeechListener, Utterance
from aura.safety import audit


@dataclass
class Presence:
    """Someone currently considered present."""

    name: str
    relation: str
    first_seen: float
    last_seen: float
    similarity: float

    @property
    def is_primary(self) -> bool:
        return self.name == config.SETTINGS.primary_user

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN

    def seconds_present(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class PerceptionState:
    """Immutable snapshot of what AURA can currently perceive."""

    people: list[Presence] = field(default_factory=list)
    last_utterance: Utterance | None = None
    camera_ok: bool = False
    timestamp: str = ""

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.people]

    @property
    def primary_present(self) -> bool:
        return any(p.is_primary for p in self.people)

    @property
    def others_present(self) -> list[Presence]:
        return [p for p in self.people if not p.is_primary]

    @property
    def unknown_present(self) -> bool:
        return any(not p.is_known for p in self.people)

    def describe(self) -> str:
        if not self.people:
            return "nobody visible"
        parts = []
        for p in self.people:
            label = p.name if p.is_known else "someone I don't recognise"
            parts.append(f"{label}{f' ({p.relation})' if p.relation else ''}")
        return ", ".join(parts)


class PerceptionService:
    """Background vision + hearing."""

    def __init__(
        self,
        on_utterance: Callable[[Utterance], None] | None = None,
        on_presence_change: Callable[[PerceptionState], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        enable_camera: bool = True,
        enable_mic: bool = True,
        barge_in: bool = False,
    ) -> None:
        self.on_utterance = on_utterance
        self.on_presence_change = on_presence_change
        self.on_barge_in = on_barge_in
        self.enable_camera = enable_camera
        self.enable_mic = enable_mic
        self.barge_in = barge_in

        self._present: dict[str, Presence] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._vision_thread: threading.Thread | None = None
        self._recognizer: FaceRecognizer | None = None
        self._listener: SpeechListener | None = None
        self._last_utterance: Utterance | None = None
        self._camera_ok = False

    # ----------------------------------------------------------------- vision
    def _vision_loop(self) -> None:
        import cv2

        from aura.perception.faces import open_camera

        cfg = config.SETTINGS.vision
        try:
            self._recognizer = FaceRecognizer()
            cap = open_camera()
            self._camera_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.error("vision unavailable: {}", exc)
            self._camera_ok = False
            return

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.2)
                    continue

                try:
                    detections = self._recognizer.detect(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.error("detection failed: {}", exc)
                    detections = []

                self._update_presence(detections)
                self._stop.wait(cfg.frame_interval)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _update_presence(self, detections: list[Detection]) -> None:
        now = time.time()
        cfg = config.SETTINGS.vision
        changed = False

        with self._lock:
            for d in detections:
                # Unknown faces share one slot; AURA cannot tell two strangers
                # apart, and pretending otherwise would be a false claim.
                key = d.name
                existing = self._present.get(key)
                if existing is None:
                    self._present[key] = Presence(
                        name=d.name, relation=d.relation,
                        first_seen=now, last_seen=now, similarity=d.similarity,
                    )
                    changed = True
                    audit.record(
                        audit.Event.PERSON_SEEN,
                        detail={"name": d.name, "similarity": round(d.similarity, 3)},
                    )
                    logger.info("present: {} ({:.2f})", d.name, d.similarity)
                else:
                    existing.last_seen = now
                    existing.similarity = d.similarity

            # Debounced departure.
            gone = [
                name for name, p in self._present.items()
                if now - p.last_seen > cfg.absent_after
            ]
            for name in gone:
                del self._present[name]
                changed = True
                logger.info("left: {}", name)

        if changed and self.on_presence_change:
            try:
                self.on_presence_change(self.state())
            except Exception:  # noqa: BLE001
                logger.exception("presence callback failed")

    # ------------------------------------------------------------------ audio
    def _handle_utterance(self, utterance: Utterance) -> None:
        self._last_utterance = utterance
        if self.on_utterance:
            self.on_utterance(utterance)

    # -------------------------------------------------------------------- api
    def start(self) -> None:
        if self.enable_camera:
            self._stop.clear()
            self._vision_thread = threading.Thread(
                target=self._vision_loop, name="aura-vision", daemon=True
            )
            self._vision_thread.start()

        if self.enable_mic:
            self._listener = SpeechListener(
                on_utterance=self._handle_utterance,
                on_barge_in=self.on_barge_in,
                barge_in=self.barge_in,
            )
            self._listener.start()

        audit.record(audit.Event.STARTUP, detail={"service": "perception"})

    def stop(self) -> None:
        self._stop.set()
        if self._listener:
            self._listener.stop()
        if self._vision_thread:
            self._vision_thread.join(timeout=5)

    def state(self) -> PerceptionState:
        with self._lock:
            people = sorted(
                self._present.values(),
                key=lambda p: (not p.is_primary, p.name),
            )
            people = [
                Presence(p.name, p.relation, p.first_seen, p.last_seen, p.similarity)
                for p in people
            ]
        return PerceptionState(
            people=people,
            last_utterance=self._last_utterance,
            camera_ok=self._camera_ok,
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    def mute_mic(self, muted: bool) -> None:
        """Silence input while AURA speaks, so it never hears itself."""
        if self._listener:
            if muted:
                self._listener.muted.set()
            else:
                self._listener.muted.clear()

    def screen_context(self, include_text: bool = True) -> screen.ScreenContext:
        return screen.read_screen(include_text=include_text)

    def enroll(self, name: str, relation: str = "") -> bool:
        """Enrol whoever is in front of the camera right now (Phase 4 flow)."""
        from aura.perception.faces import open_camera

        if self._recognizer is None:
            logger.error("cannot enrol - vision is not running")
            return False
        cap = open_camera()
        try:
            captured = 0
            for _ in range(12):
                ok, frame = cap.read()
                if ok and self._recognizer.enroll_from_frame(frame, name, relation):
                    captured += 1
                    if captured >= 3:
                        break
                time.sleep(0.4)
            return captured > 0
        finally:
            cap.release()


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA perception service (Phase 2)")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-mic", action="store_true")
    parser.add_argument("--seconds", type=int, default=0, help="run for N seconds then exit")
    args = parser.parse_args()

    bootstrap("perception")
    service = PerceptionService(
        enable_camera=not args.no_camera, enable_mic=not args.no_mic
    )
    service.start()

    print("perception running - ctrl-c to stop\n")
    started = time.time()
    try:
        while True:
            state = service.state()
            heard = state.last_utterance.text if state.last_utterance else "-"
            print(f"[{state.timestamp}] present: {state.describe():<45} last heard: {heard[:50]}")
            time.sleep(2)
            if args.seconds and time.time() - started > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()

    context = service.screen_context()
    print("\n--- screen context ---")
    print(context.summary(300))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
