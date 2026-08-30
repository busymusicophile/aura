"""
AURA — face recognition (Phase 2).

Recognises multiple distinct people in frame simultaneously, by name, using
InsightFace (SCRFD detection + ArcFace 512-d embeddings) on the GPU.

Threshold choice matters more than model choice here. A false "that is
Keerthana" hands her full access tier to someone else, which is a privacy
breach. A false "unknown" only costs a re-greeting. The default cosine threshold
of 0.38 is therefore deliberately conservative - it prefers saying "I don't know
you" over guessing.

The face database is a single JSON file: readable, inspectable, easy to delete a
person from by hand, and straightforward to encrypt at rest in Phase 12.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from aura import config
from aura.safety import audit

UNKNOWN = "unknown"


@dataclass
class Detection:
    """One face seen in one frame."""

    name: str
    relation: str
    similarity: float
    bbox: tuple[int, int, int, int]
    is_known: bool

    @property
    def is_primary(self) -> bool:
        return self.name == config.SETTINGS.primary_user


@dataclass
class Enrolment:
    name: str
    relation: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    enrolled_at: str = ""

    def mean_embedding(self) -> np.ndarray:
        stacked = np.stack(self.embeddings)
        mean = stacked.mean(axis=0)
        return mean / np.linalg.norm(mean)


class FaceDatabase:
    """Known people and their face embeddings."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.FACES_DIR / "faces.json")
        self.people: dict[str, Enrolment] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            logger.debug("no face database yet at {}", self.path)
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("face database corrupt ({}) - starting empty", exc)
            return
        for name, entry in raw.items():
            self.people[name] = Enrolment(
                name=name,
                relation=entry.get("relation", ""),
                embeddings=[np.asarray(e, dtype=np.float32) for e in entry.get("embeddings", [])],
                enrolled_at=entry.get("enrolled_at", ""),
            )
        logger.info("loaded {} known face(s): {}", len(self.people), ", ".join(self.people))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: {
                "relation": p.relation,
                "enrolled_at": p.enrolled_at,
                "embeddings": [e.tolist() for e in p.embeddings],
            }
            for name, p in self.people.items()
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, name: str, relation: str, embedding: np.ndarray) -> None:
        person = self.people.get(name)
        if person is None:
            person = Enrolment(
                name=name,
                relation=relation,
                enrolled_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            self.people[name] = person
        if relation:
            person.relation = relation
        # Several samples per person handles lighting and angle far better than one.
        person.embeddings.append(embedding.astype(np.float32))
        person.embeddings = person.embeddings[-10:]
        self.save()

    def remove(self, name: str) -> bool:
        if name in self.people:
            del self.people[name]
            self.save()
            return True
        return False

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str, str, float]:
        """Nearest known identity, or unknown."""
        if not self.people:
            return UNKNOWN, "", 0.0

        query = embedding / np.linalg.norm(embedding)
        best_name, best_relation, best_score = UNKNOWN, "", 0.0

        for person in self.people.values():
            # Compare against every stored sample, keep the strongest match.
            for stored in person.embeddings:
                stored_norm = stored / np.linalg.norm(stored)
                score = float(np.dot(query, stored_norm))
                if score > best_score:
                    best_name, best_relation, best_score = person.name, person.relation, score

        if best_score < threshold:
            return UNKNOWN, "", best_score
        return best_name, best_relation, best_score

    def names(self) -> list[str]:
        return sorted(self.people)


class FaceRecognizer:
    """InsightFace wrapper. Requires `aura.runtime.bootstrap()` first."""

    def __init__(self, db: FaceDatabase | None = None) -> None:
        from insightface.app import FaceAnalysis

        cfg = config.SETTINGS.vision
        self.db = db or FaceDatabase()
        self.threshold = cfg.match_threshold

        started = time.perf_counter()
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=cfg.det_size)

        bound = [
            m.session.get_providers()[0]
            for m in self._app.models.values()
            if getattr(m, "session", None) is not None
        ]
        on_gpu = sum(1 for p in bound if "CUDA" in p or "Tensorrt" in p)
        if on_gpu == 0:
            logger.warning(
                "face models are running on CPU - recognition will be slow. "
                "Check that runtime.bootstrap() ran and onnxruntime-gpu is 1.22.0"
            )
        logger.info(
            "face recognizer ready in {:.1f}s ({}/{} models on GPU)",
            time.perf_counter() - started, on_gpu, len(bound),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Every face in the frame, identified where possible."""
        results: list[Detection] = []
        for face in self._app.get(frame):
            name, relation, score = self.db.match(face.normed_embedding, self.threshold)
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            results.append(
                Detection(
                    name=name,
                    relation=relation,
                    similarity=score,
                    bbox=(x1, y1, x2, y2),
                    is_known=name != UNKNOWN,
                )
            )
        # Largest face first - usually the person actually addressing AURA.
        results.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return results

    def enroll_from_frame(self, frame: np.ndarray, name: str, relation: str = "") -> bool:
        """Register the largest face in frame as `name`."""
        faces = self._app.get(frame)
        if not faces:
            logger.warning("enrolment failed: no face detected")
            return False
        if len(faces) > 1:
            logger.warning("{} faces in frame - enrolling the largest", len(faces))

        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        self.db.add(name, relation, largest.normed_embedding)
        audit.record(
            audit.Event.PERSON_ENROLLED,
            detail={"name": name, "relation": relation,
                    "samples": len(self.db.people[name].embeddings)},
        )
        logger.info(
            "enrolled {} ({}) - {} sample(s) stored",
            name, relation or "no relation", len(self.db.people[name].embeddings),
        )
        return True


def open_camera(index: int | None = None):
    """Open the webcam, trying each Windows backend in turn.

    DSHOW is preferred (it starts fastest and is the most forgiving), but it
    intermittently refuses to open a camera that MSMF can still reach - and vice
    versa - depending on what else has touched the device. Opening is also not
    enough on its own: a handle can open successfully and then deliver no
    frames when another application holds the camera, so each backend is
    verified by actually reading a frame before it is accepted.
    """
    import cv2

    idx = config.SETTINGS.vision.camera_index if index is None else index
    attempts = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "default"),
    ]

    failures: list[str] = []
    for backend, name in attempts:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            failures.append(f"{name}: would not open")
            continue
        ok, _ = cap.read()
        if ok:
            logger.debug("camera {} opened via {}", idx, name)
            return cap
        cap.release()
        failures.append(f"{name}: opened but delivered no frames")

    raise RuntimeError(
        f"could not capture from camera {idx} ({'; '.join(failures)}). "
        "Usually this means another application is holding the camera, or "
        "Windows camera privacy is switched off for desktop apps."
    )


def enroll_interactive(name: str, relation: str = "", samples: int = 5) -> bool:
    """Capture several frames from the webcam and enrol them as one person."""
    import cv2

    recognizer = FaceRecognizer()
    cap = open_camera()
    captured = 0
    logger.info("look at the camera - capturing {} samples", samples)

    try:
        while captured < samples:
            ok, frame = cap.read()
            if not ok:
                continue
            if recognizer.enroll_from_frame(frame, name, relation):
                captured += 1
                logger.info("sample {}/{}", captured, samples)
                time.sleep(0.6)  # spacing gives varied angles
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured > 0


def main() -> int:
    import argparse

    import cv2

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA face tools")
    parser.add_argument("--enroll", metavar="NAME", help="enrol a person")
    parser.add_argument("--relation", default="", help="relation, e.g. 'mother'")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--list", action="store_true", help="list known people")
    parser.add_argument("--forget", metavar="NAME", help="remove a person")
    parser.add_argument("--watch", action="store_true", help="live recognition preview")
    args = parser.parse_args()

    bootstrap("faces")

    if args.list:
        db = FaceDatabase()
        if not db.people:
            print("no one enrolled yet")
        for name, person in db.people.items():
            print(f"{name:<20} {person.relation:<16} "
                  f"{len(person.embeddings)} sample(s)  since {person.enrolled_at}")
        return 0

    if args.forget:
        db = FaceDatabase()
        print("removed" if db.remove(args.forget) else "not found")
        return 0

    if args.enroll:
        return 0 if enroll_interactive(args.enroll, args.relation, args.samples) else 1

    if args.watch:
        recognizer = FaceRecognizer()
        cap = open_camera()
        print("press q to quit")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                for d in recognizer.detect(frame):
                    x1, y1, x2, y2 = d.bbox
                    colour = (0, 200, 0) if d.is_known else (0, 140, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(
                        frame, f"{d.name} {d.similarity:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                    )
                cv2.imshow("AURA perception", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
