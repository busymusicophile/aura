"""
AURA — hardware registry and device model (Phase 9 mock, Phase 10 real).

Defines what a controllable device is, and a `DeviceBackend` interface with two
implementations: a mock used to build and test the control panel before any
hardware exists, and the real Home Assistant client in `aura.home.hass`.

Building the panel against a mock first is deliberate. It means the UI is fully
exercised - including error and unavailable states, which are awkward to produce
on demand with real hardware - before a single device is bought. When the Pi
arrives, only the backend changes.

The registry is a JSON file listing every device and the room it is in. Room
grouping is what makes "turn off the lights in the study" resolvable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from aura import config

REGISTRY_FILE = config.DATA_DIR / "home" / "devices.json"


class DeviceKind(str, Enum):
    LIGHT = "light"
    SWITCH = "switch"
    FAN = "fan"
    AC = "ac"
    TV = "tv"
    PROJECTOR = "projector"
    SPEAKER = "speaker"
    SENSOR = "sensor"
    LOCK = "lock"
    CAMERA = "camera"
    OTHER = "other"


# Anything that projects, plays or displays to the room needs a preview first
# (design principle 3). A light does not - it is instantly obvious and instantly
# reversible.
PREVIEW_REQUIRED = {DeviceKind.PROJECTOR, DeviceKind.TV, DeviceKind.SPEAKER}


@dataclass
class Device:
    id: str
    name: str
    kind: DeviceKind
    room: str
    entity_id: str = ""          # Home Assistant entity, when real
    state: str = "unknown"       # on | off | unavailable | unknown
    attributes: dict[str, Any] = field(default_factory=dict)
    reachable: bool = True
    last_changed: str = ""

    @property
    def needs_preview(self) -> bool:
        return self.kind in PREVIEW_REQUIRED

    @property
    def is_on(self) -> bool:
        return self.state == "on"

    def describe(self) -> str:
        status = self.state if self.reachable else "unreachable"
        return f"{self.name} ({self.room}): {status}"


class DeviceBackend(Protocol):
    """What the control panel and action layer need from a device source."""

    def devices(self) -> list[Device]: ...
    def get(self, device_id: str) -> Device | None: ...
    def set_state(self, device_id: str, state: str) -> Device: ...
    def available(self) -> bool: ...
    def describe_backend(self) -> str: ...


# --------------------------------------------------------------------------
# Registry file
# --------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> list[Device]:
    target = path or REGISTRY_FILE
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("device registry is corrupt ({}) - treating as empty", exc)
        return []
    devices = []
    for entry in raw.get("devices", []):
        try:
            devices.append(
                Device(
                    id=entry["id"],
                    name=entry["name"],
                    kind=DeviceKind(entry.get("kind", "other")),
                    room=entry.get("room", "unassigned"),
                    entity_id=entry.get("entity_id", ""),
                    state=entry.get("state", "unknown"),
                    attributes=entry.get("attributes", {}),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("skipping malformed device entry: {}", exc)
    return devices


def save_registry(devices: list[Device], path: Path | None = None) -> Path:
    target = path or REGISTRY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "devices": [
            {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(d).items()}
            for d in devices
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def by_room(devices: list[Device]) -> dict[str, list[Device]]:
    grouped: dict[str, list[Device]] = {}
    for device in devices:
        grouped.setdefault(device.room, []).append(device)
    for room in grouped.values():
        room.sort(key=lambda d: d.name)
    return dict(sorted(grouped.items()))


# --------------------------------------------------------------------------
# Mock backend — Phase 9
# --------------------------------------------------------------------------

_MOCK_LAYOUT = [
    ("bedroom_light", "Ceiling light", DeviceKind.LIGHT, "bedroom"),
    ("bedroom_fan", "Fan", DeviceKind.FAN, "bedroom"),
    ("bedroom_ac", "Air conditioner", DeviceKind.AC, "bedroom"),
    ("study_light", "Desk lamp", DeviceKind.LIGHT, "study"),
    # A second light in one room on purpose: real rooms have several, and it is
    # the case that exercises multi-device commands ("lights off in the study").
    ("study_shelf_light", "Shelf light", DeviceKind.LIGHT, "study"),
    ("study_projector", "Projector", DeviceKind.PROJECTOR, "study"),
    ("study_speaker", "Speaker", DeviceKind.SPEAKER, "study"),
    ("living_light", "Main light", DeviceKind.LIGHT, "living room"),
    ("living_tv", "Television", DeviceKind.TV, "living room"),
    ("living_fan", "Fan", DeviceKind.FAN, "living room"),
    ("kitchen_light", "Ceiling light", DeviceKind.LIGHT, "kitchen"),
    ("front_lock", "Front door", DeviceKind.LOCK, "entrance"),
    ("hall_motion", "Motion sensor", DeviceKind.SENSOR, "hall"),
]


class MockBackend:
    """Fake devices so the panel can be built and tested with no hardware.

    One device is deliberately left unreachable. The unavailable state is the
    one most likely to be handled badly, and the easiest to forget to design for
    if every mock device always answers.
    """

    def __init__(self, seed: int = 7) -> None:
        rng = random.Random(seed)
        self._devices: dict[str, Device] = {}
        for device_id, name, kind, room in _MOCK_LAYOUT:
            state = "off" if kind != DeviceKind.SENSOR else "idle"
            if kind in (DeviceKind.LIGHT, DeviceKind.FAN) and rng.random() > 0.6:
                state = "on"
            self._devices[device_id] = Device(
                id=device_id, name=name, kind=kind, room=room,
                entity_id=f"{kind.value}.{device_id}", state=state,
                last_changed=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        self._devices["living_tv"].reachable = False
        self._devices["living_tv"].state = "unavailable"

    def devices(self) -> list[Device]:
        return list(self._devices.values())

    def get(self, device_id: str) -> Device | None:
        return self._devices.get(device_id)

    def set_state(self, device_id: str, state: str) -> Device:
        device = self._devices.get(device_id)
        if device is None:
            raise KeyError(f"no device {device_id}")
        if not device.reachable:
            raise RuntimeError(f"{device.name} is unreachable")
        device.state = state
        device.last_changed = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("[mock] {} -> {}", device.name, state)
        return device

    def available(self) -> bool:
        return True

    def describe_backend(self) -> str:
        return "mock (no hardware — Phase 9 placeholder data)"


def get_backend(prefer_real: bool = True) -> DeviceBackend:
    """Return the Home Assistant backend if configured, else the mock."""
    if prefer_real:
        try:
            from aura.home.hass import HomeAssistantBackend

            backend = HomeAssistantBackend()
            if backend.available():
                return backend
            logger.info("Home Assistant not reachable - using mock devices")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Home Assistant backend unavailable ({}) - using mock", exc)
    return MockBackend()
