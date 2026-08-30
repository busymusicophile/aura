"""
AURA — home control and the preview-confirm-act flow (Phase 10).

Resolves spoken device references ("the lamp in the study", "lights off in the
bedroom") and routes every physical change through the Phase 5 action broker.

Design principle 3, made concrete
---------------------------------
Anything projected, played or displayed to the room is shown as a small preview
first and committed to hardware only after confirmation. A simple on/off toggle
does not need a preview - a light is instantly obvious and instantly reversible,
so a preview would be pure friction.

The asymmetry is about how visible and how undoable a mistake is. Switching a
lamp on in the wrong room is noticed and fixed in two seconds. Throwing the wrong
thing onto a projector in a room with other people in it cannot be un-seen. So
projectors, televisions and speakers preview locally on her laptop first; lights,
fans and switches confirm and go.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

from loguru import logger

from aura.actions.base import Action, ActionBroker, ActionKind
from aura.home.registry import Device, DeviceKind, by_room, get_backend

_ON_WORDS = {"on", "up", "start", "open", "enable", "switch on", "turn on"}
_OFF_WORDS = {"off", "down", "stop", "close", "disable", "switch off", "turn off"}

# Spoken synonyms for device kinds, so "lamp" finds a light.
_KIND_WORDS = {
    DeviceKind.LIGHT: {"light", "lights", "lamp", "lamps", "bulb"},
    DeviceKind.FAN: {"fan", "fans"},
    DeviceKind.AC: {"ac", "air conditioner", "aircon", "cooling"},
    DeviceKind.TV: {"tv", "television", "telly"},
    DeviceKind.PROJECTOR: {"projector", "beamer"},
    DeviceKind.SPEAKER: {"speaker", "speakers", "music", "audio"},
    DeviceKind.LOCK: {"lock", "door"},
    DeviceKind.SWITCH: {"switch", "plug", "socket"},
}


@dataclass
class Resolution:
    """What a spoken phrase resolved to."""

    devices: list[Device]
    room: str = ""
    state: str = ""
    ambiguous: bool = False
    note: str = ""


@dataclass
class Preview:
    """What would happen, shown before it happens."""

    description: str
    local_render: str = ""   # where a local preview artefact was written
    committed: bool = False


def parse_state(text: str) -> str:
    """Extract on/off from a spoken phrase."""
    lowered = f" {text.lower()} "
    for word in _OFF_WORDS:
        if f" {word} " in lowered:
            return "off"
    for word in _ON_WORDS:
        if f" {word} " in lowered:
            return "on"
    return ""


def resolve(text: str, devices: list[Device]) -> Resolution:
    """Turn a spoken phrase into the devices it refers to."""
    lowered = text.lower()
    state = parse_state(text)

    rooms = {d.room.lower() for d in devices}
    room = next((r for r in sorted(rooms, key=len, reverse=True) if r and r in lowered), "")

    kind = None
    for candidate, words in _KIND_WORDS.items():
        if any(f"{w}" in lowered for w in words):
            kind = candidate
            break

    pool = devices
    if room:
        pool = [d for d in pool if d.room.lower() == room]
    if kind is not None:
        pool = [d for d in pool if d.kind == kind]

    # Room + kind, e.g. "lights off in the study" - act on all of them.
    if room and kind is not None and pool:
        return Resolution(devices=pool, room=room, state=state)

    # A named room is a hard constraint, never a hint. If she said "the
    # projector in the kitchen" and there is no projector there, the answer is
    # "there isn't one" - widening the search to the whole house would find the
    # study projector and switch on a device in a different room from the one
    # she named, which is exactly the kind of wrong that is hard to forgive.
    if room and not pool:
        in_room = [d for d in devices if d.room.lower() == room]
        if kind is not None:
            return Resolution(
                devices=[], room=room, state=state,
                note=f"there is no {kind.value} in the {room}"
                     + (f" (it has: {', '.join(d.name for d in in_room)})" if in_room else ""),
            )
        return Resolution(
            devices=[], room=room, state=state,
            note=f"nothing matching that in the {room}",
        )

    # Otherwise try to match a specific device by name.
    #
    # Note this must NOT key devices by name in a dict: several rooms have a
    # device called "Fan", and a dict would silently collapse them, so "turn on
    # the fan" would pick one arbitrarily instead of asking which. Duplicates
    # are exactly the case that needs detecting, so matches stay a list.
    if not pool:
        pool = devices

    named = [d for d in pool if d.name.lower() in lowered]
    if not named:
        # Fall back to fuzzy matching, but compare against the device name only
        # and require a high similarity. A loose cutoff against the whole phrase
        # matches almost anything - "do the thing" scored against "Television".
        scored: list[tuple[float, Device]] = []
        for device in pool:
            ratio = difflib.SequenceMatcher(None, device.name.lower(), lowered).ratio()
            best_word = max(
                (
                    difflib.SequenceMatcher(None, device.name.lower(), word).ratio()
                    for word in lowered.split()
                ),
                default=0.0,
            )
            score = max(ratio, best_word)
            if score >= 0.75:
                scored.append((score, device))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        named = [device for _, device in scored]

    if len(named) == 1:
        return Resolution(devices=named, room=room, state=state)
    if len(named) > 1:
        return Resolution(
            devices=named, room=room, state=state, ambiguous=True,
            note="more than one device matches",
        )
    if kind is not None and pool:
        return Resolution(devices=pool, room=room, state=state,
                          ambiguous=len(pool) > 1)

    return Resolution(devices=[], room=room, state=state, note="nothing matched")


class HomeController:
    """Proposes device changes, with a preview where principle 3 requires one."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend or get_backend()

    # ---------------------------------------------------------------- reading
    def inventory(self) -> dict[str, list[Device]]:
        return by_room(self.backend.devices())

    def describe_house(self) -> str:
        """Everything AURA knows about, by room. Reading changes nothing."""
        lines: list[str] = []
        for room, devices in self.inventory().items():
            lines.append(f"{room}:")
            for device in devices:
                lines.append(f"  {device.describe()}")
        return "\n".join(lines) if lines else "no devices known"

    # --------------------------------------------------------------- previews
    def build_preview(self, device: Device, state: str) -> Preview:
        """Render what would happen, without touching the hardware."""
        if not device.needs_preview:
            return Preview(description=f"{device.name} in the {device.room} -> {state}")

        if device.kind == DeviceKind.SPEAKER:
            return Preview(
                description=(
                    f"Would play through {device.name} in the {device.room}.\n"
                    "Preview plays on the laptop speakers first, at low volume."
                ),
            )
        if device.kind in (DeviceKind.PROJECTOR, DeviceKind.TV):
            return Preview(
                description=(
                    f"Would put this on {device.name} in the {device.room}, "
                    "visible to everyone in that room.\n"
                    "Preview appears on the laptop screen first."
                ),
            )
        return Preview(description=f"{device.name} -> {state}")

    # -------------------------------------------------------------- proposals
    def propose(
        self,
        device: Device,
        state: str,
        broker: ActionBroker,
        perception_state: Any = None,
    ) -> Action:
        preview = self.build_preview(device, state)
        return broker.propose(
            Action(
                kind=ActionKind.DEVICE,
                summary=f"Turn {state} {device.name} in the {device.room}",
                preview=preview.description,
                run=lambda: self.backend.set_state(device.id, state),
                needs_preview=device.needs_preview,
                # Physical toggles are reversible; a lock is not treated as one,
                # since unlocking a door is not something to undo casually.
                reversible=device.kind != DeviceKind.LOCK,
            ),
            perception_state,
        )

    def propose_batch(
        self,
        devices: list[Device],
        state: str,
        broker: ActionBroker,
        perception_state: Any = None,
        room: str = "",
    ) -> Action:
        """One confirmation covering several devices.

        "Lights off in the study" is a single intention and must not become
        three separate questions. Each device is still switched individually, so
        one unreachable bulb does not abandon the rest.
        """
        from aura.actions.base import compound

        names = ", ".join(d.name for d in devices)
        where = f" in the {room}" if room else ""
        steps = [
            (d.name, (lambda dev=d: self.backend.set_state(dev.id, state)))
            for d in devices
        ]

        return broker.propose(
            compound(
                kind=ActionKind.DEVICE,
                summary=f"Turn {state} {len(devices)} devices{where}: {names}",
                preview="\n".join(
                    self.build_preview(d, state).description for d in devices
                ),
                steps=steps,
                reversible=all(d.kind != DeviceKind.LOCK for d in devices),
                needs_preview=any(d.needs_preview for d in devices),
            ),
            perception_state,
        )

    def propose_from_speech(
        self,
        text: str,
        broker: ActionBroker,
        perception_state: Any = None,
    ) -> tuple[list[Action], Resolution]:
        """Full path: spoken phrase → resolved devices → proposed actions."""
        resolution = resolve(text, self.backend.devices())

        if not resolution.devices:
            logger.info("could not resolve '{}': {}", text, resolution.note)
            return [], resolution
        if not resolution.state:
            resolution.note = "could not tell whether that means on or off"
            return [], resolution
        if resolution.ambiguous and len(resolution.devices) > 1 and not resolution.room:
            resolution.note = (
                "ambiguous: " + ", ".join(f"{d.name} ({d.room})" for d in resolution.devices)
            )
            return [], resolution

        actions = [
            self.propose(device, resolution.state, broker, perception_state)
            for device in resolution.devices
        ]
        return actions, resolution


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA home control (Phase 10)")
    parser.add_argument("command", nargs="*", help='e.g. "turn off the lights in the study"')
    parser.add_argument("--list", action="store_true", help="list all known hardware by room")
    parser.add_argument("--mock", action="store_true", help="force mock devices")
    parser.add_argument("--yes", action="store_true", help="confirm without prompting")
    args = parser.parse_args()

    bootstrap("home")

    backend = None
    if args.mock:
        from aura.home.registry import MockBackend

        backend = MockBackend()

    controller = HomeController(backend=backend)
    print(f"source: {controller.backend.describe_backend()}\n")

    if args.list or not args.command:
        print(controller.describe_house())
        return 0

    text = " ".join(args.command)
    broker = ActionBroker()
    actions, resolution = controller.propose_from_speech(text, broker)

    if not actions:
        print(f'could not act on "{text}"')
        print(f"  {resolution.note}")
        return 1

    print(f'"{text}" resolved to {len(actions)} action(s):\n')
    for action in actions:
        print("--- proposed ---")
        print(action.describe())
        if action.needs_preview:
            print("\n[preview required before this is committed to hardware]")
        print("----------------")

    answer = "y" if args.yes else input("\nconfirm? [y/N] ").strip().lower()
    if answer != "y":
        count = broker.reject_all("declined at prompt")
        print(f"cancelled - {count} action(s) discarded, nothing changed")
        return 0

    for action in actions:
        done = broker.confirm(action.id)
        print(f"  {'FAILED: ' + done.error if done.error else 'done: ' + done.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
