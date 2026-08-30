"""
AURA — the assembled assistant.

Wires perception, intent routing, brain, action layers, voice and orb into one
running system:

    speech in -> access tier -> intent -> action or conversation -> voice -> orb

Threading model: Qt owns the main thread because the orb is a widget and Qt
widgets may only be touched from the thread that created them. Perception,
inference and speech all run on worker threads and hand results back through Qt
signals. Every orb update therefore crosses a signal boundary rather than being
called directly - doing it the direct way appears to work and then crashes
intermittently, which is worse than failing loudly.

Confirmation by voice
---------------------
Actions are proposed, described aloud, and executed only after she says yes.
Exactly one action is pending at a time. That is a deliberate limit: with two
pending, "yes" becomes ambiguous, and the safe way to resolve an ambiguous
confirmation is to not have one. A new proposal supersedes an unconfirmed
previous one, which is discarded rather than silently queued.

At the restricted tier nothing is proposed aloud - it goes to the orb as a
silent flag, and only her confirmation there executes it.

Do-not-disturb
--------------
When she says "shut up" (or the like), AURA stops initiating anything -
arrival greetings and proactive suggestions both go silent, and ordinary chat
gets no reply. Real requests still work: asking to switch off a light while
quiet still switches it off, still describes what it did, because "leave me
alone" is a request to stop being talked at, not a request to stop being
useful. DND clears on an explicit resume phrase, or automatically after a
timeout, so a forgotten magic word never leaves her permanently unheard.

Call detection
--------------
An utterance heard while a calling app (WhatsApp, Zoom, Teams, Phone Link...)
is the foreground window is assumed to be her talking on that call, not to
AURA, and is dropped before it reaches intent routing. This is a heuristic
based on what window has focus, not a guarantee - it does not know about a call
on her phone sitting next to the laptop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Protocol

from loguru import logger
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from aura import config
from aura.actions.base import Action, ActionBroker
from aura.chat import AuraBrain
from aura.intent import Intent, IntentKind, IntentRouter
from aura.llm.client import LLMError
from aura.perception.service import PerceptionService, PerceptionState
from aura.perception.speech import Utterance
from aura.runtime import bootstrap
from aura.safety import audit
from aura.ui.orb import OrbState, PresenceOrb
from aura.voice.tts import PiperUnavailable, Speaker


class TierPolicy(Protocol):
    def decide(self, state: PerceptionState, utterance: Utterance): ...


class _FullAccessPolicy:
    """Phase 3 default: everything is allowed and spoken aloud."""

    def decide(self, state: PerceptionState, utterance: Utterance):
        from aura.access.tiers import Decision, Tier

        return Decision(
            tier=Tier.FULL, may_respond=True, may_speak_aloud=True,
            may_reveal_personal=True, may_act=True, instruction="",
            reason="tiers disabled",
        )


# Calling apps whose foreground presence means "she is talking to a person,
# not to AURA". Matched case-insensitively against the foreground process name.
_CALL_APPS = {
    "whatsapp.exe", "zoom.exe", "teams.exe", "ms-teams.exe", "discord.exe",
    "skype.exe", "phonelink.exe", "yourphone.exe", "slack.exe", "webex.exe",
    "googlemeet.exe", "facetime.exe",
}

# How long "shut up" holds before AURA starts initiating things again on its
# own, if she never says an explicit resume phrase. Long enough that a genuine
# "leave me alone" is respected for the rest of a bad hour; short enough that a
# throwaway comment overheard once does not go silent for the rest of the day.
_DND_TIMEOUT_SECONDS = 90 * 60

# An arrival greeting fires at most this often, so a camera flicker that drops
# and re-detects her face inside a few seconds cannot trigger a second one.
_GREETING_COOLDOWN_SECONDS = 20 * 60


class Bridge(QObject):
    """Marshals worker-thread events onto the Qt thread."""

    state_changed = Signal(object)
    flag_raised = Signal(str)
    flag_cleared = Signal()


class Assistant:
    def __init__(
        self,
        tier_policy: TierPolicy | None = None,
        voice: bool = True,
        barge_in: bool = False,
        autonomy: bool = False,
    ) -> None:
        self.bridge = Bridge()
        self.brain = AuraBrain()
        self.policy = tier_policy or _FullAccessPolicy()
        self.router = IntentRouter(llm=self.brain.llm)
        self.broker = ActionBroker(policy=tier_policy)
        self.voice_enabled = voice
        self._busy = threading.Lock()
        self._pending: Action | None = None

        # Do-not-disturb: set by an explicit "shut up", cleared by an explicit
        # resume phrase or by the timeout in _dnd_active().
        self._dnd_until: float = 0.0
        # Arrival greeting bookkeeping, so it fires once per return rather than
        # once per frame she happens to be visible in.
        self._was_present = False
        self._last_greeting = 0.0

        self.perception = PerceptionService(
            on_utterance=self._on_utterance,
            on_presence_change=self._on_presence_change,
            on_barge_in=self._on_barge_in,
            barge_in=barge_in,
        )

        self.speaker: Speaker | None = None
        if voice:
            try:
                self.speaker = Speaker()
                self.speaker.on_speaking_changed = self._on_speaking
            except PiperUnavailable as exc:
                logger.error("voice disabled: {}", exc)
                self.voice_enabled = False

        # Lazily built, because each pulls in heavy imports.
        self._home: Any = None
        self._browser: Any = None
        self._desktop: Any = None

        self.autonomy: Any = None
        if autonomy:
            self._start_autonomy()

        self.orb: PresenceOrb | None = None
        self.broker.on_flag = self._on_action_flagged

    # ------------------------------------------------------------- lazy deps
    @property
    def home(self) -> Any:
        if self._home is None:
            from aura.home.control import HomeController

            self._home = HomeController()
        return self._home

    @property
    def browser(self) -> Any:
        if self._browser is None:
            from aura.actions.browser import BrowserController

            self._browser = BrowserController()
        return self._browser

    @property
    def desktop(self) -> Any:
        if self._desktop is None:
            from aura.actions.desktop import DesktopController

            self._desktop = DesktopController()
        return self._desktop

    # ------------------------------------------------------------------ orb
    def attach_orb(self, orb: PresenceOrb) -> None:
        self.orb = orb
        self.bridge.state_changed.connect(orb.set_state, Qt.QueuedConnection)
        self.bridge.flag_raised.connect(orb.show_flag, Qt.QueuedConnection)
        self.bridge.flag_cleared.connect(orb.clear_flag, Qt.QueuedConnection)

    def _set_state(self, state: OrbState) -> None:
        self.bridge.state_changed.emit(state)

    def _on_speaking(self, speaking: bool) -> None:
        self.perception.mute_mic(speaking)
        self._set_state(OrbState.SPEAKING if speaking else OrbState.IDLE)

    def _on_barge_in(self) -> None:
        if self.speaker:
            self.speaker.stop()
        self._set_state(OrbState.LISTENING)

    def _on_action_flagged(self, action: Action) -> None:
        """An action needs her attention on the orb (restricted tier)."""
        self.bridge.flag_raised.emit(
            "Someone else is here — go ahead, or remind me later?\n\n"
            f"{action.summary}"
        )

    # -------------------------------------------------------- do not disturb
    def _dnd_active(self) -> bool:
        return time.time() < self._dnd_until

    def _do_dnd(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        self._dnd_until = time.time() + _DND_TIMEOUT_SECONDS
        # A stale pending confirmation is exactly the kind of thing "leave me
        # alone" is asking AURA to drop, not carry forward silently.
        if self._pending is not None:
            self.broker.reject(self._pending.id, "do-not-disturb requested")
            self._pending = None
            self.bridge.flag_cleared.emit()
        logger.info("do-not-disturb on for {:.0f} min", _DND_TIMEOUT_SECONDS / 60)
        self._say("Okay.")

    def _do_resume(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        was_quiet = self._dnd_active()
        self._dnd_until = 0.0
        if was_quiet:
            logger.info("do-not-disturb cleared")
            self._say("I'm here.")

    # ------------------------------------------------------------- presence
    def _on_presence_change(self, state: PerceptionState) -> None:
        """Greet her on arrival. Never with anyone else in the room."""
        primary_here = state.primary_present and not state.others_present

        if primary_here and not self._was_present:
            self._maybe_greet()
        self._was_present = primary_here

    def _maybe_greet(self) -> None:
        if self._dnd_active():
            return
        now = time.time()
        if now - self._last_greeting < _GREETING_COOLDOWN_SECONDS:
            return
        self._last_greeting = now

        try:
            reply = self.brain.llm.complete(
                user="She has just arrived and is now alone with you. Greet her "
                     "in one short line - warm, brief, her register. No preamble.",
                system=self.brain.persona.system_prompt(),
            )
        except LLMError as exc:
            logger.debug("greeting skipped: {}", exc)
            return
        self._say(reply.strip())

    # ------------------------------------------------------------ call check
    def _on_a_call(self) -> bool:
        """Best-effort guess that she is talking to a person, not to AURA."""
        from aura.perception import screen

        _, process = screen.active_window()
        return process.lower() in _CALL_APPS

    # ------------------------------------------------------------- autonomy
    def _start_autonomy(self) -> None:
        from aura.autonomy.engine import AutonomyEngine, build_context

        self.autonomy = AutonomyEngine(brain=self.brain, enabled=True)
        self.autonomy.on_suggestion = self._on_suggestion

        def context() -> Any:
            from aura.access.tiers import AccessPolicy

            state = self.perception.state()
            tier = "full"
            if isinstance(self.policy, AccessPolicy):
                tier = self.policy.classify(state)[0].value
            return build_context(
                perception_state=state, tier=tier, memory=self.brain.memory
            )

        self.autonomy.start(context, interval=300.0)

    def _on_suggestion(self, suggestion: Any) -> None:
        """A proactive remark. Spoken only if she is alone and not in DND."""
        if self._dnd_active():
            logger.debug("suggestion withheld - do-not-disturb")
            return
        state = self.perception.state()
        from aura.access.tiers import AccessPolicy, Tier

        tier = Tier.FULL
        if isinstance(self.policy, AccessPolicy):
            tier = self.policy.classify(state)[0]

        if tier is not Tier.FULL:
            logger.debug("suggestion withheld at tier {}", tier.value)
            return
        self._say(suggestion.message)

    # ------------------------------------------------------------ main cycle
    def _on_utterance(self, utterance: Utterance) -> None:
        if self._on_a_call():
            # She is talking to a person in a call app, not to AURA. Dropped
            # before intent routing, not just left unanswered - an unanswered
            # command sits as a stale pending action, a dropped one does not.
            logger.debug("dropped (call in progress): {}", utterance.text)
            return
        if not self._busy.acquire(blocking=False):
            logger.debug("still busy; dropping: {}", utterance.text)
            return
        threading.Thread(
            target=self._handle, args=(utterance,), name="aura-turn", daemon=True
        ).start()

    def _handle(self, utterance: Utterance) -> None:
        try:
            state = self.perception.state()
            decision = self.policy.decide(state, utterance)

            if not decision.may_respond:
                if decision.deflection:
                    self._say(decision.deflection)
                self._set_state(OrbState.IDLE)
                return

            self._set_state(OrbState.THINKING)
            self.brain.conversation.system = self.brain.persona.system_prompt(
                decision.instruction
            )

            intent = self.router.route(
                utterance.text, has_pending=self._pending is not None
            )
            logger.info("intent: {} ({})", intent.kind.value, intent.source)
            self._dispatch(intent, decision, state)

        except LLMError as exc:
            logger.error("llm failed: {}", exc)
            self._say("I couldn't reach the model just then.")
            self._set_state(OrbState.ERROR)
        except Exception:  # noqa: BLE001
            logger.exception("turn failed")
            self._set_state(OrbState.ERROR)
        finally:
            self._set_state(OrbState.IDLE)
            self._busy.release()

    def _dispatch(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        handlers = {
            IntentKind.CONFIRM: self._do_confirm,
            IntentKind.CANCEL: self._do_cancel,
            IntentKind.DO_NOT_DISTURB: self._do_dnd,
            IntentKind.RESUME: self._do_resume,
            IntentKind.DEVICE: self._do_device,
            IntentKind.HOUSE_STATUS: self._do_house_status,
            IntentKind.LAUNCH: self._do_launch,
            IntentKind.BROWSE: self._do_browse,
            IntentKind.SCREEN_QUERY: self._do_screen,
            IntentKind.MESSAGE: self._do_message,
            IntentKind.ENROL: self._do_enrol,
        }
        handler = handlers.get(intent.kind)
        if handler is None:
            self._do_conversation(intent, decision)
            return
        handler(intent, decision, state)

    # ---------------------------------------------------------- conversation
    def _do_conversation(self, intent: Intent, decision: Any) -> None:
        if self._dnd_active():
            # "Leave me alone" was a request to stop being talked at, not a
            # request to stop being useful - so this only silences small talk.
            # A real request (device, launch, browse...) still goes to its own
            # handler and still works; it never reaches here.
            logger.debug("conversation withheld - do-not-disturb")
            return
        if decision.may_speak_aloud and self.voice_enabled and self.speaker:
            reply = self.speaker.say_stream(
                self.brain.respond(intent.text, stream=True)
            )
            logger.info("said: {}", reply[:120])
        else:
            result = self.brain.respond(intent.text)
            logger.info("replied silently: {}", result.text[:120])
            if not decision.may_speak_aloud:
                self.bridge.flag_raised.emit(
                    "Someone else is here — go ahead, or remind me later?\n\n"
                    f"{result.text[:220]}"
                )

    # ------------------------------------------------------------ confirming
    def _propose(self, action: Action, decision: Any) -> None:
        """Register one pending action and describe it."""
        if action.error:
            self._say(action.error)
            return
        if action.status.value == "rejected":
            self._say("I can't do that right now.")
            return

        if self._pending is not None:
            # Only one at a time - see the module docstring.
            self.broker.reject(self._pending.id, "superseded by a newer request")
        self._pending = action

        if not decision.may_speak_aloud:
            self._on_action_flagged(action)
            return

        preview = f" {action.preview}" if action.needs_preview and action.preview else ""
        self._say(f"{action.summary}.{preview} Shall I?")

    def _do_confirm(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        if self._pending is None:
            self._say("There's nothing waiting.")
            return
        action, self._pending = self._pending, None
        done = self.broker.confirm(action.id, actor=config.SETTINGS.primary_user)
        self.bridge.flag_cleared.emit()

        if done.error:
            self._say(f"That didn't work. {done.error}")
            return

        # A multi-device action can partly succeed. Saying a flat "done" when one
        # bulb was unreachable would be a quiet lie about the state of her house.
        from aura.actions.base import CompoundResult

        if isinstance(done.result, CompoundResult) and done.result.failed:
            self._say(done.result.describe())
        else:
            self._say("Done.")

    def _do_cancel(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        if self._pending is None:
            self._say("Nothing to cancel.")
            return
        action, self._pending = self._pending, None
        self.broker.reject(action.id, "cancelled by voice")
        self.bridge.flag_cleared.emit()
        self._say("Cancelled.")

    # --------------------------------------------------------------- actions
    def _do_device(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        from aura.home.control import resolve

        resolution = resolve(intent.text, self.home.backend.devices())

        if not resolution.devices:
            self._say(f"I couldn't work that one out. {resolution.note}")
            return
        if not resolution.state:
            self._say("Did you mean on or off?")
            return
        if resolution.ambiguous and len(resolution.devices) > 1 and not resolution.room:
            options = ", ".join(f"{d.name} in the {d.room}" for d in resolution.devices)
            self._say(f"Which one? There's {options}.")
            return

        if len(resolution.devices) == 1:
            action = self.home.propose(
                resolution.devices[0], resolution.state, self.broker, state
            )
        else:
            # One intention, one confirmation, but every device still switched.
            action = self.home.propose_batch(
                resolution.devices, resolution.state, self.broker, state,
                room=resolution.room,
            )
        self._propose(action, decision)

    def _do_house_status(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        devices = self.home.backend.devices()
        on = [d for d in devices if d.is_on]
        if not on:
            self._say("Nothing is on right now.")
            return
        listing = ", ".join(f"{d.name} in the {d.room}" for d in on[:6])
        self._say(f"{len(on)} things are on: {listing}.")

    def _do_launch(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        target = _strip_verb(intent.text)
        self._propose(self.desktop.propose_launch(target, self.broker, state), decision)

    def _do_browse(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        target = _strip_verb(intent.text)
        self._propose(self.browser.propose_open(target, self.broker, state), decision)

    def _do_screen(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        context = self.perception.screen_context()
        if not decision.may_reveal_personal:
            # Her screen is personal by default; do not narrate it to a room.
            self.bridge.flag_raised.emit(f"On screen:\n\n{context.summary(200)}")
            return
        reply = self.brain.llm.complete(
            user=f"In one or two sentences, what is she looking at?\n\n{context.summary()}",
            system=self.brain.persona.system_prompt(),
        )
        self._say(reply)

    def _do_message(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        # Reading mail needs credentials that may not exist; fail informatively
        # rather than silently falling back to chatter.
        from aura.actions.comms import GmailClient, GmailUnavailable

        try:
            unread = GmailClient().inbox(limit=3)
        except GmailUnavailable as exc:
            logger.info("gmail unavailable: {}", exc)
            self._say("Email isn't set up yet.")
            return

        if not unread:
            self._say("Nothing new.")
            return
        if not decision.may_reveal_personal:
            self.bridge.flag_raised.emit(
                f"{len(unread)} unread. Not reading them out with someone here."
            )
            return
        self._say(f"{len(unread)} unread. " + " ".join(m.describe() for m in unread[:2]))

    def _do_enrol(self, intent: Intent, decision: Any, state: PerceptionState) -> None:
        if not decision.may_act and decision.tier.value not in ("full", "restricted"):
            self._say("I can't do that.")
            return
        name = intent.detail.get("name", "")
        relation = intent.detail.get("relation", "")
        if not name:
            return
        ok = self.perception.enroll(name, relation)
        self._say(
            f"Got it, I'll remember {name}." if ok
            else f"I couldn't get a clear look at {name}. Try again facing the camera."
        )

    # ------------------------------------------------------------------ util
    def _say(self, text: str) -> None:
        if self.voice_enabled and self.speaker:
            self.speaker.say(text)
        else:
            logger.info("(silent) {}", text)

    # -------------------------------------------------------------------- api
    def start(self) -> None:
        self.perception.start()
        self._set_state(OrbState.IDLE)
        audit.record(audit.Event.STARTUP, detail={"interface": "assistant"})
        logger.info("AURA is listening")

    def stop(self) -> None:
        if self.autonomy:
            self.autonomy.stop()
        self.perception.stop()
        if self.speaker:
            self.speaker.stop()
        if self._browser is not None:
            self._browser.close()


_VERB_PREFIX = ("open ", "launch ", "start ", "run ", "go to ", "visit ",
                "pull up ", "search for ", "look up ", "google ")


def _strip_verb(text: str) -> str:
    """Turn 'open notepad' into 'notepad'."""
    lowered = text.strip().lower()
    for prefix in _VERB_PREFIX:
        if lowered.startswith(prefix):
            return text.strip()[len(prefix):].strip(" .?!") or text
    return text.strip(" .?!")


def main() -> int:
    parser = argparse.ArgumentParser(description="AURA voice assistant")
    parser.add_argument("--no-voice", action="store_true", help="text output only")
    parser.add_argument("--no-orb", action="store_true", help="headless")
    parser.add_argument("--no-tiers", action="store_true",
                        help="treat everyone as the primary user")
    parser.add_argument("--barge-in", action="store_true",
                        help="allow interrupting AURA (use headphones)")
    parser.add_argument("--autonomy", action="store_true",
                        help="enable proactive suggestions")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    bootstrap("assistant", level="DEBUG" if args.debug else "INFO")

    policy = None
    if not args.no_tiers:
        from aura.access.tiers import AccessPolicy

        policy = AccessPolicy()

    app = QApplication(sys.argv)
    assistant = Assistant(
        tier_policy=policy, voice=not args.no_voice,
        barge_in=args.barge_in, autonomy=args.autonomy,
    )

    if not args.no_orb:
        orb = PresenceOrb()
        assistant.attach_orb(orb)
        orb.show()

    assistant.start()
    if assistant.speaker:
        assistant.speaker.say(f"Ready, {config.SETTINGS.primary_user}.", block=False)

    try:
        return app.exec()
    finally:
        assistant.stop()


if __name__ == "__main__":
    raise SystemExit(main())
