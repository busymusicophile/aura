"""
AURA — communications (Phase 6).

Email through the Gmail API, and outbound messages through WhatsApp Desktop or
Phone Link.

Three rules, all enforced here rather than left to the caller:

* **Read-only by default.** The Gmail client requests the readonly scope unless
  compose is explicitly enabled. The token it holds cannot send mail at all - not
  "does not", cannot. If the send path is ever reached by accident, Google
  refuses it.

* **Draft, never auto-send.** Outbound messages are proposed through the action
  broker and require confirmation. The body is shown in full first.

* **Sensitive mail is flagged, not summarised.** Mail matching sensitive keywords
  is surfaced as "there is something here you should read yourself" rather than
  having its contents read out. A bank mail summarised aloud in a room with other
  people is exactly the failure the access tiers exist to prevent.

Everything read is redacted before it reaches memory, the model, or speech.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from aura import config
from aura.actions.base import Action, ActionBroker, ActionKind
from aura.safety import redaction

# Read-only. Sending requires the caller to opt in explicitly.
SCOPES_READONLY = ["https://www.googleapis.com/auth/gmail.readonly"]
SCOPES_COMPOSE = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

CREDENTIALS_FILE = config.DATA_DIR / "google" / "credentials.json"
TOKEN_FILE = config.DATA_DIR / "google" / "token.json"

# Subjects and senders where the content must not be spoken or summarised.
SENSITIVE_KEYWORDS = [
    "bank", "account statement", "credit card", "debit card", "transaction",
    "otp", "one-time", "verification code", "password", "reset your password",
    "aadhaar", "pan card", "income tax", "loan", "emi", "insurance", "policy",
    "medical", "diagnosis", "test result", "prescription", "lab report",
    "salary", "payslip", "invoice", "payment failed", "kyc",
]
_SENSITIVE = re.compile("|".join(re.escape(k) for k in SENSITIVE_KEYWORDS), re.I)


@dataclass
class Email:
    id: str
    sender: str
    subject: str
    snippet: str
    date: str
    is_sensitive: bool = False
    matched: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.is_sensitive:
            return (
                f"From {self.sender}: something about "
                f"{', '.join(self.matched[:2])}. I'm not reading this one out - "
                f"open it yourself when you can."
            )
        return f"From {self.sender}: {self.subject}. {self.snippet[:160]}"


@dataclass
class MessageDraft:
    channel: str          # "whatsapp" | "phonelink" | "email"
    recipient: str
    body: str
    subject: str = ""

    def preview(self) -> str:
        lines = [f"To: {self.recipient}", f"Via: {self.channel}"]
        if self.subject:
            lines.append(f"Subject: {self.subject}")
        lines += ["", self.body]
        return "\n".join(lines)


def classify_sensitivity(subject: str, sender: str, snippet: str) -> tuple[bool, list[str]]:
    """Does this mail need her eyes rather than AURA's voice?"""
    haystack = f"{subject} {sender} {snippet}"
    matched = sorted({m.group(0).lower() for m in _SENSITIVE.finditer(haystack)})
    return bool(matched), matched


# --------------------------------------------------------------------------
# Gmail
# --------------------------------------------------------------------------


class GmailUnavailable(RuntimeError):
    pass


class GmailClient:
    """Gmail, read-only unless compose is explicitly requested."""

    def __init__(self, allow_compose: bool = False) -> None:
        self.allow_compose = allow_compose
        self.scopes = SCOPES_COMPOSE if allow_compose else SCOPES_READONLY
        self._service: Any = None

    def _authenticate(self) -> Any:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        if not CREDENTIALS_FILE.exists():
            raise GmailUnavailable(
                f"no Google OAuth client at {CREDENTIALS_FILE}. "
                "Create a Desktop OAuth client in Google Cloud Console, enable the "
                "Gmail API, download the JSON, and save it there."
            )

        creds = None
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), self.scopes
                )
                # Local loopback only; the token never leaves this machine.
                creds = flow.run_local_server(port=0)
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    def _ensure(self) -> Any:
        if self._service is None:
            self._service = self._authenticate()
        return self._service

    def inbox(self, limit: int = 10, unread_only: bool = True) -> list[Email]:
        service = self._ensure()
        query = "is:unread" if unread_only else ""
        listing = (
            service.users().messages()
            .list(userId="me", maxResults=limit, q=query)
            .execute()
        )

        emails: list[Email] = []
        for ref in listing.get("messages", []):
            detail = (
                service.users().messages()
                .get(userId="me", id=ref["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"]
                for h in detail.get("payload", {}).get("headers", [])
            }
            subject = headers.get("subject", "(no subject)")
            sender = headers.get("from", "(unknown)")
            snippet = detail.get("snippet", "")

            sensitive, matched = classify_sensitivity(subject, sender, snippet)
            emails.append(
                Email(
                    id=ref["id"],
                    sender=redaction.redact(sender).text,
                    subject=redaction.redact(subject).text,
                    # Sensitive mail keeps no body at all in memory.
                    snippet="" if sensitive else redaction.redact(snippet).text,
                    date=headers.get("date", ""),
                    is_sensitive=sensitive,
                    matched=matched,
                )
            )
        return emails

    def read(self, message_id: str) -> str:
        """Full body of one message, redacted. Explicit request only."""
        service = self._ensure()
        detail = (
            service.users().messages()
            .get(userId="me", id=message_id, format="full").execute()
        )
        body = _extract_body(detail.get("payload", {}))
        return redaction.redact(body).text

    def create_draft(self, draft: MessageDraft) -> dict[str, Any]:
        """Save a Gmail draft. Saving is not sending; she still presses send."""
        if not self.allow_compose:
            raise GmailUnavailable(
                "this client is read-only - construct GmailClient(allow_compose=True)"
            )
        import email.message

        service = self._ensure()
        message = email.message.EmailMessage()
        message["To"] = draft.recipient
        message["Subject"] = draft.subject or "(no subject)"
        message.set_content(draft.body)

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return (
            service.users().drafts()
            .create(userId="me", body={"message": {"raw": encoded}}).execute()
        )


def _extract_body(payload: dict[str, Any]) -> str:
    """Walk a MIME tree for the best text part."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []) or []:
        text = _extract_body(part)
        if text:
            return text
    data = payload.get("body", {}).get("data", "")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return ""


# --------------------------------------------------------------------------
# Messaging
# --------------------------------------------------------------------------


class MessageSender:
    """Outbound messages via WhatsApp Desktop or Phone Link.

    Both channels open the app with the message pre-filled and stop there. AURA
    does not press send - she does. That is a deliberate second gate on top of
    the broker confirmation: even if something went wrong upstream, the worst
    case is a composed message sitting on screen, not a sent one.
    """

    def __init__(self, channel: str = "whatsapp") -> None:
        if channel not in ("whatsapp", "phonelink"):
            raise ValueError(f"unknown channel: {channel}")
        self.channel = channel

    def _compose_whatsapp(self, draft: MessageDraft) -> str:
        import os
        import urllib.parse

        phone = re.sub(r"[^\d+]", "", draft.recipient).lstrip("+")
        text = urllib.parse.quote(draft.body)
        uri = f"whatsapp://send?phone={phone}&text={text}" if phone else f"whatsapp://send?text={text}"
        os.startfile(uri)  # noqa: S606
        return uri

    def _compose_phonelink(self, draft: MessageDraft) -> str:
        import os
        import urllib.parse

        body = urllib.parse.quote(draft.body)
        uri = f"ms-phone:?PhoneNumber={draft.recipient}&Body={body}"
        os.startfile(uri)  # noqa: S606
        return uri

    def compose(self, draft: MessageDraft) -> str:
        clean = redaction.redact(draft.body)
        if not clean.is_clean:
            raise ValueError(
                f"refusing to compose a message containing {clean.summary()}"
            )
        if self.channel == "whatsapp":
            return self._compose_whatsapp(draft)
        return self._compose_phonelink(draft)

    def propose(
        self, draft: MessageDraft, broker: ActionBroker, state: Any = None
    ) -> Action:
        return broker.propose(
            Action(
                kind=ActionKind.MESSAGE,
                summary=f"Open {self.channel} with a message to {draft.recipient}",
                preview=draft.preview(),
                run=lambda: self.compose(draft),
                # Nothing is sent by this action, so it is reversible: she can
                # close the window without pressing send.
                reversible=True,
            ),
            state,
        )


def draft_reply(brain: Any, incoming: str, sender: str = "") -> MessageDraft:
    """Have the persona draft a reply. Returns a draft; sends nothing."""
    prompt = (
        f"Draft a short reply to this message{f' from {sender}' if sender else ''}. "
        "Write only the reply text, in her voice, no preamble or sign-off.\n\n"
        f"Message: {redaction.redact(incoming).text}"
    )
    body = brain.llm.complete(user=prompt, system=brain.persona.system_prompt())
    return MessageDraft(channel="whatsapp", recipient=sender or "(unspecified)", body=body.strip())


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA communications (Phase 6)")
    parser.add_argument("--inbox", action="store_true", help="list unread mail")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--draft-to", metavar="RECIPIENT")
    parser.add_argument("--reply-to", metavar="TEXT", help="draft a reply to this text")
    parser.add_argument("--channel", default="whatsapp", choices=["whatsapp", "phonelink"])
    parser.add_argument("--check", action="store_true", help="check setup without connecting")
    args = parser.parse_args()

    bootstrap("comms")

    if args.check:
        print(f"credentials : {CREDENTIALS_FILE} {'FOUND' if CREDENTIALS_FILE.exists() else 'MISSING'}")
        print(f"token       : {TOKEN_FILE} {'FOUND' if TOKEN_FILE.exists() else 'not yet authorised'}")
        print(f"scopes      : read-only unless --draft-to is used")
        print("\nsensitivity classifier:")
        for subject in ["Your bank statement is ready", "Lunch tomorrow?",
                        "Your OTP is inside", "Re: project notes"]:
            sensitive, matched = classify_sensitivity(subject, "", "")
            print(f"  {subject:<36} sensitive={sensitive} {matched}")
        return 0

    if args.inbox:
        try:
            for mail in GmailClient().inbox(limit=args.limit):
                flag = "[SENSITIVE] " if mail.is_sensitive else ""
                print(f"{flag}{mail.describe()}")
        except GmailUnavailable as exc:
            print(f"gmail unavailable: {exc}")
            return 1
        return 0

    if args.reply_to:
        from aura.chat import AuraBrain

        brain = AuraBrain()
        draft = draft_reply(brain, args.reply_to, args.draft_to or "")
        draft.channel = args.channel
        broker = ActionBroker()
        sender = MessageSender(args.channel)
        action = sender.propose(draft, broker)
        print("\n--- draft (nothing sent) ---")
        print(action.describe())
        print("----------------------------")
        if input("open the app with this message? [y/N] ").strip().lower() == "y":
            done = broker.confirm(action.id)
            print(f"failed: {done.error}" if done.error else "app opened - press send yourself")
        else:
            broker.reject(action.id, "declined")
            print("discarded")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
