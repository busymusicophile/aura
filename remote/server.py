"""
AURA — remote access (Phase 11).

A minimal authenticated command interface so her phone can check the house and
toggle a device from outside. Not remote desktop - a small, explicit API with a
fixed set of operations.

Design principle 8, enforced mechanically
-----------------------------------------
The control laptop is never exposed to the public internet. This server refuses
to bind to anything except a Tailscale address, and it verifies that at startup
rather than trusting configuration:

* It locates the Tailscale interface (100.64.0.0/10, the CGNAT range Tailscale
  uses) and binds only to that address.
* Binding to 0.0.0.0, or to a LAN or public address, raises. There is no flag to
  override it - a flag would eventually get used.
* Every request is checked against a bearer token stored locally.
* Requests arriving from outside the Tailscale range are refused even if the
  token is correct, so a misconfigured router cannot open a hole.

Built on the standard library rather than a web framework. Fewer moving parts on
the one component whose failure mode is "the house is on the internet".
"""

from __future__ import annotations

import ipaddress
import json
import secrets
import socket
import subprocess
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from loguru import logger

from aura import config
from aura.safety import audit

# Tailscale hands out addresses from the CGNAT range.
TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
TOKEN_FILE = config.DATA_DIR / "remote" / "token.txt"
DEFAULT_PORT = 8787


class RemoteAccessError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Tailscale discovery
# --------------------------------------------------------------------------


def tailscale_ip() -> str | None:
    """This machine's Tailscale address, or None if Tailscale is not up."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=6, check=False
        )
        candidate = (result.stdout or "").strip().splitlines()
        if candidate:
            address = candidate[0].strip()
            if is_tailscale_address(address):
                return address
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fall back to scanning local interfaces for a CGNAT-range address.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if is_tailscale_address(address):
                return address
    except socket.gaierror:
        pass
    return None


def is_tailscale_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address) in TAILSCALE_NET
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Token
# --------------------------------------------------------------------------


def load_token(create: bool = True) -> str:
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    if not create:
        raise RemoteAccessError(f"no token at {TOKEN_FILE}")
    token = secrets.token_urlsafe(32)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    logger.info("generated a new remote access token at {}", TOKEN_FILE)
    return token


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


@dataclass
class RemoteContext:
    """What the server is allowed to reach."""

    home: Any = None          # HomeController
    perception: Any = None    # PerceptionService
    token: str = ""


class _Handler(BaseHTTPRequestHandler):
    context: RemoteContext = RemoteContext()
    server_version = "AURA/1.0"

    # ------------------------------------------------------------------ util
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("remote: " + fmt, *args)

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # This API is for her phone, not a browser page. No CORS, no caching.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        peer = self.client_address[0]
        # Even a valid token is refused from outside the VPN, so a
        # misconfigured router cannot turn this into a public endpoint.
        if not is_tailscale_address(peer) and not peer.startswith("127."):
            logger.warning("refused remote request from non-Tailscale address {}", peer)
            audit.record(
                audit.Event.REMOTE_ACCESS,
                actor=peer, outcome="refused",
                detail={"reason": "source outside Tailscale range"},
            )
            return False

        header = self.headers.get("Authorization", "")
        supplied = header.removeprefix("Bearer ").strip()
        if not supplied or not secrets.compare_digest(supplied, self.context.token):
            audit.record(
                audit.Event.REMOTE_ACCESS,
                actor=peer, outcome="refused", detail={"reason": "bad token"},
            )
            return False
        return True

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        if not self._authorised():
            self._send(401, {"error": "unauthorised"})
            return

        route = self.path.rstrip("/") or "/"
        audit.record(
            audit.Event.REMOTE_ACCESS,
            actor=self.client_address[0], outcome="ok",
            detail={"method": "GET", "route": route},
        )

        if route == "/status":
            self._send(200, self._status())
        elif route == "/devices":
            self._send(200, {"devices": self._devices()})
        elif route == "/presence":
            self._send(200, self._presence())
        else:
            self._send(404, {"error": "no such route",
                             "routes": ["/status", "/devices", "/presence"]})

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        if not self._authorised():
            self._send(401, {"error": "unauthorised"})
            return

        route = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "malformed JSON"})
            return

        if not route.startswith("/device/"):
            self._send(404, {"error": "no such route"})
            return

        device_id = route.removeprefix("/device/")
        state = str(payload.get("state", "")).lower()
        if state not in ("on", "off"):
            self._send(400, {"error": "state must be 'on' or 'off'"})
            return

        home = self.context.home
        if home is None:
            self._send(503, {"error": "home control is not running"})
            return

        device = home.backend.get(device_id)
        if device is None:
            self._send(404, {"error": f"no device {device_id}"})
            return

        # Anything requiring a preview cannot be driven remotely. A preview is
        # only meaningful to someone in the room, and she is not in the room.
        if device.needs_preview:
            audit.record(
                audit.Event.REMOTE_ACCESS,
                actor=self.client_address[0], outcome="refused",
                detail={"device": device.name, "reason": "needs an in-room preview"},
            )
            self._send(403, {
                "error": f"{device.name} needs a preview first and cannot be "
                         "controlled remotely",
            })
            return

        try:
            updated = home.backend.set_state(device_id, state)
        except Exception as exc:  # noqa: BLE001
            audit.record(
                audit.Event.REMOTE_ACCESS,
                actor=self.client_address[0], outcome="failed",
                detail={"device": device.name, "error": str(exc)},
            )
            self._send(500, {"error": str(exc)})
            return

        audit.record(
            audit.Event.DEVICE_COMMAND,
            actor=f"remote:{self.client_address[0]}", outcome="ok",
            detail={"device": updated.name, "room": updated.room, "state": state},
        )
        self._send(200, {"device": updated.name, "room": updated.room, "state": updated.state})

    # ----------------------------------------------------------------- data
    def _status(self) -> dict[str, Any]:
        from aura.runtime import gpu_report

        home = self.context.home
        devices = home.backend.devices() if home else []
        return {
            "ok": True,
            "devices": len(devices),
            "on": sum(1 for d in devices if d.is_on),
            "unreachable": sum(1 for d in devices if not d.reachable),
            "backend": home.backend.describe_backend() if home else "not running",
            "gpu": gpu_report().get("torch_cuda", False),
        }

    def _devices(self) -> list[dict[str, Any]]:
        home = self.context.home
        if home is None:
            return []
        return [
            {
                "id": d.id, "name": d.name, "room": d.room, "kind": d.kind.value,
                "state": d.state, "reachable": d.reachable,
                "remote_controllable": not d.needs_preview,
            }
            for d in home.backend.devices()
        ]

    def _presence(self) -> dict[str, Any]:
        perception = self.context.perception
        if perception is None:
            return {"available": False, "people": []}
        state = perception.state()
        # Names only. No images, no similarity scores, nothing leaves the house
        # that is not needed to answer "is anyone home".
        return {
            "available": True,
            "people": [p.name for p in state.people],
            "summary": state.describe(),
        }


class RemoteServer:
    """Tailscale-only HTTP interface."""

    def __init__(
        self,
        home: Any = None,
        perception: Any = None,
        port: int = DEFAULT_PORT,
        bind: str | None = None,
    ) -> None:
        self.port = port
        self.address = bind or tailscale_ip()

        if not self.address:
            raise RemoteAccessError(
                "no Tailscale address found. Install Tailscale and sign in, then "
                "run this again. AURA will not bind to a LAN or public address."
            )
        if self.address in ("0.0.0.0", "::"):  # noqa: S104 - explicitly rejecting it
            raise RemoteAccessError(
                "refusing to bind to all interfaces - that would expose the "
                "control laptop beyond the VPN"
            )
        if not is_tailscale_address(self.address):
            raise RemoteAccessError(
                f"{self.address} is not a Tailscale address (100.64.0.0/10). "
                "Remote access is VPN-only by design."
            )

        _Handler.context = RemoteContext(
            home=home, perception=perception, token=load_token()
        )
        self._httpd: ThreadingHTTPServer | None = None

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer((self.address, self.port), _Handler)
        logger.info("remote access listening on {}:{} (Tailscale only)",
                    self.address, self.port)
        audit.record(
            audit.Event.STARTUP,
            detail={"service": "remote", "bind": f"{self.address}:{self.port}"},
        )
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()


def check_exposure(port: int = DEFAULT_PORT) -> dict[str, Any]:
    """Report whether anything is listening on a non-Tailscale interface."""
    import psutil

    findings: list[dict[str, Any]] = []
    for conn in psutil.net_connections(kind="inet"):
        if conn.status != psutil.CONN_LISTEN or not conn.laddr:
            continue
        if conn.laddr.port != port:
            continue
        address = conn.laddr.ip
        findings.append({
            "address": address,
            "port": conn.laddr.port,
            "tailscale_only": is_tailscale_address(address),
            "all_interfaces": address in ("0.0.0.0", "::"),  # noqa: S104
        })

    exposed = [f for f in findings if f["all_interfaces"] or not f["tailscale_only"]]
    return {
        "listeners": findings,
        "publicly_exposed": bool(exposed),
        "verdict": "EXPOSED - fix this" if exposed else "no public listener on this port",
    }


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA remote access (Phase 11)")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check", action="store_true", help="report readiness and exposure")
    parser.add_argument("--token", action="store_true", help="print the access token")
    parser.add_argument("--mock", action="store_true", help="serve mock devices")
    args = parser.parse_args()

    bootstrap("remote")

    if args.token:
        print(load_token())
        return 0

    if args.check or not args.serve:
        address = tailscale_ip()
        print(f"tailscale address : {address or 'NOT FOUND — Tailscale is not set up'}")
        print(f"token             : {'set' if TOKEN_FILE.exists() else 'not yet generated'}")
        exposure = check_exposure(args.port)
        print(f"port {args.port} listeners  : {exposure['listeners'] or 'none'}")
        print(f"exposure verdict  : {exposure['verdict']}")
        if not address:
            print("\nAURA will refuse to start remote access until Tailscale is up.")
            return 1
        return 0

    from aura.home.control import HomeController
    from aura.home.registry import MockBackend

    home = HomeController(backend=MockBackend() if args.mock else None)
    server = RemoteServer(home=home, port=args.port)
    print(f"token: {load_token()}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
