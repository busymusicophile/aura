"""
AURA — Home Assistant client (Phase 10).

Implements the same `DeviceBackend` interface as the Phase 9 mock, so the control
panel and action layer switch to real hardware without changing a line.

Connection details live in `C:\\AURA\\data\\home\\hass.json` (or the
AURA_HASS_URL / AURA_HASS_TOKEN environment variables). The token is a
long-lived access token created in Home Assistant under your user profile.

Local network only. The base URL is validated as a private address, because a
public Home Assistant URL would mean the house is reachable from the internet -
exactly what design principle 8 forbids. Remote access is Tailscale-only, and
that is Phase 11's job, not this file's.
"""

from __future__ import annotations

import ipaddress
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from aura import config
from aura.home.registry import Device, DeviceKind

CONFIG_FILE = config.DATA_DIR / "home" / "hass.json"
DEFAULT_URL = "http://homeassistant.local:8123"

# Home Assistant domain -> our device kind.
_DOMAIN_KIND = {
    "light": DeviceKind.LIGHT,
    "switch": DeviceKind.SWITCH,
    "fan": DeviceKind.FAN,
    "climate": DeviceKind.AC,
    "media_player": DeviceKind.TV,
    "lock": DeviceKind.LOCK,
    "camera": DeviceKind.CAMERA,
    "binary_sensor": DeviceKind.SENSOR,
    "sensor": DeviceKind.SENSOR,
}

# Domains AURA will not touch at all, whatever is asked. Anything here is either
# a safety device (where a wrong toggle matters a great deal) or has no business
# being voice-controlled.
BLOCKED_DOMAINS = {"alarm_control_panel", "water_heater", "vacuum"}


class HomeAssistantUnavailable(RuntimeError):
    pass


def _load_settings() -> tuple[str, str]:
    url = os.environ.get("AURA_HASS_URL", "")
    token = os.environ.get("AURA_HASS_TOKEN", "")
    if url and token:
        return url, token

    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return raw.get("url", DEFAULT_URL), raw.get("token", "")
        except json.JSONDecodeError as exc:
            logger.error("hass.json is malformed: {}", exc)
    return url or DEFAULT_URL, token


def is_local_address(url: str) -> bool:
    """True when the URL points somewhere on the local network.

    `.local` mDNS names and bare hostnames are accepted; a public IP is not.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith((".local", ".lan", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_private or address.is_loopback
    except ValueError:
        # A hostname we cannot resolve here. Treat as local only if it has no
        # dots (a LAN name); anything domain-shaped is assumed public.
        return "." not in host


class HomeAssistantBackend:
    """Talks to Home Assistant's REST API."""

    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        loaded_url, loaded_token = _load_settings()
        self.url = (url or loaded_url).rstrip("/")
        self.token = token or loaded_token
        self._cache: dict[str, Device] = {}

        if not is_local_address(self.url):
            raise HomeAssistantUnavailable(
                f"{self.url} is not a local address. AURA only talks to Home "
                "Assistant over the local network; remote access is Tailscale-only."
            )

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: Any = None, timeout: float = 8.0) -> Any:
        import urllib.error
        import urllib.request

        if not self.token:
            raise HomeAssistantUnavailable(
                f"no Home Assistant token. Create a long-lived token in Home "
                f"Assistant and save it to {CONFIG_FILE}"
            )

        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.url}/api/{path.lstrip('/')}",
            data=data, headers=self._headers(), method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode()
            return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise HomeAssistantUnavailable("token rejected by Home Assistant") from exc
            raise HomeAssistantUnavailable(f"HTTP {exc.code} from Home Assistant") from exc
        except urllib.error.URLError as exc:
            raise HomeAssistantUnavailable(f"cannot reach {self.url}: {exc.reason}") from exc

    # -------------------------------------------------------------- interface
    def available(self) -> bool:
        try:
            result = self._request("GET", "/", timeout=3.0)
            return bool(result)
        except Exception:  # noqa: BLE001
            return False

    def describe_backend(self) -> str:
        return f"Home Assistant at {self.url}"

    def devices(self) -> list[Device]:
        states = self._request("GET", "/states") or []
        devices: list[Device] = []

        for entry in states:
            entity_id = entry.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain in BLOCKED_DOMAINS or domain not in _DOMAIN_KIND:
                continue

            attributes = entry.get("attributes", {}) or {}
            name = attributes.get("friendly_name", entity_id)
            state = entry.get("state", "unknown")

            device = Device(
                id=entity_id.replace(".", "_"),
                name=name,
                kind=_DOMAIN_KIND[domain],
                # Home Assistant areas are not exposed by /states, so the room
                # comes from the local registry, which is the authority anyway.
                room=self._room_for(entity_id, attributes),
                entity_id=entity_id,
                state=state,
                attributes=attributes,
                reachable=state not in ("unavailable", "unknown"),
                last_changed=entry.get("last_changed", ""),
            )
            devices.append(device)
            self._cache[device.id] = device

        return devices

    def _room_for(self, entity_id: str, attributes: dict[str, Any]) -> str:
        from aura.home.registry import load_registry

        for known in load_registry():
            if known.entity_id == entity_id:
                return known.room
        return str(attributes.get("area", "") or "unassigned")

    def get(self, device_id: str) -> Device | None:
        if device_id in self._cache:
            return self._cache[device_id]
        for device in self.devices():
            if device.id == device_id:
                return device
        return None

    def set_state(self, device_id: str, state: str) -> Device:
        device = self.get(device_id)
        if device is None:
            raise KeyError(f"no device {device_id}")
        if not device.reachable:
            raise RuntimeError(f"{device.name} is unreachable")

        domain = device.entity_id.split(".")[0]
        if domain in BLOCKED_DOMAINS:
            raise PermissionError(f"{domain} devices are not controllable by AURA")

        service = self._service_for(domain, state)
        self._request(
            "POST", f"/services/{domain}/{service}",
            payload={"entity_id": device.entity_id},
        )

        device.state = state
        device.last_changed = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("{} -> {}", device.name, state)
        return device

    @staticmethod
    def _service_for(domain: str, state: str) -> str:
        if domain == "lock":
            return "lock" if state == "off" else "unlock"
        if domain == "media_player":
            return "turn_off" if state == "off" else "turn_on"
        return "turn_off" if state == "off" else "turn_on"

    # ------------------------------------------------------------------ setup
    @staticmethod
    def write_config(url: str, token: str) -> Path:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps({"url": url, "token": token}, indent=2), encoding="utf-8"
        )
        return CONFIG_FILE


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA Home Assistant client (Phase 10)")
    parser.add_argument("--check", action="store_true", help="test the connection")
    parser.add_argument("--list", action="store_true", help="list devices")
    parser.add_argument("--configure", nargs=2, metavar=("URL", "TOKEN"))
    args = parser.parse_args()

    bootstrap("hass")

    if args.configure:
        path = HomeAssistantBackend.write_config(*args.configure)
        print(f"saved to {path}")
        return 0

    url, token = _load_settings()
    print(f"url    : {url}")
    print(f"local  : {is_local_address(url)}")
    print(f"token  : {'set' if token else 'MISSING'}")

    try:
        backend = HomeAssistantBackend()
    except HomeAssistantUnavailable as exc:
        print(f"\nnot configured: {exc}")
        return 1

    if not backend.available():
        print("\nnot reachable - AURA will fall back to mock devices")
        return 1

    print("\nconnected")
    if args.list:
        from aura.home.registry import by_room

        for room, items in by_room(backend.devices()).items():
            print(f"\n{room.upper()}")
            for device in items:
                print(f"  {device.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
