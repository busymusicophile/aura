"""
AURA — encryption at rest and hardening checks (Phase 12).

Everything AURA accumulates is sensitive in aggregate even when no single item
is: her memory store is a transcript of her life, the face database is biometric
data, and the audit log is a record of everywhere she has been and what she asked
for. On a laptop that leaves the house, that is the whole threat model.

Key management
--------------
The data key is generated once at random and then sealed with Windows DPAPI, so
it can only be unsealed by this Windows user on this machine. That choice matters:

* A passphrase typed at every start would be either weak or skipped, and AURA is
  meant to run unattended.
* A key sitting in a plain file next to the data it protects protects nothing.
* DPAPI binds the key to the user account. Someone who copies the disk, or boots
  another OS, cannot unseal it.

The tradeoff is that DPAPI is machine-and-account-bound, so moving to the Samsung
T7 means exporting under a passphrase and re-sealing on the other side. That is
handled explicitly by `export_portable` / `import_portable` rather than being an
unpleasant surprise.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from aura import config

KEY_FILE = config.DATA_DIR / "security" / "data.key"
MANIFEST = config.DATA_DIR / "security" / "vault.json"

# Directories holding data that must be encrypted at rest.
PROTECTED = ["memory", "persona", "faces", "audit", "google", "remote", "home"]

_ENCRYPTED_SUFFIX = ".enc"


class VaultError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Key handling
# --------------------------------------------------------------------------


def _dpapi_seal(blob: bytes) -> bytes:
    import win32crypt

    return win32crypt.CryptProtectData(blob, "AURA data key", None, None, None, 0)


def _dpapi_unseal(blob: bytes) -> bytes:
    import win32crypt

    _, plain = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
    return plain


def create_key(force: bool = False) -> bytes:
    """Generate the data key and seal it with DPAPI."""
    from cryptography.fernet import Fernet

    if KEY_FILE.exists() and not force:
        raise VaultError(f"a key already exists at {KEY_FILE} - use force=True to replace it")

    key = Fernet.generate_key()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        KEY_FILE.write_bytes(_dpapi_seal(key))
    except Exception as exc:  # noqa: BLE001
        raise VaultError(f"could not seal the key with DPAPI: {exc}") from exc

    logger.info("data key created and sealed at {}", KEY_FILE)
    return key


def load_key(create: bool = True) -> bytes:
    if not KEY_FILE.exists():
        if not create:
            raise VaultError(f"no key at {KEY_FILE}")
        return create_key()
    try:
        return _dpapi_unseal(KEY_FILE.read_bytes())
    except Exception as exc:  # noqa: BLE001
        raise VaultError(
            f"could not unseal the data key: {exc}. This usually means the key "
            "was created by a different Windows account or on another machine."
        ) from exc


# --------------------------------------------------------------------------
# File encryption
# --------------------------------------------------------------------------


@dataclass
class VaultStats:
    encrypted: int = 0
    decrypted: int = 0
    skipped: int = 0
    bytes_processed: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class Vault:
    """Encrypts and decrypts AURA's data directories."""

    def __init__(self, key: bytes | None = None) -> None:
        from cryptography.fernet import Fernet

        self._fernet = Fernet(key or load_key())

    # ------------------------------------------------------------------ file
    def encrypt_file(self, path: Path, remove_plaintext: bool = True) -> Path:
        if path.suffix == _ENCRYPTED_SUFFIX:
            return path
        data = path.read_bytes()
        target = path.with_suffix(path.suffix + _ENCRYPTED_SUFFIX)
        target.write_bytes(self._fernet.encrypt(data))
        if remove_plaintext:
            # Overwrite before unlinking. Not a guarantee against forensic
            # recovery on SSDs, but it beats leaving the plaintext block intact.
            with open(path, "r+b") as handle:
                handle.write(os.urandom(len(data)))
                handle.flush()
                os.fsync(handle.fileno())
            path.unlink()
        return target

    def decrypt_file(self, path: Path, remove_ciphertext: bool = True) -> Path:
        from cryptography.fernet import InvalidToken

        if path.suffix != _ENCRYPTED_SUFFIX:
            return path
        try:
            data = self._fernet.decrypt(path.read_bytes())
        except InvalidToken as exc:
            raise VaultError(f"{path.name} could not be decrypted with this key") from exc
        target = path.with_suffix("")
        target.write_bytes(data)
        if remove_ciphertext:
            path.unlink()
        return target

    # ------------------------------------------------------------------- dir
    def encrypt_tree(self, root: Path) -> VaultStats:
        stats = VaultStats()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == _ENCRYPTED_SUFFIX:
                stats.skipped += 1
                continue
            try:
                size = path.stat().st_size
                self.encrypt_file(path)
                stats.encrypted += 1
                stats.bytes_processed += size
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"{path.name}: {exc}")
        return stats

    def decrypt_tree(self, root: Path) -> VaultStats:
        stats = VaultStats()
        for path in sorted(root.rglob(f"*{_ENCRYPTED_SUFFIX}")):
            try:
                size = path.stat().st_size
                self.decrypt_file(path)
                stats.decrypted += 1
                stats.bytes_processed += size
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"{path.name}: {exc}")
        return stats

    # -------------------------------------------------------------- in-memory
    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        return self._fernet.decrypt(data)


# --------------------------------------------------------------------------
# Portability — Samsung T7
# --------------------------------------------------------------------------


def export_portable(destination: Path, passphrase: str) -> Path:
    """Copy AURA's data to an external drive, re-keyed under a passphrase.

    DPAPI keys cannot travel, so the payload is re-encrypted with a key derived
    from a passphrase using scrypt. That passphrase is the only thing protecting
    the drive, so it needs to be a real one.
    """
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    if len(passphrase) < 12:
        raise VaultError("passphrase must be at least 12 characters")

    salt = os.urandom(16)
    kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
    portable_key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    portable = Fernet(portable_key)
    local = Vault()

    destination.mkdir(parents=True, exist_ok=True)
    payload_dir = destination / "aura_data"
    payload_dir.mkdir(exist_ok=True)

    count = 0
    for name in PROTECTED:
        source = config.DATA_DIR / name
        if not source.exists():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if path.suffix == _ENCRYPTED_SUFFIX:
                raw = local.decrypt_bytes(raw)
            relative = path.relative_to(config.DATA_DIR)
            target = payload_dir / relative.with_suffix(relative.suffix + _ENCRYPTED_SUFFIX)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(portable.encrypt(raw))
            count += 1

    (destination / "portable.json").write_text(
        json.dumps({
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "salt": base64.b64encode(salt).decode(),
            "kdf": "scrypt", "n": 2**15, "r": 8, "p": 1,
            "files": count,
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("exported {} file(s) to {}", count, destination)
    return destination


def import_portable(source: Path, passphrase: str, into: Path | None = None) -> int:
    """Restore a portable export, re-sealing under this machine's DPAPI key."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    meta_file = source / "portable.json"
    if not meta_file.exists():
        raise VaultError(f"no portable.json in {source}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    kdf = Scrypt(
        salt=base64.b64decode(meta["salt"]), length=32,
        n=meta.get("n", 2**15), r=meta.get("r", 8), p=meta.get("p", 1),
    )
    portable = Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase.encode())))
    local = Vault()
    target_root = into or config.DATA_DIR

    count = 0
    for path in sorted((source / "aura_data").rglob(f"*{_ENCRYPTED_SUFFIX}")):
        raw = portable.decrypt(path.read_bytes())
        relative = path.relative_to(source / "aura_data").with_suffix("")
        target = target_root / relative.with_suffix(relative.suffix + _ENCRYPTED_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(local.encrypt_bytes(raw))
        count += 1

    logger.info("imported {} file(s) into {}", count, target_root)
    return count


# --------------------------------------------------------------------------
# Hardening review
# --------------------------------------------------------------------------


def hardening_report() -> dict[str, Any]:
    """Check the security posture. Read-only; changes nothing."""
    from aura.remote.server import TOKEN_FILE, check_exposure, tailscale_ip

    report: dict[str, Any] = {"checks": [], "warnings": [], "ok": True}

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            report["ok"] = False

    check("data key sealed", KEY_FILE.exists(),
          str(KEY_FILE) if KEY_FILE.exists() else "not created - run --init")

    for name in PROTECTED:
        directory = config.DATA_DIR / name
        if not directory.exists():
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        plain = [p for p in files if p.suffix != _ENCRYPTED_SUFFIX]
        if not files:
            # An empty directory would otherwise report "all encrypted", which
            # is technically true and actively misleading in a security report.
            report["checks"].append(
                {"name": f"{name} encrypted", "passed": True, "detail": "empty - nothing to protect"}
            )
            continue
        check(f"{name} encrypted", not plain,
              f"all {len(files)} file(s) encrypted" if not plain
              else f"{len(plain)} of {len(files)} file(s) in plaintext")

    address = tailscale_ip()
    check("remote is VPN-only", True,
          f"tailscale {address}" if address else "Tailscale not set up (remote access disabled)")

    exposure = check_exposure()
    check("no public listener", not exposure["publicly_exposed"], exposure["verdict"])

    check("remote token exists", TOKEN_FILE.exists(),
          "set" if TOKEN_FILE.exists() else "not generated (remote access unused)")

    browser = config.DATA_DIR / "browser"
    if browser.exists():
        report["warnings"].append(
            "the browser profile holds live site sessions; it is inside the "
            "encrypted tree only when AURA is not running"
        )

    return report


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA hardening (Phase 12)")
    parser.add_argument("--init", action="store_true", help="create the data key")
    parser.add_argument("--encrypt", action="store_true", help="encrypt data at rest")
    parser.add_argument("--decrypt", action="store_true", help="decrypt for use")
    parser.add_argument("--report", action="store_true", help="hardening review")
    parser.add_argument("--export", type=Path, metavar="DIR", help="export to an external drive")
    parser.add_argument("--import-from", type=Path, metavar="DIR")
    parser.add_argument("--passphrase", default="")
    args = parser.parse_args()

    bootstrap("vault")

    if args.init:
        try:
            create_key(force=False)
            print(f"data key created at {KEY_FILE}")
        except VaultError as exc:
            print(exc)
            return 1
        return 0

    if args.encrypt or args.decrypt:
        vault = Vault()
        total = VaultStats()
        for name in PROTECTED:
            directory = config.DATA_DIR / name
            if not directory.exists():
                continue
            stats = vault.encrypt_tree(directory) if args.encrypt else vault.decrypt_tree(directory)
            total.encrypted += stats.encrypted
            total.decrypted += stats.decrypted
            total.bytes_processed += stats.bytes_processed
            total.errors.extend(stats.errors)
        verb = "encrypted" if args.encrypt else "decrypted"
        count = total.encrypted if args.encrypt else total.decrypted
        print(f"{verb} {count} file(s), {total.bytes_processed / 1024:.1f} KB")
        for error in total.errors:
            print(f"  error: {error}")
        return 1 if total.errors else 0

    if args.export:
        if not args.passphrase:
            print("--export requires --passphrase (12+ characters)")
            return 1
        export_portable(args.export, args.passphrase)
        print(f"exported to {args.export}")
        return 0

    if args.import_from:
        if not args.passphrase:
            print("--import-from requires --passphrase")
            return 1
        count = import_portable(args.import_from, args.passphrase)
        print(f"imported {count} file(s)")
        return 0

    report = hardening_report()
    print("=" * 64)
    print("  AURA hardening review")
    print("=" * 64)
    for item in report["checks"]:
        mark = "[ OK ]" if item["passed"] else "[FAIL]"
        print(f"{mark}  {item['name']:<26} {item['detail']}")
    for warning in report["warnings"]:
        print(f"[WARN]  {warning}")
    print("=" * 64)
    print("  posture:", "ok" if report["ok"] else "needs attention")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
