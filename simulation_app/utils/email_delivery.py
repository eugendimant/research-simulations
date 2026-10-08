"""Reliable, observable SMTP delivery for the instructor notification and the other app emails.

Why this module exists
----------------------
The instructor notification used to be fire-and-forget: when SMTP was not configured, or the
send failed for any reason (message too large, transient server error, refused recipient), the
code path was a bare ``pass`` and nothing was logged or shown anywhere, so a missing email could
not be diagnosed. This module keeps the message building, the send (with retries), the size
handling and the delivery log in one place that does not depend on Streamlit, so it can be unit
tested and reused by every email entry point in ``app.py``.

What it does
------------
* ``load_smtp_config`` reads the SMTP secrets through a getter (never prints them).
* ``parse_recipients`` accepts one or several addresses (comma/semicolon separated, optionally
  with display names) so the notification can also go to a second inbox.
* ``build_message`` builds a standards-compliant message: text plus optional HTML alternative,
  a proper MIME type per attachment (``text/html`` for a report, not a generic
  ``application/octet-stream``), ``Date``, ``Message-ID`` and ``Auto-Submitted`` headers, and
  header values stripped of line breaks.
* ``fit_attachments`` degrades gracefully when a message would be too large: it first swaps an
  attachment for a smaller alternative (for example a ZIP without large source files), then drops
  the lowest-priority attachment, and reports what was left out.
* ``send_with_retries`` retries transient failures with backoff and classifies permanent ones.
* ``deliver`` ties these together, shrinks the budget once more if the server answers "message
  too large", and appends one JSON line per delivery to a log that the admin dashboard shows.

Nothing here logs a password, and only a masked recipient address is written to the log.
"""
from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import smtplib
import socket
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.utils import format_datetime, formataddr, getaddresses, make_msgid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "1.3.0.4"

logger = logging.getLogger(__name__)

# Gmail and Google Workspace refuse messages above ~25 MB INCLUDING the base64 encoding, and many
# Exchange tenants are stricter. 15 MB of encoded message leaves a wide margin; override with the
# secret EMAIL_MAX_MESSAGE_MB when the mail provider is known to allow more.
DEFAULT_MAX_MESSAGE_BYTES = 15 * 1024 * 1024
# base64 plus MIME line breaks inflate binary data by about 37%.
_ENCODING_OVERHEAD = 1.37
_MAX_RECIPIENTS = 10
_TRANSIENT_SMTP_CODES = frozenset({421, 450, 451, 452, 454})


def _is_transient_code(code: Optional[int]) -> bool:
    return code is not None and (code in _TRANSIENT_SMTP_CODES or 400 <= int(code) < 500)
_ADDRESS_RE = re.compile(r"^[^@\s<>,;\"]+@[^@\s<>,;\"]+\.[^@\s<>,;\"]+$")
_LOG_LOCK = threading.Lock()
_MAX_LOG_LINES = 400

# Extension -> MIME type for what the app attaches (stdlib mimetypes does not know .md/.jl/.sps/.do)
_MIME_BY_EXT = {
    ".zip": "application/zip",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".r": "text/plain",
    ".py": "text/plain",
    ".jl": "text/plain",
    ".sps": "text/plain",
    ".do": "text/plain",
}


# --------------------------------------------------------------------------------------
# Configuration and addresses
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SMTPConfig:
    """SMTP settings. ``public_summary`` is the only representation meant for display."""

    server: str = ""
    port: int = 587
    username: str = ""
    password: str = field(default="", repr=False)
    from_email: str = ""
    from_name: str = "Behavioral Experiment Simulation Tool"
    use_tls: bool = True
    timeout: float = 25.0
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES

    @property
    def configured(self) -> bool:
        return bool(self.server and self.username and self.password)

    @property
    def sender_address(self) -> str:
        return self.from_email or self.username

    def public_summary(self) -> Dict[str, Any]:
        return {
            "server_set": bool(self.server),
            "server_host": self.server,
            "port": self.port,
            "username_set": bool(self.username),
            "password_set": bool(self.password),
            "from_set": bool(self.sender_address),
            "tls": self.use_tls,
            "max_message_mb": round(self.max_message_bytes / (1024 * 1024), 1),
        }


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def load_smtp_config(get_secret: Callable[[str, Any], Any]) -> SMTPConfig:
    """Build the SMTP configuration from a ``get_secret(name, default)`` callable."""
    try:
        port = int(get_secret("SMTP_PORT", 587) or 587)
    except (TypeError, ValueError):
        port = 587
    try:
        max_mb = float(get_secret("EMAIL_MAX_MESSAGE_MB", 0) or 0)
    except (TypeError, ValueError):
        max_mb = 0.0
    username = str(get_secret("SMTP_USERNAME", "") or "").strip()
    return SMTPConfig(
        server=str(get_secret("SMTP_SERVER", "") or "").strip(),
        port=port,
        username=username,
        password=str(get_secret("SMTP_PASSWORD", "") or ""),
        from_email=str(get_secret("SMTP_FROM_EMAIL", "") or username).strip(),
        from_name=str(get_secret("SMTP_FROM_NAME", "") or "Behavioral Experiment Simulation Tool").strip(),
        use_tls=_as_bool(get_secret("SMTP_USE_TLS", True), True),
        max_message_bytes=int(max_mb * 1024 * 1024) if max_mb >= 1 else DEFAULT_MAX_MESSAGE_BYTES,
    )


_FREE_MAIL_DOMAINS = frozenset({"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "yahoo.com",
                                "icloud.com", "aol.com", "proton.me", "protonmail.com", "gmx.com", "mail.com"})


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1].strip().strip(">").lower() if "@" in address else ""


def _registrable(domain: str) -> str:
    """Last two labels of a host or domain name (good enough to compare mail setups)."""
    labels = [part for part in domain.lower().strip(".").split(".") if part]
    return ".".join(labels[-2:])


def deliverability_warnings(config: SMTPConfig, recipients: Sequence[str]) -> List[str]:
    """Plain-language findings about settings that make the receiving mail system hold, junk or
    silently drop a message that the SMTP server accepted (the "sent, but never arrived" case)."""
    findings: List[str] = []
    sender = config.sender_address
    from_domain = _registrable(_domain_of(sender))
    login_domain = _registrable(_domain_of(config.username))
    host_domain = _registrable(config.server)
    recipient_domains = {_registrable(_domain_of(r)) for r in recipients if _domain_of(r)} - {""}
    if from_domain and from_domain in recipient_domains and from_domain not in (login_domain, host_domain):
        findings.append(
            f"The From address ({mask_address(sender)}) is on the recipient's own domain ({from_domain}) but the message is "
            f"sent through {config.server or 'another server'}. Microsoft 365 and Google treat that as spoofing "
            "(SPF/DKIM/DMARC) and may quarantine or silently drop it. Use a From address that belongs to the sending "
            "account (SMTP_FROM_EMAIL = the SMTP username), or send through your institution's own mail server.")
    elif from_domain and login_domain and from_domain != login_domain:
        findings.append(
            f"The From address ({mask_address(sender)}) differs from the SMTP login ({mask_address(config.username)}). Many "
            "servers rewrite or reject that, and receivers may mark it as spoofed. Set SMTP_FROM_EMAIL to the login address "
            "unless the account is allowed to send as that address.")
    sender_domain = _domain_of(sender)
    if sender_domain in _FREE_MAIL_DOMAINS and recipient_domains and sender_domain not in {_domain_of(r) for r in recipients}:
        findings.append(
            f"The sender is a free-mail address ({sender_domain}). Institutional filters often put the first messages from "
            "such an address into Junk or quarantine: mark one message as 'Not junk' and add the sender address to the "
            "Safe Senders list in Outlook.")
    if config.server and not config.use_tls and config.port != 465:
        findings.append("TLS is switched off (SMTP_USE_TLS). Most servers refuse a login or deliver such mail as untrusted.")
    return findings


def parse_recipients(value: Any, limit: int = _MAX_RECIPIENTS) -> Tuple[List[str], List[str]]:
    """Split a secret such as ``"a@x.edu, Name <b@y.com>; c@z.org"`` into (valid, invalid).

    Valid addresses are lower-cased only for de-duplication; the original spelling is kept.
    """
    if value is None:
        return [], []
    text = ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    if len(text) > 2000:  # nobody types a 2 KB address list; refuse instead of cutting an address in half
        return [], [text[:40]]
    text = text.replace(";", ",").replace("\n", ",")
    valid: List[str] = []
    invalid: List[str] = []
    seen = set()
    try:
        parsed = getaddresses([text])
    except Exception:  # noqa: BLE001 - e.g. RecursionError on deeply nested comments
        return [], [text[:40]]
    for _name, addr in parsed:
        addr = (addr or "").strip()
        if not addr:
            continue
        if not _ADDRESS_RE.match(addr):
            invalid.append(addr)
            continue
        key = addr.lower()
        if key in seen:
            continue
        seen.add(key)
        valid.append(addr)
    return valid[:limit], invalid


def mask_address(address: str) -> str:
    """``edimant@sas.upenn.edu`` -> ``e***t@sas.upenn.edu`` (for logs and admin tables)."""
    local, _, domain = str(address).partition("@")
    if not domain:
        return "***"
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029\u202a-\u202e\u2066-\u2069]+")  # C0 controls, the other splitlines() separators, bidi overrides


def _clean_header(value: Any, limit: int = 250) -> str:
    """One physical line: no CR/LF/tab/VT/FF/NEL/LS/PS, so user-controlled text can neither inject
    headers nor make the message builder raise (a pasted U+2028 used to drop the whole notification)."""
    text = str(value or "").encode("utf-8", "replace").decode("utf-8")  # lone surrogates cannot be encoded
    return _CTRL_RE.sub(" ", text).strip()[:limit]


def _safe_filename(name: str) -> str:
    base = os.path.basename(str(name).replace("\\", "/")) or "attachment"
    return re.sub(r"[^A-Za-z0-9._ \-]", "_", base)[:120]


def guess_mime_type(filename: str) -> Tuple[str, str]:
    ext = os.path.splitext(str(filename).lower())[1]
    ctype = _MIME_BY_EXT.get(ext, "application/octet-stream")
    maintype, _, subtype = ctype.partition("/")
    return maintype, subtype


# --------------------------------------------------------------------------------------
# Message building and size handling
# --------------------------------------------------------------------------------------
@dataclass
class Attachment:
    """A file to attach. ``alternatives`` are smaller stand-ins tried before the file is dropped."""

    name: str
    data: bytes
    alternatives: Tuple["Attachment", ...] = ()
    protected: bool = False  # never dropped (it may still be replaced by an alternative)

    @property
    def encoded_size(self) -> int:
        return int(len(self.data) * _ENCODING_OVERHEAD) + 300  # + MIME part headers


def estimate_message_bytes(body_text: str, body_html: Optional[str], attachments: Sequence[Attachment]) -> int:
    body = len(body_text.encode("utf-8")) + (len(body_html.encode("utf-8")) if body_html else 0)
    return int(body * 1.05) + 2000 + sum(a.encoded_size for a in attachments)


def fit_attachments(
    slots: Sequence[Attachment], budget_bytes: int, fixed_bytes: int = 0
) -> Tuple[List[Attachment], List[Dict[str, Any]]]:
    """Choose what to attach so the encoded message stays within ``budget_bytes``.

    ``slots`` are in priority order (most important first). While the estimate is too large, one
    reduction is applied per round: a slot is either replaced by its next smaller alternative or,
    when it has none left and is not ``protected``, dropped. The slot chosen is the one with the
    best ``bytes saved x (priority rank + 1)`` score, so a large low-priority file goes first and
    small files are left alone. Returns (kept, notes) where notes say what was reduced or left out.
    """
    current: List[Optional[Attachment]] = list(slots)
    used_alt = [0] * len(current)  # alternatives already used per slot
    notes: List[Dict[str, Any]] = []

    def total() -> int:
        return fixed_bytes + sum(a.encoded_size for a in current if a is not None)

    for _ in range(len(slots) * 4 + 4):
        if total() <= budget_bytes:
            break
        best: Optional[Tuple[float, int, str]] = None  # (score, slot index, action)
        for i, att in enumerate(current):
            if att is None:
                continue
            slot = slots[i]
            if used_alt[i] < len(slot.alternatives):
                saved = att.encoded_size - slot.alternatives[used_alt[i]].encoded_size
                action = "replace"
            elif not slot.protected:
                saved = att.encoded_size
                action = "drop"
            else:
                continue
            if saved <= 0:
                continue
            score = saved * (i + 1)
            if best is None or score > best[0]:
                best = (score, i, action)
        if best is None:
            break  # nothing left to reduce; the send may still fail and the caller will report it
        _, i, action = best
        att = current[i]
        assert att is not None
        if action == "replace":
            repl = slots[i].alternatives[used_alt[i]]
            notes.append({"name": slots[i].name, "action": "replaced", "from_bytes": len(att.data),
                          "to_bytes": len(repl.data), "with": repl.name})
            current[i] = repl
            used_alt[i] += 1
        else:
            notes.append({"name": slots[i].name, "action": "omitted", "bytes": len(att.data)})
            current[i] = None
    return [a for a in current if a is not None], notes


def harden_html_attachment(data: bytes) -> bytes:
    """Neutralise active content in an HTML report and add a restrictive Content-Security-Policy.

    Idempotent. Used for stored reports (re-sent from the archive) that may predate the sanitiser.
    Falls back to the unchanged bytes when the sanitiser module is unavailable."""
    try:
        try:
            from .html_safety import harden_report_html
        except ImportError:
            from html_safety import harden_report_html  # type: ignore[no-redef]
        return harden_report_html(data.decode("utf-8", "replace")).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - never block a notification on the hardening step
        logger.warning("Could not harden the HTML attachment: %s", exc)
        return data


def _scrub_text(value: Any) -> str:
    """Make text encodable: lone surrogates (from a broken paste) would otherwise make the whole message fail."""
    return str(value or "").encode("utf-8", "replace").decode("utf-8")


def _address_domain(address: str) -> str:
    return address.partition("@")[2] or "localhost"


def build_message(
    *,
    from_email: str,
    from_name: str,
    recipients: Sequence[str],
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    attachments: Sequence[Attachment] = (),
    extra_headers: Optional[Dict[str, str]] = None,
) -> EmailMessage:
    """Build the MIME message. Text plus optional HTML alternative; attachments get real types."""
    msg = EmailMessage(policy=policy.default.clone(cte_type="7bit"))
    msg["From"] = formataddr((_clean_header(from_name, 100), from_email)) if from_name else from_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = _clean_header(subject)
    msg["Date"] = format_datetime(datetime.now(timezone.utc))
    msg["Message-ID"] = make_msgid(domain=_address_domain(from_email))
    msg["Auto-Submitted"] = "auto-generated"
    msg["X-Mailer"] = "Behavioral Experiment Simulation Tool"
    for key, value in (extra_headers or {}).items():
        msg[_clean_header(key, 60)] = _clean_header(value)
    msg.set_content(_scrub_text(body_text))
    if body_html:
        msg.add_alternative(_scrub_text(body_html), subtype="html")
    for att in attachments:
        maintype, subtype = guess_mime_type(att.name)
        msg.add_attachment(att.data, maintype=maintype, subtype=subtype, filename=_safe_filename(att.name))
    return msg


# --------------------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------------------
@dataclass
class DeliveryResult:
    ok: bool
    message: str = ""
    attempts: int = 0
    elapsed_s: float = 0.0
    accepted: List[str] = field(default_factory=list)
    refused: Dict[str, str] = field(default_factory=dict)
    smtp_code: Optional[int] = None
    error_class: str = ""
    error_kind: str = ""  # transient | permanent | size | auth | recipient | config
    error_detail: str = ""
    message_id: str = ""
    message_bytes: int = 0
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    omitted: List[Dict[str, Any]] = field(default_factory=list)
    possible_duplicate: bool = False  # a retry followed a dropped connection: the server may have kept the first copy


_EMAIL_IN_TEXT = re.compile(r"[\w.+'%-]+@[\w.-]+\.[A-Za-z]{2,}")


def _clean_detail(text: Any, secret: str = "") -> str:
    out = str(text or "")
    if secret:
        out = out.replace(secret, "***")
    out = _EMAIL_IN_TEXT.sub(lambda m: mask_address(m.group(0)), out)
    return re.sub(r"\s+", " ", out).strip()[:300]


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value or "")


_SIZE_RE = re.compile(r"message size|too large|too big|size limit|size exceeds|exceed\w* .{0,40}(size|limit)", re.IGNORECASE)
_MAILBOX_FULL_RE = re.compile(r"over quota|mailbox (is )?full|insufficient storage|storage (limit|quota)", re.IGNORECASE)
_QUOTA_RE = re.compile(r"daily user sending|sending (quota|limit)|quota exceeded|rate limit|too many (login|messages|connections|"
                       r"recipients|emails)|limit exceeded", re.IGNORECASE)


def classify_error(exc: BaseException) -> Tuple[str, Optional[int], str]:
    """Return (kind, smtp_code, detail) for an exception raised while sending.

    Kinds: ``transient`` (retry), ``size`` (message too large), ``quota`` (sending quota or rate
    limit: do not hammer the server), ``auth``, ``recipient``, ``config`` and ``permanent``.
    """
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        code = getattr(exc, "smtp_code", None)
        detail = _decode(getattr(exc, "smtp_error", ""))
        if _QUOTA_RE.search(detail):
            return "quota", code, detail
        return ("transient" if _is_transient_code(code) else "auth"), code, detail
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        detail = "; ".join(f"{k}: {_decode(v[1])}" for k, v in (exc.recipients or {}).items())
        code = next((v[0] for v in (exc.recipients or {}).values()), None)
        if _QUOTA_RE.search(detail):
            return "quota", code, detail
        return ("transient" if _is_transient_code(code) else "recipient"), code, detail
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return "config", None, str(exc)
    if isinstance(exc, smtplib.SMTPResponseException):  # SenderRefused, DataError, ConnectError, HeloError...
        code = exc.smtp_code
        detail = _decode(exc.smtp_error)
        if _MAILBOX_FULL_RE.search(detail):
            return "recipient", code, detail
        if _SIZE_RE.search(detail) or (code == 552 and not _QUOTA_RE.search(detail)):
            return "size", code, detail
        if _QUOTA_RE.search(detail):
            return "quota", code, detail
        if _is_transient_code(code):
            return "transient", code, detail
        return "permanent", code, detail
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return "transient", None, str(exc)
    if isinstance(exc, smtplib.SMTPException):  # e.g. "SMTP AUTH extension not supported by server"
        return "config", None, str(exc)
    if isinstance(exc, ssl.SSLCertVerificationError):
        return "config", None, str(exc)
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, OSError)):
        return "transient", None, str(exc)
    return "permanent", None, f"{type(exc).__name__}: {exc}"


def _connect(config: SMTPConfig, smtp_factory: Optional[Callable[..., Any]] = None):
    """Open an authenticated connection (SSL on 465, STARTTLS otherwise when enabled)."""
    if smtp_factory is not None:
        factory_ssl = factory_plain = smtp_factory
    else:
        factory_ssl, factory_plain = smtplib.SMTP_SSL, smtplib.SMTP
    if config.port == 465:
        server = factory_ssl(config.server, config.port, timeout=config.timeout, context=ssl.create_default_context())
    else:
        server = factory_plain(config.server, config.port, timeout=config.timeout)
    try:
        if config.port != 465:
            server.ehlo()
            if config.use_tls:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
        server.login(config.username, config.password)
    except Exception:
        try:
            server.close()
        except Exception:
            pass
        raise
    return server


def send_with_retries(
    config: SMTPConfig,
    msg: EmailMessage,
    recipients: Sequence[str],
    *,
    max_attempts: int = 3,
    deadline_s: float = 120.0,
    backoff_s: Sequence[float] = (2.0, 6.0),
    smtp_factory: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> DeliveryResult:
    """Send ``msg``; retry transient failures with backoff; classify permanent ones."""
    started = time.monotonic()
    result = DeliveryResult(ok=False, message_id=_clean_header(msg.get("Message-ID", "")))
    try:
        result.message_bytes = len(msg.as_bytes())
    except Exception:  # sizing must never prevent the attempt
        result.message_bytes = 0
    ambiguous_failure = False
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        server = None
        handed_over = False
        try:
            server = _connect(config, smtp_factory)
            handed_over = True  # from here on a dropped connection can hide an accepted message
            refused = server.send_message(msg, from_addr=config.sender_address, to_addrs=list(recipients)) or {}
            try:
                server.quit()
            except Exception:  # the message was already accepted
                pass
            result.refused = {k: _clean_detail(_decode(v[1]) if isinstance(v, (tuple, list)) and len(v) > 1 else v)
                              for k, v in refused.items()}
            result.accepted = [r for r in recipients if r not in refused]
            result.ok = bool(result.accepted)
            result.possible_duplicate = bool(result.ok and attempt > 1 and ambiguous_failure)
            result.error_kind = "" if result.ok else "recipient"
            result.message = "Email sent successfully!" if result.ok else "No recipient accepted the message."
            result.elapsed_s = round(time.monotonic() - started, 2)
            return result
        except Exception as exc:  # noqa: BLE001 - classified below
            if server is not None:
                try:
                    server.close()
                except Exception:
                    pass
            kind, code, detail = classify_error(exc)
            if kind == "transient" and code is None and handed_over:  # disconnect/timeout after the hand-over: outcome unknown
                ambiguous_failure = True
            result.error_class = type(exc).__name__
            result.error_kind = kind
            result.smtp_code = code
            result.error_detail = _clean_detail(detail, config.password)
            logger.warning("Email attempt %d/%d failed (%s, %s %s): %s", attempt, max_attempts, result.error_class,
                           kind, code or "", result.error_detail)
            if kind != "transient" or attempt >= max_attempts:
                break
            wait = backoff_s[min(attempt - 1, len(backoff_s) - 1)] if backoff_s else 0.0
            if time.monotonic() - started + wait > deadline_s:
                break
            sleep(wait)
    result.elapsed_s = round(time.monotonic() - started, 2)
    result.message = {
        "auth": "Authentication failed. For Gmail/Google Workspace: use an App Password (not your regular password). "
                "Go to Google Account > Security > App passwords.",
        "recipient": "The mail server refused the recipient address.",
        "size": "The message was too large for the mail server.",
        "quota": "The mail account hit a sending limit (quota or rate limit). Try again later.",
        "config": "The mail server connection is misconfigured (TLS or port).",
    }.get(result.error_kind, "Email could not be sent. Please check the configuration and try again.")
    return result


# --------------------------------------------------------------------------------------
# Delivery log
# --------------------------------------------------------------------------------------
def record_delivery(
    log_path: Optional[Path],
    result: DeliveryResult,
    *,
    kind: str,
    subject: str,
    recipients: Sequence[str],
    host: str = "",
) -> Dict[str, Any]:
    """Append one JSON line describing a delivery attempt. Never raises."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "ok": bool(result.ok),
        "attempts": result.attempts,
        "elapsed_s": result.elapsed_s,
        "subject": _clean_header(subject, 160),
        "to": [mask_address(r) for r in recipients],
        "accepted": [mask_address(r) for r in result.accepted],
        "refused": {mask_address(k): v for k, v in result.refused.items()},
        "possible_duplicate": bool(result.possible_duplicate),
        "smtp_host": host,
        "smtp_code": result.smtp_code,
        "error_kind": result.error_kind,
        "error_class": result.error_class,
        "error": result.error_detail,
        "message_id": result.message_id,
        "message_bytes": result.message_bytes,
        "attachments": result.attachments,
        "omitted": result.omitted,
    }
    level = logging.INFO if result.ok else logging.ERROR
    logger.log(level, "Email %s delivery %s: attempts=%s elapsed=%ss size=%s kind=%s %s",
               kind, "OK" if result.ok else "FAILED", result.attempts, result.elapsed_s, result.message_bytes,
               result.error_kind, result.error_detail)
    # The log file lives on an ephemeral disk and the INFO level is usually filtered out of the hosting
    # platform's logs, so mirror one masked line per delivery to stderr where "Manage app" shows it.
    try:
        print(f"EMAIL-DELIVERY {kind} {'OK' if result.ok else 'FAILED'} to={[mask_address(r) for r in recipients]} "
              f"attempts={result.attempts} seconds={result.elapsed_s} bytes={result.message_bytes} "
              f"problem={result.error_kind or '-'}:{result.smtp_code or '-'} {result.error_detail} "
              f"refused={len(result.refused)} dup={'maybe' if result.possible_duplicate else 'no'} "
              f"message-id={result.message_id}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - logging must never break a send
        pass
    if log_path is None:
        return entry
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=True, default=str)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if path.stat().st_size > 400_000:  # keep the file small: trim to the newest lines
                lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln][-_MAX_LOG_LINES:]
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:  # the log must never break a send
        logger.warning("Could not write the email delivery log: %s", exc)
    return entry


def read_delivery_log(log_path: Optional[Path], limit: int = 50) -> List[Dict[str, Any]]:
    """Newest-first list of logged deliveries (empty when there is no log yet)."""
    if log_path is None:
        return []
    try:
        path = Path(log_path)
        if not path.exists():
            return []
        with _LOG_LOCK:
            lines = path.read_text(encoding="utf-8").split("\n")
    except Exception:
        return []
    entries: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
        if len(entries) >= limit:
            break
    return entries


# --------------------------------------------------------------------------------------
# One entry point
# --------------------------------------------------------------------------------------
def deliver(
    config: SMTPConfig,
    recipients: Sequence[str],
    subject: str,
    body_text: str,
    *,
    body_html: Optional[str] = None,
    attachments: Sequence[Attachment] = (),
    kind: str = "email",
    log_path: Optional[Path] = None,
    max_attempts: int = 3,
    backoff_s: Sequence[float] = (2.0, 6.0),
    deadline_s: float = 120.0,
    smtp_factory: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
    extra_headers: Optional[Dict[str, str]] = None,
) -> DeliveryResult:
    """Build, size-fit, send (with retries) and log one email. Never raises.

    When the server rejects the size, the budget is halved once; if that fails too, a last message
    without attachments goes out so the text still arrives."""
    recipients = list(recipients)
    subject, body_text = _scrub_text(subject), _scrub_text(body_text)
    body_html = _scrub_text(body_html) if body_html else body_html
    omitted: List[Dict[str, Any]] = []
    try:
        if not config.configured:
            result = DeliveryResult(ok=False, message="Email not configured. Contact the administrator.",
                                    error_kind="config", error_class="NotConfigured",
                                    error_detail="SMTP_SERVER / SMTP_USERNAME / SMTP_PASSWORD are not all set")
        elif not recipients:
            result = DeliveryResult(ok=False, message="No valid recipient address.", error_kind="recipient",
                                    error_class="NoRecipient", error_detail="the recipient secret is empty or invalid")
        else:
            budget = config.max_message_bytes
            stage = 0  # 0 normal, 1 halved budget, 2 protected attachments only, 3 text only
            previous_plan: Optional[Tuple[Tuple[str, int], ...]] = None
            while True:
                fixed = estimate_message_bytes(body_text, body_html, [])
                if stage >= 3:
                    kept = []
                    notes = [{"name": a.name, "action": "omitted", "bytes": len(a.data)} for a in attachments]
                elif stage == 2:
                    kept = [a for a in attachments if a.protected]
                    notes = [{"name": a.name, "action": "omitted", "bytes": len(a.data)} for a in attachments if not a.protected]
                else:
                    kept, notes = fit_attachments(list(attachments), budget, fixed_bytes=fixed)
                plan = tuple((a.name, len(a.data)) for a in kept)
                if previous_plan is not None and plan == previous_plan and stage < 3:
                    stage += 1  # this stage would resend the same message: go straight to the next one
                    continue
                previous_plan = plan
                omitted = notes
                body, html_body = body_text, body_html
                if notes:
                    summary = "; ".join(
                        f"{n['name']} {'replaced by ' + n['with'] if n['action'] == 'replaced' else 'left out'}"
                        f" ({(n.get('from_bytes') or n.get('bytes') or 0) / 1_048_576:.1f} MB)" for n in notes)
                    note = f"NOTE: some attachments were reduced to keep the message deliverable: {summary}."
                    body = f"{note}\n\n{body_text}"
                    if body_html:
                        html_body = f"<p><b>{html_lib.escape(note)}</b></p>{body_html}"
                msg = build_message(from_email=config.sender_address, from_name=config.from_name, recipients=recipients,
                                    subject=subject, body_text=body, body_html=html_body, attachments=kept,
                                    extra_headers=extra_headers)
                result = send_with_retries(config, msg, recipients, max_attempts=max_attempts, backoff_s=backoff_s,
                                           deadline_s=deadline_s, smtp_factory=smtp_factory, sleep=sleep)
                result.attachments = [{"name": a.name, "bytes": len(a.data)} for a in kept]
                result.omitted = omitted
                if result.ok or result.error_kind != "size" or stage >= 3:
                    break
                if stage == 0:  # the server's limit is lower than assumed: halve the budget once
                    budget = max(1_000_000, int(min(budget, result.message_bytes or budget) * 0.5))
                stage += 1
    except Exception as exc:  # noqa: BLE001 - delivery must never crash the caller
        logger.exception("Unexpected error while preparing an email")
        result = DeliveryResult(ok=False, message="Email could not be sent. Please check the configuration and try again.",
                                error_kind="permanent", error_class=type(exc).__name__,
                                error_detail=_clean_detail(exc, config.password))
    record_delivery(log_path, result, kind=kind, subject=subject, recipients=recipients, host=config.server)
    return result


# --------------------------------------------------------------------------------------
# Instructor notification content
# --------------------------------------------------------------------------------------
def _headline_effects(metadata: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    """Largest observed condition contrasts, preferring scale means over single items."""
    rows = [r for r in (metadata.get("effect_sizes_observed") or []) if isinstance(r, dict)]
    means = [r for r in rows if str(r.get("variable", "")).endswith("_mean")]
    pool = means or rows
    def _abs_d(r: Dict[str, Any]) -> float:
        try:
            return abs(float(r.get("cohens_d", 0.0)))
        except (TypeError, ValueError):
            return 0.0
    return sorted(pool, key=_abs_d, reverse=True)[:limit]


def _clip(value: Any, limit: int) -> str:
    """Bound student-typed text before it enters an email (one huge paste must not make the message undeliverable)."""
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + " [...]"


def compose_instructor_notification(
    *,
    title: str,
    team_name: str,
    team_members: str,
    generation_label: str,
    mode: str,
    metadata: Dict[str, Any],
    usage_summary: str,
    analysis_markdown: str,
    attachment_names: Sequence[str],
    zip_listing: Sequence[str] = (),
    max_inline_chars: int = 120_000,
    report_problem: str = "",
) -> Tuple[str, str, str]:
    """Return (subject, plain-text body, HTML body) for the instructor notification.

    The body repeats the headline numbers and the full Markdown analysis, so the message is
    useful even when a mail filter strips every attachment. All user-controlled text is escaped
    in the HTML part and stripped of line breaks in the subject.
    """
    title = _clip(title, 200)
    team_name = _clip(team_name, 200)
    team_members = _clip(team_members, 1500)
    generation_label = _clip(generation_label, 120)
    usage_summary = _clip(usage_summary, 4000)
    problem = _clean_header(str(report_problem or ""), 400)
    flag = "[REPORT ERROR] " if problem else ""
    subject = _clean_header(f"{flag}[Behavioral Simulation] Output ({mode or 'pilot'}) [{generation_label}] - {title}", 200)
    n = metadata.get("sample_size", "N/A")
    conditions = metadata.get("conditions") or []
    oe = metadata.get("open_ended_questions") or []
    excl = metadata.get("exclusion_summary") or metadata.get("exclusions") or {}
    effects = _headline_effects(metadata)
    analysis = str(analysis_markdown or "")
    clipped = len(analysis) > max_inline_chars
    if clipped:
        analysis = analysis[:max_inline_chars] + "\n\n[... truncated in the email body; the attachment has the full text ...]"

    facts = [
        ("Study", title), ("Team", team_name), ("Members", team_members),
        ("Generation method", generation_label), ("Sample size", f"N={n}"),
        ("Conditions", str(len(conditions))), ("Open-ended questions", str(len(oe))),
        ("Generated", metadata.get("generation_timestamp", "")), ("Run ID", metadata.get("run_id", "")),
    ]
    if metadata.get("oe_data_sources"):
        facts.append(("OE data sources", ", ".join(str(x) for x in metadata["oe_data_sources"])))
    if isinstance(excl, dict) and excl:
        facts.append(("Recommended exclusions", ", ".join(f"{k}={v}" for k, v in excl.items())))

    lines = ["INSTRUCTOR NOTIFICATION", "=" * 60, ""]
    if problem:
        lines += ["WARNING: the data were generated, but part of the instructor analysis could not be built:",
                  f"  {problem}", "  The attachments named below may be short placeholders. The data files in the student package are complete.", ""]
    lines += [f"{k}: {v}" for k, v in facts if str(v).strip()]
    lines += ["", "ATTACHMENTS (what students do NOT receive: the analyses)", ""]
    lines += [f"- {name}" for name in attachment_names]
    if effects:
        lines += ["", "LARGEST OBSERVED CONDITION DIFFERENCES (Cohen's d, condition 1 minus condition 2)", ""]
        for r in effects:
            lines.append(f"- {r.get('variable', '?')}: {r.get('condition_1', '?')} vs {r.get('condition_2', '?')}  "
                         f"d = {r.get('cohens_d', '?')}")
    if zip_listing:
        lines += ["", "FILES IN THE STUDENT ZIP", ""] + [f"- {name}" for name in zip_listing]
    if usage_summary:
        lines += ["", str(usage_summary)]
    lines += ["", "FULL ANALYSIS (same text as INSTRUCTOR_Detailed_Analysis.md, repeated here in case attachments are blocked)",
              "-" * 60, analysis]
    text = "\n".join(lines) + "\n"

    esc = html_lib.escape
    rows = "".join(f"<tr><td style='padding:2px 10px 2px 0'><b>{esc(str(k))}</b></td><td>{esc(str(v))}</td></tr>"
                   for k, v in facts if str(v).strip())
    eff_rows = "".join(
        f"<tr><td>{esc(str(r.get('variable', '?')))}</td><td>{esc(str(r.get('condition_1', '?')))} vs "
        f"{esc(str(r.get('condition_2', '?')))}</td><td style='text-align:right'>{esc(str(r.get('cohens_d', '?')))}</td></tr>"
        for r in effects)
    html_body = (
        "<html><body style='font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#111'>"
        f"<h2 style='margin:0 0 8px'>Instructor notification</h2>"
        + (f"<p style='color:#b00020'><b>Warning:</b> the data were generated, but part of the instructor analysis could not "
           f"be built: {esc(problem)}. The attachments may be short placeholders. The data files in the student package are complete.</p>"
           if problem else "")
        + f"<table>{rows}</table>"
        "<h3>Attachments</h3><ul>" + "".join(f"<li>{esc(name)}</li>" for name in attachment_names) + "</ul>"
        + (f"<h3>Largest observed condition differences</h3><table cellpadding='3' border='1' style='border-collapse:collapse'>"
           f"<tr><th>Variable</th><th>Contrast</th><th>Cohen's d (1 minus 2)</th></tr>{eff_rows}</table>" if effects else "")
        + "<h3>Full analysis</h3><p style='color:#555'>Same text as INSTRUCTOR_Detailed_Analysis.md.</p>"
        f"<pre style='white-space:pre-wrap;font-family:Consolas,monospace;font-size:12px'>{esc(analysis)}</pre>"
        "</body></html>"
    )
    return subject, text, html_body


# --------------------------------------------------------------------------------------
# Rate limiting for user-directed emails
# --------------------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter (thread-safe) so student-triggered emails cannot use up the mail
    account's daily quota that the instructor notification depends on."""

    def __init__(self, max_events: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_events = int(max_events)
        self.window_s = float(window_s)
        self._clock = clock
        self._events: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def remaining(self, key: str = "global") -> int:
        """How many more events ``key`` may record right now. Records nothing."""
        now = self._clock()
        with self._lock:
            used = len([t for t in self._events.get(key, []) if now - t < self.window_s])
        return max(0, self.max_events - used)

    def allow(self, key: str = "global") -> bool:
        """Record an event for ``key`` and return True when it is within the limit."""
        now = self._clock()
        with self._lock:
            events = [t for t in self._events.get(key, []) if now - t < self.window_s]
            if len(events) >= self.max_events:
                self._events[key] = events
                return False
            events.append(now)
            self._events[key] = events
            if len(self._events) > 5000:  # bound memory: forget keys whose events all expired
                self._events = {k: v for k, v in self._events.items() if v and now - v[-1] < self.window_s}
            return True


_SEND_SLOTS = threading.BoundedSemaphore(2)  # at most two SMTP deliveries at a time; the rest wait holding only their bytes
_APP_LIMITERS: Dict[str, RateLimiter] = {}
_APP_LIMITERS_LOCK = threading.Lock()


def shared_limiter(name: str, max_events: int, window_s: float = 3600.0) -> RateLimiter:
    """Process-wide limiter by name.

    ``app.py`` is re-executed on every Streamlit rerun, so state kept in the script itself would
    reset constantly; an imported module is cached, so this one lives as long as the server process.
    """
    with _APP_LIMITERS_LOCK:
        limiter = _APP_LIMITERS.get(name)
        if limiter is None or limiter.max_events != int(max_events) or limiter.window_s != float(window_s):
            limiter = RateLimiter(max_events, window_s)
            _APP_LIMITERS[name] = limiter
        return limiter


_FATAL_KINDS = frozenset({"config", "auth", "quota", "recipient"})


def deliver_instructor_package(
    config: SMTPConfig,
    recipients: Sequence[str],
    *,
    subject: str,
    text: str,
    html_body: Optional[str],
    slots: Sequence[Attachment],
    mode: str = "split",
    log_path: Optional[Path] = None,
    smtp_factory: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 5,
    backoff_s: Sequence[float] = (3.0, 10.0, 30.0, 90.0),
    deadline_s: float = 420.0,
) -> List[DeliveryResult]:
    """Send the instructor notification; returns one result per message. Never raises.

    ``split`` (default): message 1 has NO attachments (headline numbers and the full analysis in
    the body, the kind of message mail filters rarely hold); message 2 carries the attachments and
    is threaded to the first. If a filter holds or quarantines the attachment message, the analysis
    still arrives. ``single``: everything in one message. The notification runs in a background
    thread, so it retries for several minutes (a mail server that answers 421/450 for a while is
    waited out) instead of the few seconds a student-facing button can afford.
    """
    retry = dict(max_attempts=max_attempts, backoff_s=backoff_s, deadline_s=deadline_s)
    if str(mode).strip().lower() == "single":
        return [deliver(config, recipients, subject, text, body_html=html_body, attachments=slots, kind="instructor",
                        log_path=log_path, smtp_factory=smtp_factory, sleep=sleep, **retry)]
    first = deliver(config, recipients, subject, text, body_html=html_body, attachments=[], kind="instructor_summary",
                    log_path=log_path, smtp_factory=smtp_factory, sleep=sleep, **retry)
    package_subject = f"{subject} [attachments]"
    if not first.ok and first.error_kind in _FATAL_KINDS:
        skipped = DeliveryResult(ok=False, message="Skipped: the summary message failed for a reason that would stop "
                                 "the attachment message too.", error_kind="skipped", error_class="Skipped",
                                 error_detail=f"first message failed ({first.error_kind})")
        record_delivery(log_path, skipped, kind="instructor_package", subject=package_subject, recipients=list(recipients),
                        host=config.server)
        return [first, skipped]
    names = ", ".join(a.name for a in slots) or "(none)"
    facts = text[:1800].rsplit("\n", 1)[0] if len(text) > 1800 else text.split("FULL ANALYSIS", 1)[0]
    pkg_text = (f"Attachments for the previous message of this run ({subject}):\n{names}\n\n"
                "The headline numbers and the full analysis are in the body of that message.\n\n" + facts.strip() + "\n")
    headers = {"In-Reply-To": first.message_id, "References": first.message_id} if first.message_id else None
    second = deliver(config, recipients, package_subject, pkg_text, attachments=slots, kind="instructor_package",
                     log_path=log_path, smtp_factory=smtp_factory, sleep=sleep, extra_headers=headers, **retry)
    return [first, second]


TEST_CONTENT_CHOICES = ("Body only", "With a .md attachment", "With a .html report", "With a .zip", "With a 3 MB attachment",
                        "With a 10 MB attachment")


def build_test_attachments(choice: str) -> List[Attachment]:
    """Harmless sample attachments for the admin test email, to learn which type or size a mail
    filter delays: a markdown file, an inert HTML page (no scripts, no external resources), a small
    ZIP, and incompressible ZIPs of about 3 and 10 MB."""
    import io
    import zipfile

    def _zip(payload: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("sample.bin", payload)
        return buf.getvalue()

    choice = str(choice)
    if choice == "With a .md attachment":
        return [Attachment("test_analysis.md", b"# Test analysis\n\nThis is a harmless test file.\n")]
    if choice == "With a .html report":
        page = (b"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Test report</title></head><body>"
                b"<h1>Test report</h1><p>This is a harmless test page.</p></body></html>")
        return [Attachment("test_report.html", page)]
    if choice == "With a .zip":
        return [Attachment("test_package.zip", _zip(b"harmless sample"))]
    if choice == "With a 3 MB attachment":
        return [Attachment("test_3mb.zip", _zip(os.urandom(3 * 1024 * 1024)))]
    if choice == "With a 10 MB attachment":
        return [Attachment("test_10mb.zip", _zip(os.urandom(10 * 1024 * 1024)))]
    return []


def run_in_background(target: Callable[..., Any], *args: Any, name: str = "email-delivery", **kwargs: Any) -> threading.Thread:
    """Run ``target`` in a daemon thread that survives the Streamlit script run ending.

    Streamlit stops a script run when the browser disconnects; work started in a thread keeps
    going, so the instructor notification no longer depends on the tab staying open. The target
    must not call ``st.*``.
    """
    def _runner() -> None:
        try:
            with _SEND_SLOTS:
                target(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("Background email task failed")

    thread = threading.Thread(target=_runner, name=name, daemon=True)
    thread.start()
    return thread
