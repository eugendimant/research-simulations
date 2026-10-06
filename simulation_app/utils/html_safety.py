"""Neutralise active content in generated HTML reports.

The instructor report is an HTML document that the app builds from templates plus text that users
control (study title, team members, condition names, variable names, open-ended question text).
Some of that text used to be interpolated unescaped, so a title such as ``<script>...</script>`` or
a condition named ``<iframe src=...>`` ended up as live markup in a file the owner opens in a
browser (and that mail filters inspect). ``sanitize_report_html`` is a safety net applied to the
finished document: it re-emits the markup with dangerous elements and attributes turned into
harmless visible text. The report's own markup (headings, tables, inline SVG charts, a style
block, in-page anchors) passes through unchanged in meaning.

Rules
-----
* Elements that can run code, load resources, submit data or redirect (script, iframe, object,
  embed, form controls, link, base, meta other than charset/viewport, SVG animation, foreignObject,
  media elements, template ...) are written out as escaped text, content included.
* ``img`` is kept only with an inline ``data:image/...;base64`` source.
* Event-handler attributes (``on*``), ``srcdoc``, ``http-equiv`` and URL attributes that are not an
  in-page ``#anchor`` (or an inline image data URI) are dropped.
* In ``style`` attributes and ``<style>`` blocks, ``@import``, ``url(...)`` to anything but a
  ``data:`` or ``#`` target, ``expression(`` and ``javascript:`` are removed.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

__version__ = "1.2.9.1"

_BLOCKED_TAGS = frozenset({
    "script", "iframe", "frame", "frameset", "object", "embed", "applet", "form", "input", "button", "select",
    "option", "optgroup", "textarea", "link", "base", "noscript", "template", "slot", "portal", "audio", "video",
    "source", "track", "canvas", "dialog", "foreignobject", "animate", "set", "animatemotion", "animatetransform",
    "use", "image", "feimage", "math", "xmp", "plaintext", "listing", "marquee", "bgsound", "isindex",
})
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "col", "area", "wbr", "input", "link", "base", "source", "track", "embed"})
_URL_ATTRS = frozenset({"href", "src", "xlink:href", "action", "formaction", "background", "poster", "data", "cite",
                        "longdesc", "usemap", "ping", "manifest"})
_DROPPED_ATTRS = frozenset({"srcdoc", "http-equiv", "formaction", "ping"})
_DATA_IMAGE_RE = re.compile(r"^\s*data:image/(png|jpe?g|gif|webp|svg\+xml);base64,[A-Za-z0-9+/=\s]+$", re.IGNORECASE)
_CSS_URL_RE = re.compile(r"url\s*\(\s*(['\"]?)\s*([^)'\"]*)\1\s*\)", re.IGNORECASE)
_CSS_IMPORT_RE = re.compile(r"@import[^;{}]*(;|(?=[{}]|$))", re.IGNORECASE)


def _clean_css(css: str) -> str:
    """Remove CSS constructs that fetch resources or run script."""
    css = _CSS_IMPORT_RE.sub("", css)

    def _url(match: "re.Match[str]") -> str:
        target = match.group(2).strip().lower()
        return match.group(0) if target.startswith(("data:", "#")) else "none"

    css = _CSS_URL_RE.sub(_url, css)
    css = re.sub(r"expression\s*\(", "blocked(", css, flags=re.IGNORECASE)
    css = re.sub(r"javascript\s*:", "blocked:", css, flags=re.IGNORECASE)
    css = re.sub(r"-moz-binding\s*:", "blocked:", css, flags=re.IGNORECASE)
    return css


def _meta_is_harmless(attrs: List[Tuple[str, Optional[str]]]) -> bool:
    names = {name.lower() for name, _ in attrs}
    if "http-equiv" in names or "content" in names and "name" not in names:
        return False
    if names == {"charset"}:
        return True
    return names <= {"name", "content"} and any(
        name.lower() == "name" and (value or "").lower() == "viewport" for name, value in attrs)


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: List[str] = []
        self._neutralised: List[str] = []  # stack of blocked elements whose content must be escaped
        self.changed = False

    # ---- helpers -------------------------------------------------------------------
    def _escaped(self, text: Optional[str]) -> None:
        self.out.append(html.escape(text or "", quote=False))
        self.changed = True

    def _attr_text(self, name: str, value: Optional[str], tag: str) -> Optional[str]:
        lname = name.lower()
        if lname.startswith("on") or lname in _DROPPED_ATTRS:
            self.changed = True
            return None
        if value is None:
            return f" {name}"
        if lname in _URL_ATTRS:
            stripped = re.sub(r"[\x00-\x20]+", "", value)
            if stripped.startswith("#"):
                return f' {name}="{html.escape(value, quote=True)}"'
            if tag == "img" and lname == "src" and _DATA_IMAGE_RE.match(value):
                return f' {name}="{html.escape(value, quote=True)}"'
            self.changed = True
            return None
        if lname == "style":
            cleaned = _clean_css(value)
            self.changed = self.changed or cleaned != value
            return f' {name}="{html.escape(cleaned, quote=True)}"'
        return f' {name}="{html.escape(value, quote=True)}"'

    def _start(self, tag: str, attrs: List[Tuple[str, Optional[str]]], selfclosing: bool) -> None:
        if self._neutralised:  # inside a blocked element: everything is text
            self._escaped(self.get_starttag_text())
            return
        blocked = tag in _BLOCKED_TAGS
        if tag == "meta" and not _meta_is_harmless(attrs):
            blocked = True
        if tag == "img" and not any(n.lower() == "src" and v and _DATA_IMAGE_RE.match(v) for n, v in attrs):
            blocked = True
        if blocked:
            self._escaped(self.get_starttag_text())
            if tag not in _VOID_TAGS and not selfclosing:
                self._neutralised.append(tag)
            return
        parts = [f"<{tag}"]
        for name, value in attrs:
            rendered = self._attr_text(name, value, tag)
            if rendered:
                parts.append(rendered)
        parts.append(" />" if selfclosing else ">")
        self.out.append("".join(parts))

    # ---- parser events -------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs, selfclosing=False)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs, selfclosing=True)

    def handle_endtag(self, tag: str) -> None:
        if self._neutralised:
            self._escaped(f"</{tag}>")
            if tag == self._neutralised[-1]:
                self._neutralised.pop()
            return
        if tag in _BLOCKED_TAGS and tag not in _VOID_TAGS:
            self._escaped(f"</{tag}>")
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._neutralised or self.cdata_elem == "script":
            self._escaped(data)  # content of a neutralised element is shown, never executed
        elif self.cdata_elem == "style":
            cleaned = _clean_css(data)
            self.changed = self.changed or cleaned != data
            self.out.append(cleaned)
        else:
            self.out.append(data)

    def handle_entityref(self, name: str) -> None:
        self.out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.out.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.changed = True  # comments can hide conditional-comment payloads; drop them

    def handle_decl(self, decl: str) -> None:
        self.out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self.changed = True  # processing instructions are not part of HTML5

    def unknown_decl(self, data: str) -> None:
        self.changed = True  # <![CDATA[ ... ]]> sections


def sanitize_report_html(document: str) -> str:
    """Return ``document`` with active content neutralised (see the module docstring)."""
    parser = _Sanitizer()
    parser.feed(str(document))
    parser.close()
    # anything still open when the input ended is flushed as text by close(); nothing else to do
    return "".join(parser.out)
