"""Neutralise active content in generated HTML reports.

The instructor report is an HTML document that the app builds from templates plus text that users
control (study title, team members, condition names, variable names, open-ended question text).
Some of that text used to be interpolated unescaped, so a title such as ``<script>...</script>`` or
a condition named ``<iframe src=...>`` ended up as live markup in a file the owner opens in a
browser (and that mail filters inspect). ``sanitize_report_html`` is a safety net applied to the
finished document: it re-emits the markup with dangerous elements and attributes turned into
harmless visible text. The report's own markup (headings, tables, inline SVG charts, a style
block, in-page anchors) passes through unchanged in meaning. ``harden_report_html`` adds a
Content-Security-Policy ``<meta>`` as a second line of defence enforced by the browser.

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

__version__ = "1.3.0.4"

# A second line of defence for the browser that opens the report: no script, no network, no frames,
# no forms, inline styles and inline (data:) images only. Enforced by the browser itself, so it holds
# even if a sanitiser bypass were ever found.
# ``frame-ancestors`` is not honoured in a <meta> policy (Chromium logs an error for it), so it is not part of this one.
CONTENT_SECURITY_POLICY = ("default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                           "form-action 'none'; base-uri 'none'")
# Policy strings written by earlier versions: a stored report carrying one keeps it as live markup instead of showing it as text.
_LEGACY_POLICIES = ("default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                    "form-action 'none'; base-uri 'none'; frame-ancestors 'none'",)

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
# Decide on the *opening* of every url( token only: no search for a closing parenthesis and no adjacent
# whitespace quantifiers, so the cost is linear in the input (the old pattern needed a closing ")" and backtracked
# polynomially on "url(" followed by whitespace or by many unterminated "url(" tokens). Fail closed: any url( whose
# argument does not start with data: or # becomes an unknown function, which the browser drops with its declaration.
_CSS_URL_OPEN_RE = re.compile(r"url\s*\(\s*(['\"]?)(?:\s*(data:|#))?", re.IGNORECASE)
_CSS_IMPORT_RE = re.compile(r"@import[^;{}]*(;|(?=[{}]|$))", re.IGNORECASE)
# CSS escapes (backslash + up to 6 hex digits, or backslash + any other char) let "url(" be spelled "ur\6c(",
# defeating a literal-text search; decode them first so detection sees what the browser's CSS tokenizer sees.
# Bounded repetition only (no nested quantifiers next to each other) - linear, not ReDoS-prone.
_CSS_ESCAPE_RE = re.compile(r"\\([0-9a-fA-F]{1,6})[ \t\n\r\f]?|\\(.)", re.DOTALL)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# CSS functions other than url() that can make the browser fetch a resource.
_CSS_FETCH_FN_RE = re.compile(r"(?<![\w-])(?:-webkit-)?(?:image-set|cross-fade|image|src)\s*\(", re.IGNORECASE)


def _decode_css_escapes(css: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        if m.group(1):
            try:
                return chr(int(m.group(1), 16))
            except (ValueError, OverflowError):
                return ""
        return m.group(2)
    return _CSS_ESCAPE_RE.sub(_repl, css)


def _literal_clean(css: str) -> str:
    """Neutralise resource-fetching and script constructs spelled out literally."""
    css = _CSS_IMPORT_RE.sub("", css)
    css = _CSS_URL_OPEN_RE.sub(lambda m: m.group(0) if m.group(2) else "blocked(" + m.group(1), css)
    css = _CSS_FETCH_FN_RE.sub("blocked(", css)
    css = re.sub(r"expression\s*\(", "blocked(", css, flags=re.IGNORECASE)
    css = re.sub(r"javascript\s*:", "blocked:", css, flags=re.IGNORECASE)
    css = re.sub(r"-moz-binding\s*:", "blocked:", css, flags=re.IGNORECASE)
    return css


def _clean_css(css: str) -> str:
    """Remove CSS constructs that fetch resources or run script."""
    cleaned = _literal_clean(css)
    # Fail closed: decoding CSS escapes and comments may reveal a construct the literal pass could not see
    # ("ur\6c(", "exp\72 ession(", "url/**/(", "@\69mport"). The literal source cannot be patched in place (an
    # escape maps to no fixed substring), so the whole declaration block is dropped. Harmless CSS - including
    # url(data:...) - is unchanged by a second literal pass over the decoded text and therefore kept.
    decoded = _decode_css_escapes(_CSS_COMMENT_RE.sub("", cleaned))
    if _literal_clean(decoded) != decoded:
        return "/* blocked */"
    return cleaned


def _meta_is_harmless(attrs: List[Tuple[str, Optional[str]]]) -> bool:
    lowered = {name.lower(): (value or "") for name, value in attrs}
    if set(lowered) == {"http-equiv", "content"}:  # our own policy tag survives a second pass; nothing else with http-equiv does
        return (lowered["http-equiv"].lower() == "content-security-policy"
                and lowered["content"] in (CONTENT_SECURITY_POLICY,) + _LEGACY_POLICIES)
    names = {name.lower() for name, _ in attrs}
    if "http-equiv" in names or "content" in names and "name" not in names:
        return False
    if names == {"charset"}:
        return True
    return names <= {"name", "content"} and any(
        name.lower() == "name" and (value or "").lower() == "viewport" for name, value in attrs)


_FOREIGN_ROOTS = frozenset({"svg", "math"})  # namespaces where the browser does NOT give <style> RAWTEXT parsing


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: List[str] = []
        self._neutralised: List[str] = []  # stack of blocked elements whose content must be escaped
        self._foreign_depth = 0  # >0 inside an <svg>/<math> subtree
        self.changed = False

    # ---- helpers -------------------------------------------------------------------
    def _escaped(self, text: Optional[str]) -> None:
        self.out.append(html.escape(text or "", quote=False))
        self.changed = True

    def _attr_text(self, name: str, value: Optional[str], tag: str) -> Optional[str]:
        lname = name.lower()
        if lname == "http-equiv" and tag == "meta":
            # _meta_is_harmless() only lets our own policy tag reach this point
            return f' {name}="{html.escape(value or "", quote=True)}"'
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
        # Outside foreign content, html.parser (like every HTML5 engine) gives <style> RAWTEXT parsing, so its
        # content is just CSS text (see handle_data) and _clean_css() is the right tool. Inside <svg>/<math>,
        # Chromium does NOT: it parses <style> children as ordinary markup, so e.g. <svg><style><img onerror=...>
        # is a live <img>, not CSS text - CSS-cleaning that content leaves the <img> start tag untouched. Block
        # <style> there instead so the whole subtree is escaped like any other blocked element's content.
        if tag == "style" and self._foreign_depth > 0:
            blocked = True
        if blocked:
            self._escaped(self.get_starttag_text())
            if tag not in _VOID_TAGS and not selfclosing:
                self._neutralised.append(tag)
            return
        if tag in _FOREIGN_ROOTS and not selfclosing:
            self._foreign_depth += 1
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
        if tag in _FOREIGN_ROOTS and self._foreign_depth > 0:
            self._foreign_depth -= 1
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
            # Text never carries markup. Newer Python versions hand tag-like text inside <title>/<textarea>
            # over as data (RCDATA), and an SVG <title> is an HTML integration point in browsers, so such
            # text could turn into a live element there. Escaping < and > makes the result independent of
            # the parser version and of the browser's context.
            self.out.append(data.replace("<", "&lt;").replace(">", "&gt;"))

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


_CSP_META = f'<meta http-equiv="Content-Security-Policy" content="{html.escape(CONTENT_SECURITY_POLICY, quote=True)}">'
_HEAD_OPEN_RE = re.compile(r"<head(\s[^>]*)?>", re.IGNORECASE)
_CSP_PRESENT_RE = re.compile(r"<meta\s[^>]*http-equiv\s*=\s*[\"']?content-security-policy", re.IGNORECASE)


def add_content_security_policy(document: str) -> str:
    """Insert the report's Content-Security-Policy as the first element of ``<head>`` (idempotent)."""
    if _CSP_PRESENT_RE.search(document[:4000]):
        return document
    match = _HEAD_OPEN_RE.search(document)
    if match:
        return document[:match.end()] + _CSP_META + document[match.end():]
    return _CSP_META + document


def harden_report_html(document: str) -> str:
    """Sanitise ``document`` and add the Content-Security-Policy. The one call for finished reports."""
    return add_content_security_policy(sanitize_report_html(document))
