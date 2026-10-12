"""Text transformations used to remove one family of signals at a time (RQ4)."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import tokenize

_FENCE = re.compile(r"^\s*```.*$", re.M)
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+", re.M)
_EMPH = re.compile(r"(\*\*|__|\*|`)")
_WS = re.compile(r"\s+")


def strip_format(t: str) -> str:
    """Remove markdown structure: code fences, headers, bullets/numbering, bold/italics/backticks, line breaks."""
    t = _FENCE.sub(" ", t)
    t = _HEADER.sub("", t)
    t = _BULLET.sub("", t)
    t = _EMPH.sub("", t)
    return _WS.sub(" ", t).strip()


def lower_nopunct(t: str) -> str:
    """Words only: lowercase, no punctuation, no line breaks, no formatting."""
    return _WS.sub(" ", re.sub(r"[^a-z0-9' ]", " ", t.lower())).strip()


_SEG = re.compile(r"(?<=[.!?])\s+|\n+")


def drop_open_close(t: str) -> str:
    """Remove the first and the last sentence/line (greetings like 'Certainly!' and sign-offs).
    Everything in between is kept verbatim, so line breaks and markdown are unchanged."""
    s = t.strip()
    seps = list(_SEG.finditer(s))
    if not seps:
        return s
    if len(seps) == 1:  # two segments: drop only the opener
        return s[seps[0].end():]
    mid = s[seps[0].end(): seps[-1].start()]
    return mid if mid.strip() else s


FUNC = set(ENGLISH_STOP_WORDS) | {"i", "you", "it", "s", "t"}


def _rejoin(toks):
    return " ".join("\n" if x == "<nl>" else x for x in toks)


def skeleton(t: str) -> str:
    """Style skeleton: keep function words, punctuation, markdown symbols and line breaks;
    every content word becomes 'W'. Topic/content information is removed."""
    out = []
    for x in tokenize(t):
        if x == "<nl>" or not x[0].isalnum() or x.lower() in FUNC:
            out.append(x)
        else:
            out.append("W" if x[0].isalpha() else "D")
    return _rejoin(out)


def content_only(t: str) -> str:
    """Opposite of skeleton: lowercase content words only (no function words, punctuation, layout)."""
    return " ".join(x.lower() for x in tokenize(t) if x[0].isalnum() and x.lower() not in FUNC)


TRANSFORMS = {
    "no_format": strip_format,
    "words_only": lower_nopunct,
    "no_open_close": drop_open_close,
    "skeleton": skeleton,
    "content_only": content_only,
}
