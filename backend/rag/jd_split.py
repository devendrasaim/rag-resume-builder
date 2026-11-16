"""Split a job post into tagged requirement lines (self-contained; part of the RAG pipeline).

    split(jd_text) -> [(mode, sentence), ...]   mode = req | pref | other (duty) | exposure

A heading line ("Qualifications", "Nice to have:") sets the mode for the lines under it;
a sentence's own wording ("...is a plus", "must have") can override it.
"""
import re

HEADS = [
    ("pref", r"prefer|nice[- ]to[- ]have|bonus|\bplus\b|desired|desirable|good to have|ideal(ly)?|stand out"),
    ("exposure", r"gain exposure|you.{0,3}ll (learn|gain|grow)|we.{0,3}ll teach"),
    ("req", r"require|qualification|must[- ]have|what you.{0,3}ll (need|bring)|looking for|you have|you bring"
            r"|minimum|basic|who you are|about you"),
    ("other", r"responsibilit|what you.{0,3}ll (do|build|work on)|day[- ]to[- ]day|the role|role description"
              r"|about (us|the)|benefits|perks|compensation|salary|company description"),
]
INLINE_PREF = re.compile(r"\b(a plus|is a plus|preferred|nice to have|bonus|ideally|desirable"
                         r"|(is|are|would be) (beneficial|an advantage|advantageous)|is helpful)\b", re.I)
INLINE_REQ = re.compile(r"\bmust (have|be|hold|possess)\b|\b(is|are) required\b|\bminimum (of )?\d", re.I)
ABBREV = r"\b(?:U\.S|e\.g|i\.e|etc|Inc|vs|approx|Ph\.D|B\.S|M\.S)\."


def _sentences(line: str) -> list[str]:
    safe = re.sub(ABBREV, lambda m: m.group(0).replace(".", "\x00"), line, flags=re.I)
    parts = re.split(r"(?<=[.!?])\s+|\s*•\s*", safe)
    return [p.replace("\x00", ".").strip(" -•*\t") for p in parts if p.strip(" -•*\t")]


def _heading_mode(text: str):
    return next((m for m, pat in HEADS if re.search(pat, text, re.I)), None)


def _is_heading(line: str) -> bool:
    words = line.strip("#*:•- \t").split()
    return len(line) < 70 and (line.endswith(":") or line.startswith("#") or line.isupper() or len(words) <= 6)


def split(jd: str) -> list[tuple[str, str]]:
    mode, out = "other", []
    for raw in jd.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_heading(line) and (m := _heading_mode(line)):
            mode = m
            continue
        for s in _sentences(line):
            # an inline lead-in like "Role Description The AI Developer..." switches mode too
            if (m := _heading_mode(" ".join(s.split()[:3]))) and len(s.split()) > 6:
                mode = m
            tag = mode
            if tag in ("req", "other") and INLINE_PREF.search(s):
                tag = "pref"
            elif tag in ("other", "exposure") and INLINE_REQ.search(s):
                tag = "req"
            out.append((tag, s))
    return out
