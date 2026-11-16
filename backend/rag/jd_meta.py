"""Company + role from a job post, with a human-in-the-loop fallback.

    detect(jd, company="", role="") -> {company, role, company_options, role_options, needs_confirm, source}

* Both given by the user -> returned as-is, no LLM call.
* Otherwise one cached LLM call proposes up to 3 candidates per missing field, best first, with a confidence.
* A candidate is auto-accepted only if the model is "high" confidence AND the exact text appears in the post
  (Python check, same spirit as the bullet validator). Anything else -> needs_confirm, and the UI shows chips.
"""
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_JD_CHARS = 6000          # names sit near the top; no need to send a whole long post

PROMPT = """You read a job post and name the hiring company and the job title.

Rules:
- Copy names exactly as written in the post. Never invent or expand a name.
- company: the employer the person would work for. If the post is from a recruiter or staffing agency
  for an unnamed client, the client is unknown: give no candidates or only the agency with low confidence.
- role: the job title as written (e.g. "Senior Machine Learning Engineer"), without location or req id.
- Give up to 3 candidates per field, best first. Fewer is fine. An empty list means the post does not say.
- confidence is "high" only when the post states it plainly (a title line, "About <Company>", "<Company> is hiring").
"""


class Field_(BaseModel):
    candidates: list[str] = Field(default_factory=list, description="best first, max 3, exact text from the post")
    confidence: Literal["high", "medium", "low"] = "low"


class JDMeta(BaseModel):
    company: Field_
    role: Field_


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip(" \t\n-|·,:")


def _in_post(name: str, jd: str) -> bool:
    return bool(name) and _clean(name).lower() in re.sub(r"\s+", " ", jd).lower()


def _options(f: Field_, jd: str) -> list[str]:
    seen, out = set(), []
    for c in f.candidates:
        c = _clean(c)
        if c and c.lower() not in seen and len(c) <= 120:
            seen.add(c.lower())
            out.append(c)
    # names actually found in the post first (stable otherwise)
    return sorted(out, key=lambda c: not _in_post(c, jd))[:3]


def detect(jd: str, company: str = "", role: str = "") -> dict:
    company, role = _clean(company), _clean(role)
    if company and role:
        return {"company": company, "role": role, "company_options": [], "role_options": [],
                "needs_confirm": False, "source": "given"}

    from writer import structured
    out = structured(JDMeta, [{"role": "system", "content": PROMPT},
                              {"role": "user", "content": f"JOB POST\n\n{jd[:MAX_JD_CHARS]}"}], tag="detect: company + role")

    res = {"company": company, "role": role, "company_options": [], "role_options": [],
           "needs_confirm": False, "source": "llm"}
    for key, given in (("company", company), ("role", role)):
        if given:
            continue                                     # the user's value always wins
        f: Field_ = getattr(out, key)
        opts = _options(f, jd)
        res[f"{key}_options"] = opts
        sure = f.confidence == "high" and len(opts) >= 1 and _in_post(opts[0], jd)
        if sure:
            res[key] = opts[0]
        else:
            res["needs_confirm"] = True
    return res
