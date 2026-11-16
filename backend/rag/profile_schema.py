"""The profile: the one source of truth the RAG chunks, embeds and writes from.

Two ways in, one shape out (data/profile.json):

  JSON upload -> normalize()          your own file; extra keys are kept and chunked too
  PDF upload  -> extract_from_pdf()   pypdf text -> LLM copies facts into ExtractedProfile -> normalize()

normalize() fills every key the pipeline reads (so a sparse file never crashes ingest or the agent)
and rejects files that have nothing to build a resume from.
"""
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

# ---------- the shape (also the LLM's output schema for PDF extraction; strict = every field required) ----------


class Links(BaseModel):
    linkedin: str = Field(description="Full LinkedIn URL, or empty")
    github: str = Field(description="Full GitHub URL, or empty")
    portfolio: str = Field(description="Personal website / portfolio URL, or empty")
    other: list[str] = Field(description="Any other URLs on the resume")


class Personal(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str
    links: Links


class Job(BaseModel):
    company: str
    title: str
    location: str
    start: str = Field(description="As written, e.g. 'Jan 2023'")
    end: str = Field(description="As written, e.g. 'Present'")
    description: str = Field(description="One plain sentence on what the role/team did, copied or condensed from the resume")
    highlights: list[str] = Field(description="Every bullet under this role, copied verbatim")
    tech: list[str] = Field(description="Tools / languages named for this role")


class Project(BaseModel):
    name: str
    description: str
    highlights: list[str]
    tech: list[str]
    live: str = Field(description="Live URL if given, else empty")
    repo: str = Field(description="Repository URL if given, else empty")


class Research(BaseModel):
    title: str
    institution: str
    location: str
    start: str
    end: str
    advisor: str
    description: str
    highlights: list[str]
    tech: list[str]


class Education(BaseModel):
    institution: str
    location: str
    degree: str = Field(description="e.g. 'Bachelor of Science'")
    field: str = Field(description="e.g. 'Computer Science'")
    gpa: str
    start: str
    end: str


class Certification(BaseModel):
    name: str
    issuer: str
    date: str


class SkillGroup(BaseModel):
    category: str = Field(description="The resume's own category label, e.g. 'Languages'")
    items: list[str]


class ExtractedProfile(BaseModel):
    personal: Personal
    summary: str = Field(description="The resume's summary/objective, verbatim, or empty")
    skills: list[SkillGroup]
    work_experience: list[Job] = Field(description="Most recent first")
    projects: list[Project]
    research_experience: list[Research]
    education: list[Education]
    certifications: list[Certification]


# ---------- normalize any dict into the shape ----------

_ENTITY_DEFAULTS = {
    "work_experience": Job, "projects": Project, "research_experience": Research,
    "education": Education, "certifications": Certification,
}


def _blank(model: type[BaseModel]) -> dict:
    out = {}
    for name, f in model.model_fields.items():
        ann = f.annotation
        out[name] = [] if getattr(ann, "__origin__", None) is list else _blank(ann) if isinstance(ann, type) and issubclass(ann, BaseModel) else ""
    return out


def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "other"


class ProfileError(ValueError):
    pass


def normalize(d: dict) -> dict:
    """Fill every key the pipeline reads; keep everything else the user put in."""
    if not isinstance(d, dict):
        raise ProfileError("The JSON must be an object at the top level.")
    out = dict(d)
    pers = {**_blank(Personal), **(d.get("personal") or {})}
    pers["links"] = {**_blank(Links), **(pers.get("links") or {})}
    out["personal"] = pers
    out["summary"] = d.get("summary") or ""

    skills = d.get("skills") or {}
    if isinstance(skills, list):                      # [{category, items}] (the extraction shape) -> {cat: [...]}
        skills = {_snake(g.get("category", "other")): list(g.get("items", [])) for g in skills if isinstance(g, dict)}
    out["skills"] = {k: [str(x) for x in v] for k, v in skills.items() if isinstance(v, list) and v}

    for key, model in _ENTITY_DEFAULTS.items():
        rows = d.get(key) or []
        if not isinstance(rows, list):
            raise ProfileError(f"'{key}' must be a list.")
        out[key] = [{**_blank(model), **{k: v for k, v in r.items() if v is not None}} for r in rows if isinstance(r, dict)]

    for p in out["projects"]:                          # a project needs a name to be a heading
        p["name"] = p["name"] or (p["description"][:60] if p["description"] else "Untitled project")
    for j in out["work_experience"]:                   # an empty description falls back to the first highlight
        if not j["description"] and j["highlights"]:
            j["description"] = j["highlights"][0]

    if not pers["full_name"].strip():
        raise ProfileError("personal.full_name is empty. The resume header needs a name.")
    if not (out["work_experience"] or out["projects"] or out["research_experience"]):
        raise ProfileError("No work experience, projects or research found. There is nothing to write bullets from.")
    return out


def summary_of(p: dict) -> dict:
    """What the app shows after an upload, so the user can check the extraction."""
    return {
        "name": p["personal"]["full_name"],
        "email": p["personal"].get("email", ""),
        "counts": {k: len(p.get(k, [])) for k in ("work_experience", "projects", "research_experience",
                                                  "education", "certifications")}
                  | {"skills": sum(len(v) for v in p.get("skills", {}).values())},
        "jobs": [f"{j['title']} · {j['company']}" for j in p.get("work_experience", [])],
        "projects": [x["name"] for x in p.get("projects", [])],
        "research": [x["title"] for x in p.get("research_experience", [])],
        "education": [f"{e['degree']} {e['field']}".strip() + f" · {e['institution']}" for e in p.get("education", [])],
    }


# ---------- PDF -> profile ----------

EXTRACT = """You turn a resume's text into structured JSON. You are a copier, not a writer.

Rules:
- Copy facts exactly as written: names, titles, dates, numbers, tools, bullet text. Never reword a bullet.
- Never invent anything. If the resume doesn't say it, use "" or [].
- Every bullet under a job/project/research entry goes into that entry's highlights, verbatim, in order.
- description: one plain sentence on what the role/project is, taken from the resume's own words.
- tech: only tools/languages the resume names for that entry.
- skills: keep the resume's own category labels; if it has one flat list, use the category "Skills".
- Research/lab/thesis work goes in research_experience; jobs and internships in work_experience.
- Keep entries in the resume's order (most recent first)."""


def read_pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    text = re.sub(r"[ \t]+", " ", text)
    if len(text.strip()) < 200:
        raise ProfileError("Could not read text from this PDF (it may be a scanned image). Upload a text PDF or a JSON file.")
    return text


def extract_from_pdf(path: Path, filename: str = "") -> dict:
    from writer import structured
    text = read_pdf_text(path)
    out = structured(ExtractedProfile, [{"role": "system", "content": EXTRACT},
                                        {"role": "user", "content": f"RESUME TEXT\n\n{text}"}], tag="extract: resume pdf")
    prof = normalize(out.model_dump())
    prof["meta"] = {"source": "pdf", "file": filename or path.name}
    return prof


def template() -> dict:
    """An empty profile in the exact shape, for people who want to write their own JSON."""
    return normalize({"personal": {"full_name": "Your Name"}, "work_experience": [_blank(Job)]}) | {
        "projects": [_blank(Project)], "research_experience": [_blank(Research)],
        "education": [_blank(Education)], "certifications": [_blank(Certification)],
        "skills": {"languages": [], "frameworks": [], "tools": []}}


if __name__ == "__main__":
    print(json.dumps(template(), indent=2))
