"""RAG Resume Builder API: the LangGraph resume agent + PDF builder behind a small FastAPI server.

    python server.py                      # http://127.0.0.1:8710  (the Vite app proxies /api here)

Endpoints
  GET  /api/health                         profile + index status + missing API keys
  POST /api/source      (file)             upload your resume: .pdf (LLM-extracted) or .json -> chunk + embed
  GET  /api/profile                        the active profile (summary + full JSON)
  GET  /api/profile/template               an empty profile.json to fill in by hand
  POST /api/detect      {jd, company?, role?} company + role (LLM only if one is missing) -> chips if unsure
  POST /api/runs        {jd, company, role}  start the agent (one at a time) -> {job_id}
  GET  /api/jobs/{id}                      live status: which graph node is running, log tail, result slug
  GET  /api/runs                           every saved run with its coverage scores
  GET  /api/runs/{slug}                    resume.json + coverage.json + last render info
  PUT  /api/runs/{slug}/resume             save hand edits to bullets / summary
  POST /api/runs/{slug}/render             PDF builder: one-page fit loop -> resume.pdf
  GET  /api/runs/{slug}/pdf                the PDF (for the in-app viewer)
  GET  /api/costs                          LLM spend (LiteLLM-priced) per JD and in total
  POST /api/lint        {text}             humanizer + style checks on one line
  POST /api/ingest                         re-chunk and re-embed the profile into ChromaDB (costs embeddings)
"""
import contextlib
import io
import json
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag"))
import agent  # noqa: E402
import ingest  # noqa: E402
import render  # noqa: E402
import profile_schema  # noqa: E402
import providers  # noqa: E402
from config import API_PORT, DATA, PROFILE, RUNS, SOURCE_META, TREE_FILE, missing_keys  # noqa: E402

app = FastAPI(title="RAG Resume Builder")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
                   allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()          # the agent redirects stdout and shares caches: one job at a time
SLUG = re.compile(r"^[a-z0-9-]+$")


# ---------- jobs ----------

class _Log(io.TextIOBase):
    """Captures the agent's print() lines into the job, line by line."""

    def __init__(self, job):
        self.job, self.buf = job, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job["log"].append(line.rstrip())
        return len(s)


def _new_job(kind: str, steps: list[str]) -> dict:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "kind": kind, "status": "queued", "steps": [{"name": s, "state": "pending"} for s in steps],
                 "log": [], "slug": None, "error": None, "started": time.time(), "finished": None, "render": None}
    return JOBS[jid]


def _set_step(job, name, state):
    for s in job["steps"]:
        if s["name"] == name:
            s["state"] = state
            if state == "running":
                s["t0"] = time.time()
            elif state in ("done", "error") and "t0" in s:
                s["ms"] = int((time.time() - s.pop("t0")) * 1000)


def _run_agent(job, jd, company, role):
    with RUN_LOCK, contextlib.redirect_stdout(_Log(job)):
        try:
            job["status"] = "running"
            steps = agent.STEPS
            _set_step(job, steps[0], "running")
            graph = agent.build_graph()
            for update in graph.stream({"jd": jd, "company": company, "role": role, "dry_run": False},
                                       stream_mode="updates"):
                for node in update:
                    _set_step(job, node, "done")
                    nxt = steps.index(node) + 1
                    if nxt < len(steps):
                        _set_step(job, steps[nxt], "running")
            job["slug"] = agent.slug_of(company, role)
            _set_step(job, "render", "running")
            job["render"] = _render(job["slug"])
            _set_step(job, "render", "done")
            print(agent.usage())
            job["status"] = "done"
        except BaseException as e:                       # sys.exit inside the pipeline lands here too
            for s in job["steps"]:
                if s["state"] == "running":
                    _set_step(job, s["name"], "error")
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            job["log"].append(traceback.format_exc().splitlines()[-1])
        finally:
            job["finished"] = time.time()


INGEST_STEPS = ["load_profile", "chunk", "merge_near_dups", "store"]


def _run_ingest(job, source: Path | None = None, filename: str = ""):
    """Optional source step first (PDF: read + LLM extract, JSON: validate), then the ingest graph."""
    with RUN_LOCK, contextlib.redirect_stdout(_Log(job)):
        try:
            job["status"] = "running"
            if source is not None:
                _ingest_source(job, source, filename)
            steps = INGEST_STEPS
            _set_step(job, steps[0], "running")
            for update in ingest.build_graph().stream({"dry_run": False}, stream_mode="updates"):
                for node in update:
                    _set_step(job, node, "done")
                    nxt = steps.index(node) + 1
                    if nxt < len(steps):
                        _set_step(job, steps[nxt], "running")
            job["status"] = "done"
        except profile_schema.ProfileError as e:
            for s in job["steps"]:
                if s["state"] == "running":
                    _set_step(job, s["name"], "error")
            job["status"], job["error"] = "error", str(e)
        except BaseException as e:
            for s in job["steps"]:
                if s["state"] == "running":
                    _set_step(job, s["name"], "error")
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        finally:
            job["finished"] = time.time()


def _ingest_source(job, source: Path, filename: str):
    """Turn the uploaded file into data/profile.json (normalized). Only replaces the profile if it succeeds."""
    if source.suffix == ".pdf":
        _set_step(job, "read_pdf", "running")
        text = profile_schema.read_pdf_text(source)
        print(f"read {len(text):,} characters from {filename}")
        _set_step(job, "read_pdf", "done")
        _set_step(job, "extract", "running")
        import writer
        writer.reset_trace()
        prof = profile_schema.extract_from_pdf(source, filename)
        print(writer.usage().replace("LLM:", "extract:"))
        _set_step(job, "extract", "done")
    else:
        _set_step(job, "validate", "running")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise profile_schema.ProfileError(f"Not valid JSON: {e}")
        prof = profile_schema.normalize(raw)
        prof.setdefault("meta", {})["source"] = "json"
        _set_step(job, "validate", "done")
    s = profile_schema.summary_of(prof)
    print(f"profile: {s['name']} | " + ", ".join(f"{k} {v}" for k, v in s["counts"].items()))
    PROFILE.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
    SOURCE_META.write_text(json.dumps({"kind": source.suffix[1:], "file": filename, "uploaded_at": time.time()},
                                      indent=2), encoding="utf-8")


# ---------- runs on disk ----------

def _run_dir(slug: str) -> Path:
    if not SLUG.match(slug) or not (RUNS / slug / "resume.json").exists():
        raise HTTPException(404, f"no run named {slug}")
    return RUNS / slug


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _render(slug: str) -> dict:
    d = RUNS / slug
    info = render.build(d / "resume.json")
    info = {**info, "built_at": time.time()}
    (d / "render.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------- API ----------

class RunIn(BaseModel):
    jd: str
    company: str = ""
    role: str = ""


class LintIn(BaseModel):
    text: str


@app.get("/api/health")
def health():
    tree = _read(TREE_FILE) or {}
    nodes = tree.get("nodes", {})
    prof = _read(PROFILE) or {}
    return {"ok": True, "model": agent.current_model(), "missing_keys": missing_keys(),
            "providers": providers.describe(),
            "profile": {"exists": PROFILE.exists() and bool(nodes),
                        "name": prof.get("personal", {}).get("full_name", ""),
                        "jobs": len(prof.get("work_experience", [])), "projects": len(prof.get("projects", [])),
                        "research": len(prof.get("research_experience", [])), "source": _read(SOURCE_META)},
            "index": {"chunks": len(nodes), "built_at": TREE_FILE.stat().st_mtime if TREE_FILE.exists() else None},
            "busy": RUN_LOCK.locked()}


MAX_UPLOAD = 8 * 1024 * 1024


@app.post("/api/source")
async def upload_source(file: UploadFile = File(...)):
    name = Path(file.filename or "resume").name
    ext = Path(name).suffix.lower()
    if ext not in (".pdf", ".json"):
        raise HTTPException(415, "Upload a .pdf resume or a .json profile.")
    if RUN_LOCK.locked():
        raise HTTPException(409, "Another job is in progress.")
    if miss := missing_keys():
        raise HTTPException(412, f"Add {miss[0]} to backend/.env first (see .env.example).")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 8 MB.")
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise HTTPException(415, "That file isn't a PDF.")
    for old in DATA.glob("source.*"):
        if old.name != "source.json":
            old.unlink()
    dest = DATA / f"source{ext}" if ext == ".pdf" else DATA / "source_upload.json"
    dest.write_bytes(data)
    pre = ["read_pdf", "extract"] if ext == ".pdf" else ["validate"]
    job = _new_job("ingest", [*pre, *INGEST_STEPS])
    threading.Thread(target=_run_ingest, args=(job, dest, name), daemon=True).start()
    return {"job_id": job["id"]}


@app.get("/api/profile")
def get_profile():
    prof = _read(PROFILE)
    if not prof:
        raise HTTPException(404, "No profile yet. Upload your resume (PDF or JSON).")
    prof = profile_schema.normalize(prof)
    return {"summary": profile_schema.summary_of(prof), "source": _read(SOURCE_META), "profile": prof}


@app.get("/api/profile/template")
def profile_template():
    return profile_schema.template()


class DetectIn(BaseModel):
    jd: str
    company: str = ""
    role: str = ""


@app.post("/api/detect")
def detect_meta(body: DetectIn):
    """Company + role for a post. No LLM call when both are given; chips to confirm when the LLM is unsure."""
    if len(body.jd.strip()) < 200:
        raise HTTPException(422, "The job post looks too short. Paste the full description (200+ characters).")
    if not (body.company.strip() and body.role.strip()) and (miss := missing_keys()):
        raise HTTPException(412, f"Add {miss[0]} to backend/.env first (see .env.example).")
    import jd_meta
    try:
        return jd_meta.detect(body.jd, body.company, body.role)
    except Exception as e:                       # never block a run on this: let the person type the names
        return {"company": body.company.strip(), "role": body.role.strip(), "company_options": [],
                "role_options": [], "needs_confirm": True, "source": "error", "error": str(e)[:300]}


@app.post("/api/runs")
def start_run(body: RunIn):
    if len(body.jd.strip()) < 200:
        raise HTTPException(422, "The job post looks too short. Paste the full description (200+ characters).")
    if RUN_LOCK.locked():
        raise HTTPException(409, "Another run is in progress. Wait for it to finish.")
    if miss := missing_keys():
        raise HTTPException(412, f"Add {miss[0]} to backend/.env first (see .env.example).")
    if not PROFILE.exists() or not TREE_FILE.exists():
        raise HTTPException(412, "Upload your resume first (PDF or JSON) so there is evidence to retrieve.")
    job = _new_job("agent", [*agent.STEPS, "render"])
    threading.Thread(target=_run_agent, args=(job, body.jd, body.company.strip(), body.role.strip()),
                     daemon=True).start()
    return {"job_id": job["id"]}


@app.post("/api/ingest")
def start_ingest():
    if RUN_LOCK.locked():
        raise HTTPException(409, "Another job is in progress.")
    if not PROFILE.exists():
        raise HTTPException(412, "Upload your resume first.")
    job = _new_job("ingest", INGEST_STEPS)
    threading.Thread(target=_run_ingest, args=(job,), daemon=True).start()
    return {"job_id": job["id"]}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    return {**job, "log": job["log"][-80:]}


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS.glob("*/resume.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r, cov = _read(d), _read(d.parent / "coverage.json") or {}
        out.append({"slug": d.parent.name, "company": r.get("company", ""), "role": r.get("role", ""),
                    "updated": d.stat().st_mtime, "has_pdf": (d.parent / "resume.pdf").exists(),
                    "scores": cov.get("scores"), "cost": _cost_brief(d.parent)})
    return out


@app.get("/api/runs/{slug}")
def get_run(slug: str):
    d = _run_dir(slug)
    return {"slug": slug, "resume": _read(d / "resume.json"), "coverage": _read(d / "coverage.json"),
            "render": _read(d / "render.json"), "has_pdf": (d / "resume.pdf").exists(),
            "cost": _read(d / "cost.json")}


@app.put("/api/runs/{slug}/resume")
def save_resume(slug: str, resume: dict):
    d = _run_dir(slug)
    if "display_order" not in resume:
        raise HTTPException(422, "not a resume.json")
    (d / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@app.post("/api/runs/{slug}/render")
def render_run(slug: str):
    _run_dir(slug)
    if RUN_LOCK.locked():
        raise HTTPException(409, "A run is in progress. Build the PDF when it finishes.")
    with RUN_LOCK, contextlib.redirect_stdout(io.StringIO()):
        try:
            return _render(slug)
        except Exception as e:
            raise HTTPException(500, str(e))


@app.get("/api/runs/{slug}/pdf")
def get_pdf(slug: str):
    pdf = _run_dir(slug) / "resume.pdf"
    if not pdf.exists():
        raise HTTPException(404, "no PDF yet")
    return FileResponse(pdf, media_type="application/pdf", headers={"Cache-Control": "no-store"})


def _cost_brief(d: Path) -> dict | None:
    c = _read(d / "cost.json")
    if not c:
        return None
    last = c["builds"][-1] if c.get("builds") else {}
    return {"total": c.get("total", 0.0), "saved": c.get("saved", 0.0), "builds": len(c.get("builds", [])),
            "last": {k: last.get(k) for k in ("at", "cost", "saved", "calls", "cache_hits", "model")}}


@app.get("/api/costs")
def costs():
    runs = []
    for d in sorted(RUNS.glob("*/resume.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = _read(d)
        runs.append({"slug": d.parent.name, "company": r.get("company", ""), "role": r.get("role", ""),
                     "cost": _cost_brief(d.parent)})
    tracked = [x["cost"] for x in runs if x["cost"]]
    return {"model": agent.current_model(), "total": round(sum(c["total"] for c in tracked), 6),
            "saved": round(sum(c["saved"] for c in tracked), 6), "builds": sum(c["builds"] for c in tracked),
            "untracked": len(runs) - len(tracked), "runs": runs}


@app.post("/api/lint")
def lint(body: LintIn):
    t, issues = body.text.strip(), []
    if re.search(r"[—–]", t):
        issues.append("Contains an em or en dash")
    if m := agent.BUZZ.search(t):
        issues.append(f"Buzzword: {m.group(0)}")
    issues += [f"AI tell: {name}" for rx, name in agent.TELLS if rx.search(t)]
    if (n := len(t.split())) > 40:
        issues.append(f"{n} words (bullets max out around 32)")
    return {"issues": issues}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")
