"""RAG step 4: render the agent's resume.json into a 1-page PDF (template + printing in pdf_builder.py).

    python backend/rag/render.py backend/data/runs/<slug>/resume.json

Uses pdf_builder.py: the CSS template, the spacing LEVELS, browser print_pdf and
page_fill. Output: runs/<slug>/resume.html + resume.pdf.

Fit loop: try spacing levels roomiest -> tightest; if it is still 2 pages, drop the last bullet of the least
important block (last project, then 2nd research entry, then the oldest job's last bullet) and try again.
"""
import html
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROFILE  # noqa: E402
from profile_schema import normalize  # noqa: E402
from pdf_builder import CSS, DEFAULT_LEVEL, LEVELS, page_fill, print_pdf  # noqa: E402

e = html.escape
MON = {"January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr", "June": "Jun", "July": "Jul",
       "August": "Aug", "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec"}
MAX_TECH = 7


def short_date(s: str) -> str:
    for a, b in MON.items():
        s = s.replace(a, b)
    return s.strip(" -–")                        # "- May 2021" (no start date) -> "May 2021"


def plain(s: str) -> str:
    return re.sub(r"\s*[—–]\s*", ": ", s)          # "Graduate Research — Activation..." -> "...: Activation..."


def first_sentence(s: str) -> str:
    return re.split(r"(?<=[.!?])\s", s.strip(), maxsplit=1)[0] if s else ""


def ul(bullets: list[dict]) -> str:
    return "<ul>" + "".join(f"<li>{e(b['text'])}</li>" for b in bullets) + "</ul>"


def render(resume: dict, profile: dict, lvl: dict) -> str:
    p, links = profile["personal"], profile["personal"]["links"]
    urls = [links.get(k, "") for k in ("linkedin", "github", "portfolio")] + list(links.get("other") or [])
    contact = ([f'<a href="mailto:{e(p["email"])}">{e(p["email"])}</a>'] if p.get("email") else [])         + ([f'<a href="tel:{re.sub(r"[^0-9+]", "", p["phone"])}">{e(p["phone"])}</a>'] if p.get("phone") else [])         + [f'<a href="{e(u)}">{e(u.split("//")[-1].rstrip("/"))}</a>' for u in urls if u]
    parts = [f"<h1>{e(p['full_name'])}</h1>", f'<div class="contact">{"  |  ".join(contact)}</div>']

    for sec in resume["display_order"]:
        data = resume.get(sec)
        if not data:
            continue
        if sec == "summary":
            parts.append(f"<h2>Summary</h2><p>{e(data['text'])}</p>")
        elif sec == "skills":
            rows = "".join(f"<div><b>{e(r['label'].replace('Ai Ml', 'AI / ML').replace('Devops', 'DevOps'))}:</b> "
                           f"{e(', '.join(r['items']))}</div>" for r in data)
            parts.append(f'<h2>Technical Skills</h2><div class="skills">{rows}</div>')
        elif sec in ("experience", "projects", "research") and not any(x["bullets"] for x in data):
            continue                                    # no heading over a section with nothing in it
        elif sec == "experience":
            parts.append("<h2>Experience</h2>")
            for j in data:
                if not j["bullets"]:
                    continue
                parts.append(f'<div class="row"><span class="left">{e(j["company"])}</span>'
                             f'<span>{e(short_date(j["dates"]))}</span></div>')
                parts.append(f'<div class="sub-row"><span>{e(j["title"])}</span><span>{e(j["location"])}</span></div>')
                # context line only from an explicit profile field: the product description's first sentence
                # ("A product manager enters a product overview.") reads as noise on its own
                if j.get("context_line"):
                    parts.append(f'<p class="context">{e(j["context_line"])}</p>')
                parts.append(ul(j["bullets"]))
        elif sec == "projects":
            parts.append("<h2>Projects</h2>")
            for pr in data:
                if not pr["bullets"]:
                    continue
                link = (f'<a class="inline" href="{e(pr["link"])}">{e(pr["link"].split("//")[1].rstrip("/"))}</a>'
                        if pr.get("link") else "")
                parts.append(f'<div class="row"><span class="left">{e(plain(pr["name"]))}</span>{link}</div>')
                if pr.get("tech"):
                    parts.append(f'<div class="proj-tech">{e(" · ".join(pr["tech"][:MAX_TECH]))}</div>')
                parts.append(ul(pr["bullets"]))
        elif sec == "research":
            parts.append("<h2>Research Experience</h2>")
            for r in data:
                if not r["bullets"]:
                    continue
                sub = r["institution"] + (f" · Advisor: {r['advisor']}" if r.get("advisor") else "")
                parts.append(f'<div class="row"><span class="left">{e(plain(r["title"]))}</span>'
                             f'<span>{e(r["location"])}</span></div>')
                parts.append(f'<div class="sub-row"><span>{e(sub)}</span><span>{e(short_date(r["dates"]))}</span></div>')
                parts.append(ul(r["bullets"]))
        elif sec == "education":
            parts.append("<h2>Education</h2>")
            for ed in data:
                gpa = f"  |  GPA: {ed['gpa']}" if ed.get("gpa") else ""
                parts.append(f'<div class="row"><span class="left">{e(ed["school"])}</span><span>{e(ed["location"])}</span></div>')
                parts.append(f'<div class="sub-row"><span>{e(ed["degree"])}{e(gpa)}</span>'
                             f'<span>{e(short_date(ed["dates"]))}</span></div>')
                if ed.get("thesis"):                     # own line, so it never squeezes the dates column
                    parts.append(f'<p class="context">{e(ed["thesis"])}</p>')
        elif sec == "certifications":
            certs = [" · ".join(x for x in (c["name"], c["issuer"], short_date(c.get("date", ""))) if x) for c in data]
            parts.append(f"<h2>Certifications</h2><p>{'  |  '.join(e(c) for c in certs)}</p>")

    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{e(p["full_name"])} Resume</title>'
            f"<style>{CSS.format(**lvl)}</style></head><body>{''.join(parts)}</body></html>")


def trim_one(resume: dict) -> str | None:
    """Drop the last bullet of the least important block that still has more than one bullet."""
    blocks = ([("projects", i) for i in reversed(range(len(resume.get("projects") or [])))]
              + [("research", i) for i in reversed(range(len(resume.get("research") or [])))]
              + [("experience", i) for i in reversed(range(len(resume.get("experience") or [])))])
    for sec, i in blocks:
        b = resume[sec][i]["bullets"]
        if len(b) > 1:
            return f"{sec}: {b.pop()['text'][:60]}..."
    for sec, i in blocks[:-1]:                       # last resort: drop a whole project / research entry
        if resume[sec][i]["bullets"]:
            resume[sec][i]["bullets"] = []
            return f"{sec}: dropped entry {i}"
    return None


def build(resume_path: Path) -> Path:
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    profile = normalize(json.loads(PROFILE.read_text(encoding="utf-8")))
    work = resume_path.parent
    html_path, drafts = work / "resume.html", work / "drafts"
    cut = []

    while True:
        shutil.rmtree(drafts, ignore_errors=True)
        pages = {}

        def fits(i):
            if i not in pages:
                html_path.write_text(render(resume, profile, LEVELS[i]), encoding="utf-8")
                pages[i] = print_pdf(html_path, drafts / f"level-{i}.pdf")
            return pages[i] == 1

        if fits(DEFAULT_LEVEL):
            chosen = DEFAULT_LEVEL
            while chosen > 0 and fits(chosen - 1):
                chosen -= 1
            break
        chosen = next((i for i in range(DEFAULT_LEVEL + 1, len(LEVELS)) if fits(i)), None)
        if chosen is not None:
            break
        dropped = trim_one(resume)
        if not dropped:
            raise RuntimeError("Too long for one page and nothing left to trim.")
        cut.append(dropped)

    html_path.write_text(render(resume, profile, LEVELS[chosen]), encoding="utf-8")
    pdf = work / "resume.pdf"
    pdf.write_bytes((drafts / f"level-{chosen}.pdf").read_bytes())
    shutil.rmtree(drafts, ignore_errors=True)
    fill = page_fill(pdf)
    print(f"OK: {pdf}")
    print(f"Pages: 1 | spacing level {chosen}" + (f" | page fill ~{round(100 * fill)}%" if fill else ""))
    for c in cut:
        print(f"  trimmed to fit -> {c}")
    if fill and chosen == 0 and fill < 0.90:
        print("NOTE: still a gap at the roomiest spacing; raise a section budget in agent.py.")
    return {"pdf": str(pdf), "level": chosen, "fill": round(fill, 3) if fill else None, "trimmed": cut}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python backend/rag/render.py backend/data/runs/<slug>/resume.json")
    build(Path(sys.argv[1]).resolve())
