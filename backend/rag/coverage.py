"""G4: coverage report. How well does a built resume answer the JD's requirement lines?

    python backend/rag/coverage.py <jd.txt> backend/data/runs/<slug>/resume.json     # score an existing run
    (agent.py also calls report() at the end of every run and writes runs/<slug>/coverage.md)

Every requirement gets one status:
  PROVEN   a page bullet cites it (the evidence map's proof)
  MISSED   the profile HAS evidence for it (best distance <= WEAK) but no bullet used it  -> tuning target
  GAP      nothing in the profile is close (best distance > WEAK)                         -> honest gap
  ROUTED   answered elsewhere (education line) or soft skill (no factual proof possible)

Scores (weighted by req 3 / duty 2 / pref 1):
  coverage      = PROVEN / (PROVEN + MISSED + GAP)     how much of the JD the page proves
  recall        = PROVEN / (PROVEN + MISSED)           how much of the AVAILABLE evidence the agent used
  term hit rate = JD tech terms you have (skills list) that show up anywhere on the page
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import WEAK  # noqa: E402

BULLET_SECTIONS = ("experience", "projects", "research")


def _page_text(resume: dict) -> str:
    parts = [(resume.get("summary") or {}).get("text", "")]
    parts += [", ".join(r["items"]) for r in resume.get("skills") or []]
    parts += [b["text"] for sec in BULLET_SECTIONS for h in resume.get(sec) or [] for b in h["bullets"]]
    return " ".join(parts).lower()


def _bare(skill: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", skill).strip().lower()          # "AWS (EC2, S3, IAM)" -> "aws"


def jd_terms(jd: str, profile: dict, aliases: dict | None = None) -> list[str]:
    """Profile skills (or their aliases) that the JD names, as the JD's term."""
    low, found = jd.lower(), []
    for items in profile["skills"].values():
        for s in items:
            names = [_bare(s)] + [a.lower() for a in (aliases or {}).get(s, [])]
            for n in names:
                if len(n) > 1 and re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", low):
                    found.append(s)
                    break
    return sorted(set(found))


def statuses(requirements: list[dict], best: list[float], resume: dict) -> list[dict]:
    proofs: dict[int, list[str]] = {}
    for sec in BULLET_SECTIONS:
        for h in resume.get(sec) or []:
            for b in h["bullets"]:
                for r in b["requirements"]:
                    proofs.setdefault(r, []).append(f"{sec} / {h['heading']}: {b['text']}")
    rows = []
    for n, (r, d) in enumerate(zip(requirements, best), 1):
        if n in proofs:
            st = "PROVEN"
        elif r.get("route", "bullets") != "bullets":
            st = "ROUTED"
        elif d <= WEAK:
            st = "MISSED"
        else:
            st = "GAP"
        rows.append({"n": n, "mode": r["mode"], "weight": r["weight"], "text": r["text"], "status": st,
                     "best": round(d, 3), "route": r.get("route", "bullets"), "proofs": proofs.get(n, [])})
    return rows


def scores(rows: list[dict]) -> dict:
    w = lambda st: sum(r["weight"] for r in rows if r["status"] in st)
    proven, missed, gap = w({"PROVEN"}), w({"MISSED"}), w({"GAP"})
    return {"coverage": proven / (proven + missed + gap) if proven + missed + gap else 0.0,
            "recall": proven / (proven + missed) if proven + missed else 0.0,
            "counts": {s: sum(r["status"] == s for r in rows) for s in ("PROVEN", "MISSED", "GAP", "ROUTED")}}


def report(requirements, best, resume, jd, profile, aliases=None, out_dir: Path | None = None) -> dict:
    rows = statuses(requirements, best, resume)
    sc = scores(rows)
    terms = jd_terms(jd, profile, aliases)
    page = _page_text(resume)
    shown = [t for t in terms if any(n in page for n in [_bare(t)] + [a.lower() for a in (aliases or {}).get(t, [])])]
    sc["terms"] = {"have": len(terms), "shown": len(shown), "missing": [t for t in terms if t not in shown]}
    sc["term_hit_rate"] = len(shown) / len(terms) if terms else 0.0

    sc["filler"] = sum(bool(b.get("filler")) for sec in BULLET_SECTIONS for h in resume.get(sec) or []
                       for b in h["bullets"])
    c = sc["counts"]
    head = (f"coverage {sc['coverage']:.0%} | recall {sc['recall']:.0%} | terms {len(shown)}/{len(terms)} "
            f"| PROVEN {c['PROVEN']}  MISSED {c['MISSED']}  GAP {c['GAP']}  ROUTED {c['ROUTED']}"
            f" | filler bullets {sc['filler']}")
    print("\nCOVERAGE  " + head)
    for r in rows:
        if r["status"] in ("MISSED", "GAP"):
            print(f"  {r['status']:<6} R{r['n']} [{r['mode']}] d={r['best']}  {r['text'][:80]}")
    if sc["terms"]["missing"]:
        print("  terms you have but the page never shows: " + ", ".join(sc["terms"]["missing"]))

    if out_dir:
        md = [f"# Coverage", "", head, ""]
        for r in rows:
            md.append(f"## R{r['n']} {r['status']} [{r['mode']}] (best d={r['best']})")
            md.append(r["text"])
            md += [f"- {p}" for p in r["proofs"]]
            md.append("")
        md += ["## JD terms", f"shown: {', '.join(shown) or '-'}", f"missing: {', '.join(sc['terms']['missing']) or '-'}"]
        (out_dir / "coverage.md").write_text("\n".join(md), encoding="utf-8")
        (out_dir / "coverage.json").write_text(json.dumps({"scores": sc, "rows": rows}, indent=2), encoding="utf-8")
    return sc


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python backend/rag/coverage.py <jd.txt> <runs/<slug>/resume.json>")
    from agent import load_aliases, prepare  # noqa: E402  (lazy: agent imports this module)
    st = prepare({"jd": Path(sys.argv[1]).read_text(encoding="utf-8")})
    resume_path = Path(sys.argv[2]).resolve()
    report(st["requirements"], st["best"], json.loads(resume_path.read_text(encoding="utf-8")), st["jd"],
           st["profile"], load_aliases(), resume_path.parent)
