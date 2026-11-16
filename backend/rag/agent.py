"""RAG resume agent: JD -> section-by-section evidence-backed resume content (LLM via LiteLLM, cached).

    python backend/rag/agent.py my_jd.txt --company "Halcyon Freight" --role "AI Engineer"
    python backend/rag/agent.py my_jd.txt --dry-run      # no LLM calls: shows each field's evidence slate

Graph (BUILD order; the page DISPLAY order is set in assemble):
  prepare -> plan -> experience -> projects -> research -> education -> summary -> humanize -> skills -> certifications -> assemble

How fields stay un-jumbled:
  * prepare embeds the requirement lines ONCE (cached on disk), loads every chunk from ChromaDB, and ranks
    all chunks for every requirement locally (cosine + BM25, fused with RRF). Everything after that is offline.
  * Each field only sees its own pool: a job's bullets come only from that job's tree node and its children,
    a project's only from that project, research only from research nodes. Nothing crosses headings.
  * One shared `used` set: when a field uses a chunk, it is consumed, and the next field asking the same
    requirement gets that requirement's NEXT unused chunk from its own pool.
  * The LLM writes bullets only from the evidence slate it is handed, and must say which requirement,
    which JD keyword and which evidence ids each bullet stands on. Python rejects anything that isn't true.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import ENGINE, PROFILE, embeddings, load_tree  # noqa: E402
from profile_schema import normalize  # noqa: E402
from writer import current_model, reset_trace, structured, trace_summary, usage  # noqa: E402  (LiteLLM, cache, cost trace)
from retrieve import BM25, MODE_WEIGHT, RRF_K, WEAK, _collection, _tokens, load_jd, split_requirements  # noqa: E402

warnings.filterwarnings("ignore")

from config import ALIASES, CACHE_DIR, RUNS  # noqa: E402
CACHE = CACHE_DIR / "query_embeddings.json"

MARGIN = 0.06                   # evidence = within MARGIN of that requirement's best match (distances are
                                # squeezed into ~0.23-0.40, so a fixed cutoff lets everything through)
EDU_ONLY = re.compile(r"\b(bachelor|master|degree|ph\.?d|gpa|diploma)\b", re.I)   # answered by Education, not bullets
# soft-skill lines have no factual evidence in a profile; any bullet "proving" them would be invented
SOFT_ONLY = re.compile(r"^(strong |excellent |good )?(analytical|communication|interpersonal|problem[- ]solving"
                       r"|organizational|leadership|attention to detail|self[- ]starter|team player"
                       r"|ability to (collaborate|communicate|learn quickly|thrive))", re.I)
# "Ability to work with structured and unstructured data" is a hard skill: "ability to work" alone is not soft
MIN_PROJECT_REQS = 2            # a project must hold evidence for at least this many requirements
SEMANTIC_ONLY = 0.6             # G3: specificity when a match shares no word with the requirement (meaning only)
SPEC_MIN = 0.5                  # G3: below this the shared words are too common ("python") to count as a match
REF_DF = 6                      # G3: specificity = REF_DF / (chunks holding the rarest shared word), max 1
                                #     pytorch (4 chunks) = 1.0, data (14) = 0.43, python (26) = 0.23
PROX_MIN = 0.25                 # a requirement only counts as "matched" by a heading above this proximity
LEXICAL_PROX = 0.5              # G3 hybrid: a chunk naming the requirement's TOOL (PyTorch, Docker) is evidence
                                # even outside the meaning margin, as long as it is within WEAK distance
EXPERIENCE_BUDGET = [4, 3, 2]   # bullets per job, most recent first
PROJECTS, PROJECT_BULLETS = 3, 2
RESEARCH_BUDGET = [2, 1]        # bullets for the best and second-best research entry (0 relevance = skipped)
POOL_ALIAS = {"research_paper": "research_experience[0]"}   # the paper is evidence for that research entry
NOT_EVIDENCE = {"tech", "product", "models_tested", "subtitle", "advisor", "limitations", "future_work"}
# humanizer checklist, the parts a regex can catch (the humanize node's prompt covers the rest)
TELLS = [
    (re.compile(r"\bnot (just|only|merely|simply)\b[^.]*\bbut\b", re.I), "not-X-but-Y contrast"),
    (re.compile(r"\brather than\b", re.I), "X-rather-than-Y contrast"),
    (re.compile(r"^(in order to|with a focus on|driven by|as part of)\b", re.I), "staged opener"),
    (re.compile(r",\s*(ensuring|enabling|highlighting|showcasing|underscoring|demonstrating)\b", re.I),
     "trailing significance clause (', ensuring ...')"),
    (re.compile(r"\s-\s"), "hyphen used as a dash"),
    (re.compile(r"[\u201c\u201d\u2018\u2019]"), "curly quotes"),
]
BUZZ = re.compile(r"\b(leverag\w*|seamless\w*|robust|cutting[- ]edge|spearhead\w*|orchestrat\w*|pivotal|testament"
                  r"|synerg\w*|utiliz\w*|passionate|world[- ]class|state[- ]of[- ]the[- ]art)\b", re.I)


# ---------- LLM output shapes ----------

class Bullet(BaseModel):
    text: str = Field(description="The resume bullet. Must contain every keyword below, verbatim.")
    requirement_ids: list[int] = Field(description="R-numbers of the requirements this bullet proves (1 or 2)")
    keywords: list[str] = Field(description="Words copied verbatim from those requirements that the bullet proves")
    evidence_ids: list[str] = Field(description="E-ids of the evidence chunks every fact in the bullet comes from")


class SectionOut(BaseModel):
    bullets: list[Bullet]


class SummaryOut(BaseModel):
    text: str = Field(description="2-3 sentence professional summary")
    keywords: list[str] = Field(description="JD keywords the summary uses, verbatim from the requirements")
    bullet_refs: list[str] = Field(description="B-ids of the page bullets the summary's claims come from")


SYSTEM = """You write resume content that works as EVIDENCE for a job description.

You get REQUIREMENTS (R-numbers, from the job post) and an EVIDENCE slate (E-ids, facts from the candidate's
profile). Each bullet you write is a proof: one requirement, the job's own keyword, and the profile fact behind it.

Hard rules. Breaking any of them makes the bullet invalid:
1. Facts come ONLY from the cited evidence text. No new tools, numbers, outcomes, team sizes, users,
   companies or claims. If the evidence doesn't say it, the bullet doesn't say it.
2. A keyword is a word or phrase that appears verbatim in THREE places: the requirement text, your bullet,
   AND the cited evidence text (or the heading's TECH list). Only those count. Never force a JD phrase into
   a bullet the evidence doesn't contain ("for real-world use cases", "applying problem-solving").
   If a requirement shares no such keyword with its evidence, skip that requirement.
   ALIASES lines are the one exception: they say a JD spelling means the same as an evidence term, so you
   may write the JD spelling (e.g. evidence "PostgreSQL", JD "Postgres") and list it as the keyword.
3. Cite every evidence id you used. Use each evidence id in at most one bullet.
4. Shape: past-tense action verb + what was built or done + the keyword in context + the result or number
   if the evidence has one. The first 6-8 words say plainly what was done. Max 32 words, 1-2 sentences.
5. Write like a person, not a model. Plain English a recruiter outside the company understands.
   No em or en dashes, no "not X but Y" or "X rather than Y", no staged openers ("In order to", "With a
   focus on"), no trailing significance clauses (", ensuring...", ", enabling...", ", showcasing..."),
   no forced groups of three, no buzzwords (leveraging, seamlessly, robust, cutting-edge, spearheaded,
   orchestrated, pivotal, utilized), no "I", no two bullets starting with the same verb.
   Vary sentence length. Say the fact and stop.
6. Write exactly the number of bullets asked for (fewer only if the evidence runs out).
7. FILLER mode (the prompt says FILLER): no requirement matches this entry. Write ONE bullet from its
   strongest fact, with requirement_ids [] and keywords []. Rules 1, 3, 4, 5 still apply."""


# ---------- state ----------

class AgentState(TypedDict, total=False):
    jd: str
    company: str
    role: str
    dry_run: bool
    requirements: list[dict]
    ranks: list[list[tuple[str, float]]]     # per requirement: (chunk id, cosine distance), RRF order
    best: list[float]                        # per requirement: distance of its closest chunk anywhere
    corpus: dict
    tree: dict
    profile: dict
    aliases: dict                            # G2: profile term -> JD spellings
    toks: dict                               # G3: chunk id -> token set
    qtoks: list                              # G3: per requirement: tokens + alias-expanded profile terms
    df: dict                                 # G3: word -> how many chunks hold it
    tools: set                               # G3: one-word tool names from skills / tech lists
    home: dict                               # G1: R-number -> the pool with its best specific evidence
    proven: list[int]                        # R-numbers some page bullet already proves
    used: list[str]
    sections: dict                           # field -> content
    log: list[str]


# ---------- helpers ----------

def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def load_aliases() -> dict[str, list[str]]:
    """aliases.json: {profile term: [JD spellings that mean the SAME thing]}. Hand-written, never LLM-made."""
    if not ALIASES.exists():
        return {}
    return {k: v for k, v in json.loads(ALIASES.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def _bare(term: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", term).strip().lower()            # "AWS (EC2, S3, IAM)" -> "aws"


def _has(term: str, text: str) -> bool:
    """Whole-word, case-insensitive: 'JS' must not match 'json'."""
    t = term.strip().lower()
    return bool(t) and re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", text.lower()) is not None


def alias_pairs(aliases: dict) -> list[tuple[str, str]]:
    """(profile term, JD spelling) for every alias."""
    return [(k, a) for k, vals in aliases.items() for a in vals]


def proven(kw: str, proof: str, aliases: dict) -> bool:
    """A keyword is proven if it is in the evidence/TECH text itself, or the JD keyword is an alias
    of a profile term that is (G2). Never the other way: a profile term can't prove a looser JD word."""
    if _has(kw, proof) or kw.lower() in proof.lower():
        return True
    return any(kw.lower() == a.lower() and (_has(_bare(k), proof) or _has(k, proof)) for k, a in alias_pairs(aliases))


def alias_lines(req_text: str, proof: str, aliases: dict) -> list[str]:
    """Only the alias pairs this slate can actually use: JD spelling in the requirements, profile term in the evidence."""
    return sorted({f'"{a}" (JD) = "{_bare(k)}" (evidence)' for k, a in alias_pairs(aliases)
                   if _has(a, req_text) and (_has(_bare(k), proof) or _has(k, proof))})


def _pool_of(chunk_id: str, meta: dict) -> str:
    head = chunk_id if meta["level"] == 1 else meta["parent_id"]
    return POOL_ALIAS.get(head, head)


def _embed_cached(texts: list[str]) -> list[list[float]]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    from providers import embed_id
    space = embed_id()                                  # vectors from different models never mix
    key = lambda t: hashlib.sha1(f"{space}\n{t}".encode()).hexdigest()
    missing = [t for t in texts if key(t) not in cache]
    if missing:
        vecs = embeddings().embed_documents(missing, task_type="RETRIEVAL_QUERY")
        cache.update({key(t): v for t, v in zip(missing, vecs)})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"query embeddings: {len(texts) - len(missing)} cached, {len(missing)} embedded online")
    return [cache[key(t)] for t in texts]


def slate(state: AgentState, pool: str, n: int) -> list[dict]:
    """For one heading: each requirement's next UNUSED in-pool chunk, within MARGIN of its best match.
    A chunk picked by several requirements is listed once with all of them."""
    used, corpus = set(state["used"]), state["corpus"]
    home, proven = state.get("home", {}), set(state.get("proven", []))

    def priority(qi: int) -> int:        # G1: 0 = homed here and unproven, 1 = unproven, 2 = already proven
        n = qi + 1
        return 2 if n in proven else 0 if home.get(n) == pool else 1

    order = sorted(range(len(state["requirements"])),
                   key=lambda i: (priority(i), -state["requirements"][i]["weight"]))
    picked: dict[str, dict] = {}
    for qi in order:
        if state["requirements"][qi]["route"] != "bullets":
            continue
        for cid, dist in _in_pool(state, qi, pool):     # RRF order: meaning + keywords
            if cid in used or not proximity(state, qi, cid, dist):
                continue
            entry = picked.setdefault(cid, {"id": cid, "text": corpus[cid]["text"], "reqs": [], "best": dist,
                                            "rank": 9, "priority": []})
            entry["reqs"].append(qi + 1)
            entry["best"] = min(entry["best"], dist)
            entry["rank"] = min(entry["rank"], priority(qi))
            if priority(qi) < 2:
                entry["priority"].append(qi + 1)
            break                                  # this requirement got its next evidence for this heading
    # G1: evidence for unproven requirements first, closest first inside each rank
    return sorted(picked.values(), key=lambda e: (e["rank"], e["best"]))[: n * 2]


def specificity(state: AgentState, qi: int, cid: str, pool: str) -> float:
    """G3: how SPECIFIC the words a requirement shares with its evidence are, from document frequency
    over the profile chunks. Sharing only 'python' (26 chunks) scores 0.38; 'pytorch' (4 chunks) scores 1.
    No shared word at all = a pure meaning match, which gets a middle score."""
    tech = set(_tokens(" ".join(tech_of(state, pool))))
    common = state["qtoks"][qi] & (state["toks"][cid] | tech)
    if not common:
        return SEMANTIC_ONLY
    rarest = min(state["df"].get(t, 1) for t in common)
    return min(1.0, REF_DF / max(rarest, 1))


def proximity(state: AgentState, qi: int, cid: str, d: float) -> float:
    """1 = the best match anywhere, 0 = not evidence. Meaning: inside MARGIN of the best distance.
    Hybrid (G3): or the chunk names one of the requirement's tools and is within WEAK distance."""
    sem = max(0.0, 1 - (d - state["best"][qi]) / MARGIN)
    rare_tools = {t for t in state["qtoks"][qi] & state["toks"][cid] & state["tools"] if state["df"].get(t, 0) <= REF_DF}
    lex = LEXICAL_PROX if d <= WEAK and rare_tools else 0.0     # "pytorch" opens the door, "python" doesn't
    return max(sem, lex)


def _in_pool(state: AgentState, qi: int, pool: str):
    corpus = state["corpus"]
    for cid, d in state["ranks"][qi]:
        m = corpus[cid]["meta"]
        if _pool_of(cid, m) == pool and m["subsection"] not in NOT_EVIDENCE:
            yield cid, d


def fit(state: AgentState, qi: int, pool: str) -> tuple[float, float]:
    """(proximity, specificity) of the pool chunk that best proves requirement qi."""
    best = (0.0, 0.0)
    for cid, d in _in_pool(state, qi, pool):
        prox = proximity(state, qi, cid, d)
        if prox:
            cand = (prox, specificity(state, qi, cid, pool))
            if cand[0] * cand[1] > best[0] * best[1]:
                best = cand
    return best


def heading_score(state: AgentState, pool: str, only: set[int] | None = None) -> tuple[float, int]:
    """(score, matched). score = sum of weight x proximity x specificity; matched = requirements this pool
    proves with a SPECIFIC word (>= SPEC_MIN) or by meaning alone. `only` limits it to those R-numbers."""
    score, matched = 0.0, 0
    for qi, r in enumerate(state["requirements"]):
        if r["route"] != "bullets" or (only is not None and qi + 1 not in only):
            continue
        prox, spec = fit(state, qi, pool)
        score += r["weight"] * prox * spec
        matched += prox >= PROX_MIN and spec >= SPEC_MIN
    return score, matched


def candidate_pools(state: AgentState) -> list[str]:
    p = state["profile"]
    return ([f"work_experience[{i}]" for i in range(len(p["work_experience"]))]
            + [f"projects[{i}]" for i in range(len(p["projects"]))]
            + [f"research_experience[{i}]" for i in range(len(p["research_experience"]))])


def tech_of(state: AgentState, pool: str) -> list[str]:
    m = re.match(r"(\w+)\[(\d+)\]$", pool)
    if not m:
        return []
    return state["profile"].get(m.group(1), [])[int(m.group(2))].get("tech", [])


def validate(bullets: list[Bullet], ev: dict[str, dict], reqs: list[dict], n: int, tech: list[str],
             aliases: dict | None = None, filler: bool = False) -> list[str]:
    errs, seen_ev, verbs = [], set(), []
    for k, b in enumerate(bullets, 1):
        tag = f"bullet {k}"
        if bad := [e for e in b.evidence_ids if e not in ev]:
            errs.append(f"{tag}: evidence ids not in the slate: {bad}")
        if dup := [e for e in b.evidence_ids if e in seen_ev]:
            errs.append(f"{tag}: evidence already used by another bullet: {dup}")
        seen_ev.update(b.evidence_ids)
        if bad := [r for r in b.requirement_ids if not 1 <= r <= len(reqs)]:
            errs.append(f"{tag}: unknown requirement ids {bad}")
        req_text = " ".join(reqs[r - 1]["text"] for r in b.requirement_ids if 1 <= r <= len(reqs)).lower()
        src = " ".join(ev[e]["text"] for e in b.evidence_ids if e in ev)
        proof = (src + " " + " ".join(tech)).lower()
        if filler and (b.keywords or b.requirement_ids):
            errs.append(f"{tag}: FILLER bullets prove nothing: requirement_ids and keywords must be []")
        if not filler and not b.keywords:
            errs.append(f"{tag}: no keyword; every bullet must prove at least one requirement keyword")
        for rx, name in TELLS:
            if rx.search(b.text):
                errs.append(f"{tag}: AI tell: {name}")
        for kw in b.keywords:
            if kw.lower() not in req_text:
                errs.append(f"{tag}: keyword '{kw}' is not in the cited requirement text")
            if kw.lower() not in b.text.lower():
                errs.append(f"{tag}: keyword '{kw}' is not in the bullet text")
            if not proven(kw, proof, aliases or {}):
                errs.append(f"{tag}: keyword '{kw}' is not in its evidence or the TECH list (not proven)")
        if extra := _numbers(b.text) - _numbers(src):
            errs.append(f"{tag}: numbers not in its evidence: {sorted(extra)}")
        if re.search(r"[—–]", b.text):
            errs.append(f"{tag}: contains a dash character")
        if m := BUZZ.search(b.text):
            errs.append(f"{tag}: buzzword '{m.group(0)}'")
        if len(b.text.split()) > 40:
            errs.append(f"{tag}: {len(b.text.split())} words (max 32)")
        verbs.append(b.text.split()[0].lower() if b.text.split() else "")
    if len(set(verbs)) < len(verbs):
        errs.append("two bullets start with the same verb")
    if len(bullets) > n:
        errs.append(f"wrote {len(bullets)} bullets, asked for {n}")
    return errs


def filler_slate(state: AgentState, pool: str, k: int = 3) -> list[dict]:
    """A fixed slot with no matching evidence: its own unused facts, heading description first."""
    used, corpus = set(state["used"]), state["corpus"]
    ids = [cid for cid, m in ((c, corpus[c]["meta"]) for c in corpus)
           if _pool_of(cid, m) == pool and m["subsection"] not in NOT_EVIDENCE and cid not in used]
    ids.sort(key=lambda c: (corpus[c]["meta"]["level"], -len(corpus[c]["text"])))
    return [{"id": c, "text": corpus[c]["text"], "reqs": [], "priority": []} for c in ids[:k]]


def _repair_loop(schema, msgs: list[dict], check, tag: str = "") -> tuple[BaseModel, list[str]]:
    """One LLM call, validate, one repair call if needed. Returns (output, remaining errors)."""
    out = structured(schema, msgs, tag=tag)
    errs = check(out)
    if errs:
        msgs = msgs + [{"role": "assistant", "content": out.model_dump_json()},
                       {"role": "user", "content": "Fix these problems and return everything again:\n- "
                                                   + "\n- ".join(errs)}]
        out = structured(schema, msgs, tag=f"{tag} (repair)")
        errs = check(out)
    return out, errs


def write_heading(state: AgentState, field: str, pool: str, heading: str, n: int, fixed: bool = False) -> dict:
    """Evidence slate -> LLM bullets -> validation (one repair) -> consume the cited chunks.
    fixed=True: the entry is always on the page; with no matching evidence it gets 1 FILLER bullet."""
    ev_list = slate(state, pool, n)
    if not ev_list:
        if fixed:
            return write_filler(state, field, pool, heading)
        state["log"].append(f"{field} {heading}: no evidence within {MARGIN} of any requirement's best match, skipped")
        return {"heading": heading, "pool": pool, "bullets": [], "slate": []}
    ev = {f"E{i}": e for i, e in enumerate(ev_list, 1)}
    reqs = state["requirements"]
    used_reqs = sorted({r for e in ev_list for r in e["reqs"]})
    tech = tech_of(state, pool)
    aliases = state.get("aliases", {})
    al = alias_lines(" ".join(reqs[r - 1]["text"] for r in used_reqs),
                     " ".join(e["text"] for e in ev_list) + " " + " ".join(tech), aliases)
    prompt = (f"SECTION: {field}\nHEADING: {heading}\nTECH: {', '.join(tech) or '(none)'}\n"
              f"Write {min(n, len(ev))} bullets.\n\nREQUIREMENTS\n"
              + "\n".join(f"R{r}: {reqs[r - 1]['text']}" for r in used_reqs)
              + ("\n\nALIASES\n" + "\n".join(al) if al else "")
              + "\n\nEVIDENCE\n"
              + "\n".join(f"{eid} (matches R{', R'.join(map(str, e['reqs']))}"
                          + (f"; PRIORITY R{', R'.join(map(str, e['priority']))}" if e["priority"] else "")
                          + f"): {e['text']}" for eid, e in ev.items())
              + "\n\nCover the PRIORITY requirements first: no other bullet on the page proves them yet.")
    if state.get("dry_run"):
        print(f"\n--- {field}: {heading}  ({n} bullets)\n{prompt}")
        return {"heading": heading, "pool": pool, "bullets": [], "slate": list(ev_list)}

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    out, errs = _repair_loop(SectionOut, msgs, lambda o: validate(o.bullets, ev, reqs, n, tech, aliases),
                             tag=f"{field}: {heading}")
    good = []
    for k, b in enumerate(out.bullets, 1):
        if any(e.startswith(f"bullet {k}:") for e in errs):
            state["log"].append(f"{field} {heading}: dropped bullet {k} ({b.text[:50]}...)")
            continue
        good.append({"text": b.text, "requirements": b.requirement_ids, "keywords": b.keywords,
                     "evidence": [ev[e]["id"] for e in b.evidence_ids]})
        state["used"].extend(ev[e]["id"] for e in b.evidence_ids)       # consumed for every later field
        state.setdefault("proven", []).extend(r for r in b.requirement_ids if r not in state["proven"])
    if fixed and not good:                       # every bullet failed validation: still fill the slot
        return write_filler(state, field, pool, heading)
    return {"heading": heading, "pool": pool, "bullets": good, "slate": list(ev_list)}


def write_filler(state: AgentState, field: str, pool: str, heading: str) -> dict:
    ev_list = filler_slate(state, pool)
    if not ev_list:
        state["log"].append(f"{field} {heading}: fixed slot but no unused facts left")
        return {"heading": heading, "pool": pool, "bullets": [], "slate": []}
    ev = {f"E{i}": e for i, e in enumerate(ev_list, 1)}
    tech = tech_of(state, pool)
    prompt = (f"FILLER\nSECTION: {field}\nHEADING: {heading}\nTECH: {', '.join(tech) or '(none)'}\n"
              "No job requirement matches this entry. Write 1 bullet from its strongest fact.\n\nEVIDENCE\n"
              + "\n".join(f"{eid}: {e['text']}" for eid, e in ev.items()))
    if state.get("dry_run"):
        print(f"\n--- {field}: {heading}  (FILLER)\n{prompt}")
        return {"heading": heading, "pool": pool, "bullets": [], "slate": ev_list}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    reqs = state["requirements"]
    out, errs = _repair_loop(SectionOut, msgs,
                             lambda o: validate(o.bullets[:1], ev, reqs, 1, tech, state.get("aliases"), filler=True),
                             tag=f"{field}: {heading} (filler)")
    good = []
    if out.bullets and not errs:
        b = out.bullets[0]
        good = [{"text": b.text, "requirements": [], "keywords": [], "filler": True,
                 "evidence": [ev[e]["id"] for e in b.evidence_ids]}]
        state["used"].extend(ev[e]["id"] for e in b.evidence_ids)
        state["log"].append(f"{field} {heading}: no requirement matched, wrote 1 filler bullet")
    else:
        state["log"].append(f"{field} {heading}: filler bullet failed validation ({'; '.join(errs)[:80]})")
    return {"heading": heading, "pool": pool, "bullets": good, "slate": ev_list}


# ---------- nodes ----------

def prepare(state: AgentState) -> AgentState:
    reset_trace()                                   # cost trace covers exactly this build
    state = {**state, **load_jd(state)}
    state.update(split_requirements(state))
    route = lambda t: "education" if EDU_ONLY.search(t) else "soft" if SOFT_ONLY.search(t) else "bullets"
    reqs = [{**r, "route": route(r["text"])} for r in state["requirements"]]
    for n, r in enumerate(reqs, 1):
        if r["route"] != "bullets":
            print(f"  R{n} routed to {r['route']} (no bullet evidence): {r['text'][:70]}")
    qvecs = np.asarray(_embed_cached([r["text"] for r in reqs]), dtype=np.float32)

    col = _collection()
    got = col.get(where={"section": {"$ne": "narrative"}}, include=["documents", "metadatas", "embeddings"])
    corpus = {i: {"text": t, "meta": m} for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])}
    ids = got["ids"]
    vecs = np.asarray(got["embeddings"], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
    dist = 1 - qvecs @ vecs.T                                # (requirements x chunks), all local

    docs = [_tokens(f"{corpus[i]['meta']['heading']} {corpus[i]['text']}") for i in ids]
    bm25 = BM25(ids, docs)

    # G2: a requirement saying "Postgres" also searches (BM25) for the profile's "PostgreSQL"
    aliases = load_aliases()
    expand = [" ".join(_bare(k) for k, a in alias_pairs(aliases) if _has(a, r["text"])) for r in reqs]
    for n, x in enumerate(expand, 1):
        if x:
            print(f"  R{n} alias-expanded with: {x}")

    ranks = []
    for qi, r in enumerate(reqs):
        by_vec = [ids[j] for j in np.argsort(dist[qi])]
        by_kw = bm25.top(f"{r['text']} {expand[qi]}", len(ids))
        rrf: dict[str, float] = {}
        for ranked in (by_vec, by_kw):
            for rank, cid in enumerate(ranked, 1):
                rrf[cid] = rrf.get(cid, 0.0) + 1 / (RRF_K + rank)
        d = dict(zip(ids, dist[qi].tolist()))
        ranks.append([(cid, round(d[cid], 4)) for cid in sorted(rrf, key=rrf.get, reverse=True)])
    print(f"ranked {len(ids)} chunks for {len(reqs)} requirements (offline from here, except the writer)")
    best = [float(dist[qi].min()) for qi in range(len(reqs))]

    # G3: document frequency per word, and the TOOL vocabulary: one-word skill / tech names (pytorch, docker)
    profile = normalize(json.loads(PROFILE.read_text(encoding="utf-8")))
    df = Counter(t for d in docs for t in set(d))
    names = [x for items in profile["skills"].values() for x in items]
    names += [x for sec in ("work_experience", "projects", "research_experience") for e in profile.get(sec, [])
              for x in e.get("tech", [])]
    tools = {tk[0] for x in names if len(tk := _tokens(_bare(x))) == 1}
    return {"jd": state["jd"], "requirements": reqs, "ranks": ranks, "best": best, "corpus": corpus, "tree": load_tree(),
            "profile": profile, "aliases": aliases, "df": dict(df), "tools": tools,
            "toks": {i: set(d) for i, d in zip(ids, docs)},
            "qtoks": [set(_tokens(f"{r['text']} {x}")) for r, x in zip(reqs, expand)],
            "home": {}, "proven": [], "used": [], "sections": {}, "log": []}


def plan(state: AgentState) -> AgentState:
    """G1: before anything is written, give every requirement ONE home: the pool (job, project, research)
    with its best SPECIFIC evidence (weight x proximity x specificity), without overfilling it.
    Greedy from the strongest (requirement, pool) pair down: a pool takes at most as many requirements as
    it has bullets, and only PROJECTS projects / len(RESEARCH_BUDGET) research entries can open.
    Slates then serve home requirements first, so proof lands where it is strongest."""
    p = state["profile"]
    cap = {f"work_experience[{i}]": EXPERIENCE_BUDGET[min(i, len(EXPERIENCE_BUDGET) - 1)]
           for i in range(len(p["work_experience"]))}
    cap |= {f"projects[{i}]": PROJECT_BULLETS for i in range(len(p["projects"]))}
    cap |= {f"research_experience[{i}]": RESEARCH_BUDGET[0] for i in range(len(p["research_experience"]))}
    limit = {"projects": PROJECTS, "research_experience": len(RESEARCH_BUDGET)}

    pairs = []
    for qi, r in enumerate(state["requirements"]):
        if r["route"] != "bullets":
            continue
        for pool in cap:
            prox, spec = fit(state, qi, pool)
            if prox >= PROX_MIN and spec >= SPEC_MIN:
                pairs.append((r["weight"] * prox * spec, qi + 1, pool))
    home, load, opened = {}, {}, {}
    for score, n, pool in sorted(pairs, reverse=True):
        kind = pool.split("[")[0]
        if n in home or load.get(pool, 0) >= cap[pool]:
            continue
        if kind in limit and pool not in load and opened.get(kind, 0) >= limit[kind]:
            continue
        if pool not in load:
            opened[kind] = opened.get(kind, 0) + 1
        home[n] = pool
        load[pool] = load.get(pool, 0) + 1
        print(f"  plan: R{n} -> {pool}  (fit {score:.2f})")
    for n, r in enumerate(state["requirements"], 1):
        if r["route"] == "bullets" and n not in home:
            print(f"  plan: R{n} -> no specific evidence with room left")
    return {"home": home}


def experience(state: AgentState) -> AgentState:
    out = []
    for i, job in enumerate(state["profile"]["work_experience"]):
        n = EXPERIENCE_BUDGET[min(i, len(EXPERIENCE_BUDGET) - 1)]
        w = write_heading(state, "experience", f"work_experience[{i}]", f"{job['title']} at {job['company']}", n,
                          fixed=True)                                    # every job, every time
        out.append({**w, "company": job["company"], "title": job["title"], "location": job.get("location", ""),
                    "dates": f"{job['start']} - {job['end']}",
                    "context": job.get("product", {}).get("what_it_does", "") if isinstance(job.get("product"), dict) else ""})
    return {"sections": {**state["sections"], "experience": out}, "used": state["used"], "log": state["log"],
            "proven": state["proven"]}


def _unproven(state: AgentState) -> set[int]:
    return {n for n, r in enumerate(state["requirements"], 1) if r["route"] == "bullets"} - set(state["proven"])


def _pick_greedy(state: AgentState, pools: dict[int, str], k: int, eligible) -> list[int]:
    """G1 + G3: pick headings one at a time by NEW coverage: weighted specific fit on requirements no
    bullet proves yet. Recomputed after each pick is written, so the next pick fills the next gap."""
    left, picked = set(pools), []
    while left and len(picked) < k:
        unproven = _unproven(state)
        gain = {i: heading_score(state, pools[i], unproven)[0] for i in left}
        total = {i: heading_score(state, pools[i])[0] for i in left}
        ok = [i for i in left if eligible(i) and gain[i] > 0]
        if not ok:                       # every requirement is proven: fill the page by total relevance
            ok = [i for i in left if eligible(i) and total[i] > 0]
        if not ok:
            break
        best = max(ok, key=lambda i: (gain[i], total[i]))
        left.discard(best)
        picked.append(best)
        yield best, gain[best]
    return picked


def projects(state: AgentState) -> AgentState:
    profs = state["profile"]["projects"]
    pools = {i: f"projects[{i}]" for i in range(len(profs))}
    matched = {i: heading_score(state, pools[i])[1] for i in pools}
    for i, m in matched.items():
        if 0 < m < MIN_PROJECT_REQS:
            state["log"].append(f"projects: {profs[i]['name']} skipped, specific evidence for only {m} requirement(s)")
    out = []
    for i, gain in _pick_greedy(state, pools, PROJECTS, lambda i: matched[i] >= MIN_PROJECT_REQS):
        p = profs[i]
        print(f"  projects: {p['name']} (new coverage {gain:.2f})")
        w = write_heading(state, "projects", pools[i], p["name"], PROJECT_BULLETS)
        out.append({**w, "name": p["name"], "link": p.get("live") or p.get("repo", ""), "tech": p.get("tech", [])})
    return {"sections": {**state["sections"], "projects": out}, "used": state["used"], "log": state["log"],
            "proven": state["proven"]}


def research(state: AgentState) -> AgentState:
    """Fixed slots: EVERY research entry is on the page. The one with the most new coverage gets
    RESEARCH_BUDGET[0] bullets, the next RESEARCH_BUDGET[1], and so on; no match -> 1 filler bullet."""
    rs = state["profile"]["research_experience"]
    pools = {i: f"research_experience[{i}]" for i in range(len(rs))}
    out, done = [], set()

    def write(i, rank, gain):
        r = rs[i]
        n = RESEARCH_BUDGET[min(rank, len(RESEARCH_BUDGET) - 1)]
        print(f"  research: {r['title'][:50]} (new coverage {gain:.2f}, {n} bullets)")
        w = write_heading(state, "research", pools[i], r["title"], n, fixed=True)
        out.append({**w, "title": r["title"], "institution": r["institution"], "location": r.get("location", ""),
                    "dates": f"{r['start']} - {r['end']}", "advisor": r.get("advisor", ""), "start": r["start"]})
        done.add(i)

    for rank, (i, gain) in enumerate(_pick_greedy(state, pools, len(rs), lambda i: True)):
        write(i, rank, gain)
    for i in pools:                                   # entries with zero relevance still get their slot
        if i not in done:
            write(i, len(done), 0.0)
    out.sort(key=lambda r: _date_key(r["start"]), reverse=True)        # newest first on the page
    return {"sections": {**state["sections"], "research": out}, "used": state["used"], "log": state["log"],
            "proven": state["proven"]}


def _date_key(s: str) -> tuple[int, int]:
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    y = re.search(r"\d{4}", s or "")
    m = next((k for k, mon in enumerate(months, 1) if mon in (s or "").lower()), 0)
    return (int(y.group(0)) if y else 0, m)


def education(state: AgentState) -> AgentState:
    """Retrieve and select: facts copied from the profile, no LLM. The thesis line is shown only
    if research evidence ended up on the page."""
    show_thesis = bool(any(r["bullets"] for r in state["sections"].get("research", [])))
    out = []
    for i, e in enumerate(state["profile"]["education"]):
        row = {"school": e["institution"], "location": e["location"],
               "degree": ", ".join(x for x in (e["degree"], e["field"]) if x),
               "gpa": e.get("gpa", ""), "dates": f"{e['start']} - {e['end']}"}
        if show_thesis and isinstance(e.get("thesis"), dict) and e["thesis"].get("title"):
            t = e["thesis"]
            row["thesis"] = (f"{t.get('type') or 'Thesis'}: {t['title']}"
                             + (f" (Advisor: {t['advisor']})" if t.get("advisor") else ""))
        out.append(row)
        state["used"].append(f"education[{i}]")
    return {"sections": {**state["sections"], "education": out}, "used": state["used"]}


def summary(state: AgentState) -> AgentState:
    """Written last-but-two, from what is already on the page, so it repeats proof instead of inventing it."""
    page = [b for sec in ("experience", "projects", "research") for h in state["sections"].get(sec, [])
            for b in h["bullets"]]
    refs = {f"B{i}": b for i, b in enumerate(page, 1)}
    base = state["corpus"].get("summary", {}).get("text", "")
    top = sorted(state["requirements"], key=lambda r: -r["weight"])[:6]
    if state.get("dry_run") or not refs:
        return {"sections": {**state["sections"], "summary": {"text": base, "keywords": [], "bullet_refs": []}}}
    prompt = ("Write a 2-3 sentence professional summary, max 60 words, for this job.\n"
              "Use ONLY facts from PAGE BULLETS and PROFILE SUMMARY. Same style rules as bullets. "
              "Lead with what the candidate builds, then the strongest proof for the top requirements.\n\n"
              "TOP REQUIREMENTS\n" + "\n".join(f"- {r['text']}" for r in top)
              + f"\n\nPROFILE SUMMARY\n{base}\n\nPAGE BULLETS\n"
              + "\n".join(f"{k}: {b['text']}" for k, b in refs.items()))
    def check(o):
        src = base + " " + " ".join(refs[r]["text"] for r in o.bullet_refs if r in refs)
        errs = [f"numbers not in the sources: {sorted(x)}" for x in [_numbers(o.text) - _numbers(src)] if x]
        errs += [f"AI tell: {name}" for rx, name in TELLS if rx.search(o.text)]
        errs += ["contains a dash character"] if re.search(r"[\u2014\u2013]", o.text) else []
        return errs

    out, _ = _repair_loop(SummaryOut, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                          check, tag="summary")
    state["used"].append("summary")
    return {"sections": {**state["sections"], "summary": out.model_dump()}, "used": state["used"]}


class Rewrite(BaseModel):
    id: str = Field(description="The B-id or S, unchanged")
    text: str = Field(description="The rewritten text (or the original, if it already reads human)")


class HumanizeOut(BaseModel):
    items: list[Rewrite]


HUMANIZE = """You are an editor removing AI-writing tells from resume lines. Rewrite each line so it reads
like the candidate wrote it, WITHOUT changing what it says.

Remove: not-X-but-Y and X-rather-than-Y contrasts, staged openers, trailing significance clauses
(", ensuring...", ", enabling..."), forced groups of three, stacked clauses, inflated or sales words
(leveraging, seamlessly, robust, cutting-edge, pivotal, spearheaded, orchestrated, utilized, showcase),
em/en dashes, filler ("various", "a wide range of"), and repeated rhythm between lines.

Hard limits (a line that breaks one is thrown away and the original is kept):
- Keep every KEEP word exactly as written (same spelling and case).
- Keep every number exactly. Add no number, tool, outcome or claim that isn't in the line already.
- Bullets: start with a past-tense verb, no two bullets in a HEADING start with the same verb, max 32 words.
- If a line already reads human, return it unchanged."""


def humanize(state: AgentState) -> AgentState:
    """Humanizer pass over every page bullet + the summary, then RE-VALIDATE each rewrite against its own
    evidence. A rewrite that drops a keyword, changes a number or adds a fact is reverted to the draft."""
    if state.get("dry_run"):
        return {}
    secs, corpus, reqs, aliases = state["sections"], state["corpus"], state["requirements"], state.get("aliases", {})
    items, refs = [], {}
    for sec in ("experience", "projects", "research"):
        for h in secs.get(sec) or []:
            for b in h["bullets"]:
                bid = f"B{len(refs) + 1}"
                refs[bid] = (h, b)
                items.append(f"{bid} [HEADING {h['heading'][:40]}] KEEP: {', '.join(b['keywords']) or '-'}\n   {b['text']}")
    summ = secs.get("summary") or {}
    if summ.get("text"):
        items.append(f"S [SUMMARY] KEEP: {', '.join(summ.get('keywords', [])) or '-'}\n   {summ['text']}")
    if not items:
        return {}
    out = structured(HumanizeOut, [{"role": "system", "content": HUMANIZE},
                                   {"role": "user", "content": "\n".join(items)}], tag="humanize: page")
    new = {r.id.strip(): r.text.strip() for r in out.items}

    changed = reverted = 0
    for bid, (h, b) in refs.items():
        text = new.get(bid, b["text"])
        if text == b["text"]:
            continue
        ev = {f"E{k}": {"id": c, "text": corpus[c]["text"]} for k, c in enumerate(b["evidence"], 1) if c in corpus}
        cand = Bullet(text=text, requirement_ids=b["requirements"], keywords=b["keywords"], evidence_ids=list(ev))
        errs = validate([cand], ev, reqs, 1, tech_of(state, h["pool"]), aliases, filler=b.get("filler", False))
        others = {x["text"].split()[0].lower() for x in h["bullets"] if x is not b and x["text"].split()}
        if text.split() and text.split()[0].lower() in others:
            errs.append("same opening verb as another bullet in this heading")
        if errs:
            reverted += 1
            state["log"].append(f"humanize: kept draft for {bid} ({errs[0]})")
            continue
        b["draft"], b["text"] = b["text"], text
        changed += 1
    if summ.get("text") and new.get("S") and new["S"] != summ["text"]:
        s_new = new["S"]
        ok = (_numbers(s_new) <= _numbers(summ["text"]) and not any(rx.search(s_new) for rx, _ in TELLS)
              and all(k.lower() in s_new.lower() for k in summ.get("keywords", [])) and not re.search(r"[\u2014\u2013]", s_new))
        if ok:
            summ["draft"], summ["text"] = summ["text"], s_new
            changed += 1
        else:
            reverted += 1
            state["log"].append("humanize: kept the summary draft (rewrite broke a number, keyword or tell rule)")
    print(f"humanize: {changed} lines rewritten, {reverted} rewrites reverted to the draft, "
          f"{len(items) - changed - reverted} already human")
    return {"sections": {**secs, "summary": summ}, "log": state["log"]}


def skills(state: AgentState) -> AgentState:
    """Select, no LLM: keep the profile skills that the JD names or the page bullets use, JD matches first."""
    jd = state["jd"].lower()
    page = " ".join(b["text"] for sec in ("experience", "projects", "research")
                    for h in state["sections"].get(sec, []) for b in h["bullets"]).lower()
    rows = []
    for cat, items in state["profile"]["skills"].items():
        names = lambda s: [_bare(s), *state.get("aliases", {}).get(s, [])]      # G2: JD may use an alias
        in_jd = [s for s in items if any(_has(n, jd) for n in names(s))]
        in_page = [s for s in items if s not in in_jd and any(_has(n, page) for n in names(s))]
        if in_jd or in_page:
            rows.append({"label": cat.replace("_", " ").title(), "items": in_jd + in_page, "jd_hits": len(in_jd)})
    rows.sort(key=lambda r: -r["jd_hits"])
    return {"sections": {**state["sections"], "skills": rows[:5]}}


def certifications(state: AgentState) -> AgentState:
    """Select, no LLM: certs whose words appear in the JD, then dated ones, max 3."""
    jd = set(_tokens(state["jd"]))
    certs = state["profile"].get("certifications", [])
    ranked = sorted(certs, key=lambda c: (-len(jd & set(_tokens(c["name"]))), not c.get("date")))
    return {"sections": {**state["sections"], "certifications": [c for c in ranked if c.get("date")][:3]}}


def assemble(state: AgentState) -> AgentState:
    s = state["sections"]
    resume = {"company": state.get("company", ""), "role": state.get("role", ""),
              "display_order": ["summary", "skills", "experience", "projects", "research", "education", "certifications"],
              **{k: s.get(k) for k in ("summary", "skills", "experience", "projects", "research", "education",
                                       "certifications")}}
    slug = slug_of(state.get("company", ""), state.get("role", ""))
    out_dir = RUNS / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    reqs = state["requirements"]
    lines = [f"# Evidence map: {state.get('company', '')} {state.get('role', '')}", ""]
    for n, r in enumerate(reqs, 1):
        proofs = [(sec, h["heading"], b) for sec in ("experience", "projects", "research")
                  for h in s.get(sec, []) for b in h["bullets"] if n in b["requirements"]]
        lines.append(f"## R{n} [{r['mode']}] {r['text']}")
        lines += [f"- **{', '.join(b['keywords'])}** in {sec} / {head}: {b['text']}  \n  evidence: `{'`, `'.join(b['evidence'])}`"
                  for sec, head, b in proofs] or ["- _no bullet proves this requirement_"]
        lines.append("")
    (out_dir / "evidence.md").write_text("\n".join(lines), encoding="utf-8")
    if not state.get("dry_run"):                   # cost trace: every build of this JD is kept
        cost_file = out_dir / "cost.json"
        hist = json.loads(cost_file.read_text(encoding="utf-8")).get("builds", []) if cost_file.exists() else []
        hist.append(trace_summary())
        cost_file.write_text(json.dumps({"total": round(sum(b["cost"] for b in hist), 6),
                                         "saved": round(sum(b["saved"] for b in hist), 6),
                                         "builds": hist}, indent=2), encoding="utf-8")
    from coverage import report                                    # G4: score this run against the JD
    report(reqs, state["best"], resume, state["jd"], state["profile"], state.get("aliases"), out_dir)
    print(f"\nwrote {out_dir / 'resume.json'} and evidence.md")
    return {}


STEPS = ["prepare", "plan", "experience", "projects", "research", "education", "summary", "humanize", "skills",
         "certifications", "assemble"]


def slug_of(company: str, role: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{company or 'job'} {role or ''}".lower()).strip("-")


def build_graph():
    g = StateGraph(AgentState)
    steps = STEPS
    for name in steps:
        g.add_node(name, globals()[name])
    g.add_edge(START, steps[0])
    for a, b in zip(steps, steps[1:]):
        g.add_edge(a, b)
    g.add_edge(steps[-1], END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jd_file")
    ap.add_argument("--company", default="")
    ap.add_argument("--role", default="")
    ap.add_argument("--dry-run", action="store_true", help="show each field's evidence slate; no LLM calls")
    a = ap.parse_args()
    final = build_graph().invoke({"jd": Path(a.jd_file).read_text(encoding="utf-8"), "company": a.company,
                                  "role": a.role, "dry_run": a.dry_run})
    s = final["sections"]
    if not a.dry_run:
        print("\nSUMMARY\n  " + s["summary"]["text"])
        print("SKILLS\n" + "\n".join(f"  {r['label']}: {', '.join(r['items'])}" for r in s["skills"]))
        for sec in ("experience", "projects", "research"):
            print(sec.upper())
            for h in s[sec]:
                print(f"  {h['heading']}")
                for b in h["bullets"]:
                    print(f"    - {b['text']}\n        proves R{b['requirements']} via {b['keywords']} <- {b['evidence']}")
    for line in final["log"]:
        print("NOTE:", line)
    if not a.dry_run:
        print(usage())
