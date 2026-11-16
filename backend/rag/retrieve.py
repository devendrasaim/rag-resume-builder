"""RAG step 2: multi-query retrieval of profile chunks for a job post (hybrid search per requirement + MMR).

    python backend/rag/retrieve.py                    needs a job post .txt
    python backend/rag/retrieve.py my_jd.txt --k 15 --lam 0.6 --with-narrative

Graph:  load_jd -> split_requirements -> embed_queries -> search_each -> fuse_all -> mmr

  split_requirements  jd_split.split tags every sentence req / pref / duty / exposure;
                      boilerplate (pay, benefits, EEO, visa, about-us) is dropped
  embed_queries       one embedding per requirement line (one batched call)
  search_each         per requirement: vector search (ChromaDB) + BM25, fused with RRF -> its own evidence list
  fuse_all            weighted RRF across requirements (req 3, duty 2, pref 1): a chunk that answers
                      many requirements, or one requirement very well, rises
  mmr                 pick k chunks: lam * relevance - (1 - lam) * similarity to chunks already picked
"""
import argparse
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import COLLECTION, DB_DIR, ENGINE, check_index, embeddings, expand, load_tree  # noqa: E402
from jd_split import split as classify_lines  # noqa: E402

warnings.filterwarnings("ignore")

PER_QUERY = 10       # candidates each search returns per requirement
EVIDENCE = 3         # chunks kept as evidence per requirement
RRF_K = 60
MAX_QUERIES = 30
WEAK = 0.40          # best cosine distance above this = weak evidence for that requirement (tune)
MODE_WEIGHT = {"req": 3.0, "other": 2.0, "pref": 1.0}   # exposure lines are skipped
BOILER = re.compile(
    r"equal[- ]opportunit|\beeo\b|we believe|proud to|our people|our culture|our values|benefit|401\(?k|\bpto\b|paid time|insurance|salary|compensation|pay range|\$\s?\d"
    r"|sponsor|visa|citizen|clearance|background check|drug test|\bapply\b|application (process|form|deadline)"
    r"|about (us|the company)|^(the|our) (company|organization|firm)\b"
    r"|founded|headquarter|our mission|diversity|inclusion|accommodation|disabilit|veteran|relocat"
    r"|contract[- ]to[- ](perm|hire)|full[- ]time|\bw-?2\b|\b1099\b|per hour|/hr\b", re.I)
COMPANY_BLURB = re.compile(r"(company (description|overview)|about (us|the company)|who we are|our story)\b", re.I)
INLINE_PREF_EXTRA = re.compile(r"\b(is|are|would be) (beneficial|an advantage|advantageous)\b|\bis helpful\b", re.I)
STOP = set("""a an and are as at be by for from has have in is it its of on or our that the this to
we will with you your their they who what can all any more about into across using use work working
team teams role job experience years strong ability skills including such other well new help""".split())


class RetrieveState(TypedDict, total=False):
    jd: str
    k: int
    lam: float
    with_narrative: bool
    requirements: list[dict]     # {"mode", "text", "weight"}
    qvecs: list[list[float]]
    corpus: dict                 # id -> {"text", "meta", "vec"}
    evidence: list[dict]         # per requirement: {"mode", "text", "best_dist", "weak", "top": [{id, path, text, dist}]}
    per_req_ranked: list[list[str]]
    fused: list[tuple[str, float]]
    results: list[dict]
    grouped: list[dict]          # results grouped under their heading + its linked subsections


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9+#./-]*[a-z0-9+#]|[a-z0-9]", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


def _collection():
    import chromadb
    col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    check_index(col)
    return col


class BM25:
    def __init__(self, ids: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.ids, self.docs, self.k1, self.b = ids, docs, k1, b
        self.tfs = [Counter(d) for d in docs]
        self.avgdl = sum(map(len, docs)) / len(docs)
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, n: int) -> list[str]:
        q = set(_tokens(query))
        scores = {}
        for i, tf, d in zip(self.ids, self.tfs, self.docs):
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
                    for t in q & tf.keys())
            if s > 0:
                scores[i] = s
        return sorted(scores, key=scores.get, reverse=True)[:n]


# ---------- nodes ----------

def load_jd(state: RetrieveState) -> RetrieveState:
    jd = state["jd"].replace("\r\n", "\n").strip()
    if len(jd) < 200:
        sys.exit("job post looks empty or too short")
    return {"jd": jd}


def split_requirements(state: RetrieveState) -> RetrieveState:
    # a paragraph that opens as the company blurb ("Company Description Halcyon is...") is dropped whole
    jd = "\n".join(l for l in state["jd"].splitlines() if not COMPANY_BLURB.match(l.strip()))
    reqs, seen = [], set()
    for mode, s in classify_lines(jd):
        s = s.strip()
        if mode not in MODE_WEIGHT or len(s) < 30 or len(s) > 400 or BOILER.search(s) or s.endswith(":"):
            continue
        if mode == "req" and INLINE_PREF_EXTRA.search(s):   # "...is beneficial"
            mode = "pref"
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        reqs.append({"mode": mode, "text": s, "weight": MODE_WEIGHT[mode]})
    # keep the strongest asks if the post is very long (stable sort keeps the post's order inside a mode)
    reqs = sorted(reqs, key=lambda r: -r["weight"])[:MAX_QUERIES]
    counts = Counter(r["mode"] for r in reqs)
    print(f"{len(reqs)} requirement lines (req {counts['req']}, duty {counts['other']}, pref {counts['pref']})")
    return {"requirements": reqs}


def embed_queries(state: RetrieveState) -> RetrieveState:
    texts = [r["text"] for r in state["requirements"]]
    return {"qvecs": embeddings().embed_documents(texts, task_type="RETRIEVAL_QUERY")}


def search_each(state: RetrieveState) -> RetrieveState:
    col = _collection()
    where = None if state.get("with_narrative") else {"section": {"$ne": "narrative"}}
    everything = col.get(where=where, include=["documents", "metadatas", "embeddings"])
    corpus = {i: {"text": t, "meta": m, "vec": np.asarray(v, dtype=np.float32)}
              for i, t, m, v in zip(everything["ids"], everything["documents"],
                                    everything["metadatas"], everything["embeddings"])}
    ids = list(corpus)
    bm25 = BM25(ids, [_tokens(f"{corpus[i]['meta']['entry']} {corpus[i]['text']}") for i in ids])

    res = col.query(query_embeddings=state["qvecs"], n_results=PER_QUERY, where=where)   # one batched call
    evidence, per_req = [], []
    for r, v_ids, v_dist in zip(state["requirements"], res["ids"], res["distances"]):
        dist = dict(zip(v_ids, v_dist))
        rrf: dict[str, float] = {}
        for ranked in (v_ids, bm25.top(r["text"], PER_QUERY)):
            for rank, i in enumerate(ranked, 1):
                rrf[i] = rrf.get(i, 0.0) + 1 / (RRF_K + rank)
        ranked = sorted(rrf, key=rrf.get, reverse=True)
        per_req.append(ranked)
        best = min(v_dist) if v_dist else 1.0
        evidence.append({**r, "best_dist": round(best, 3), "weak": best > WEAK, "top": [
            {"id": i, "path": corpus[i]["meta"]["path"], "text": corpus[i]["text"],
             "dist": round(dist[i], 3) if i in dist else None} for i in ranked[:EVIDENCE]]})
    weak = sum(e["weak"] for e in evidence)
    print(f"searched {len(evidence)} requirements ({weak} with weak evidence, best distance > {WEAK})")
    return {"corpus": corpus, "evidence": evidence, "per_req_ranked": per_req}


def fuse_all(state: RetrieveState) -> RetrieveState:
    score: dict[str, float] = {}
    for e, ranked in zip(state["evidence"], state["per_req_ranked"]):
        if e["weak"]:              # nothing in the profile matches it: report it, but don't let it vote
            continue
        for rank, i in enumerate(ranked, 1):
            r = e
            score[i] = score.get(i, 0.0) + r["weight"] / (RRF_K + rank)
    fused = sorted(score.items(), key=lambda x: x[1], reverse=True)
    print(f"{len(fused)} distinct chunks across all requirements")
    return {"fused": fused}


def mmr(state: RetrieveState) -> RetrieveState:
    corpus, fused, k, lam = state["corpus"], state["fused"], state.get("k", 15), state.get("lam", 0.6)
    ids = [i for i, _ in fused]
    rel = np.array([s for _, s in fused]) / fused[0][1]   # fused relevance scaled to 0..1
    vecs = np.stack([corpus[i]["vec"] for i in ids])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T

    picked: list[int] = []
    while len(picked) < min(k, len(ids)):
        if picked:
            score = lam * rel - (1 - lam) * sim[:, picked].max(axis=1)
            score[picked] = -np.inf
            picked.append(int(score.argmax()))
        else:
            picked.append(int(rel.argmax()))

    # which requirements each picked chunk is evidence for (1-based, in the evidence list order)
    covers = {}
    for n, e in enumerate(state["evidence"], 1):
        for t in e["top"]:
            covers.setdefault(t["id"], []).append(n)
    results = []
    for p in picked:
        m = corpus[ids[p]]["meta"]
        results.append({"id": ids[p], "path": m["path"], "section": m["section"], "entry": m["entry"],
                        "level": m["level"], "heading": m["heading"], "subsection": m["subsection"],
                        "heading_id": m["path"] if m["level"] == 1 else m["parent_id"],
                        "text": corpus[ids[p]]["text"], "relevance": round(float(rel[p]), 3),
                        "covers": covers.get(ids[p], [])})
    return {"results": results}


def link_headings(state: RetrieveState) -> RetrieveState:
    """Group the picked chunks under their heading and attach the heading's full linked context."""
    tree = load_tree()
    groups: dict[str, dict] = {}
    for r in state["results"]:                      # keeps MMR order: best heading first
        g = groups.setdefault(r["heading_id"], {**expand(r["heading_id"], tree), "hits": []})
        g["hits"].append(r)
    return {"grouped": list(groups.values())}


def build_graph():
    g = StateGraph(RetrieveState)
    steps = ["load_jd", "split_requirements", "embed_queries", "search_each", "fuse_all", "mmr", "link_headings"]
    for name in steps:
        g.add_node(name, globals()[name])
    g.add_edge(START, steps[0])
    for a, b in zip(steps, steps[1:]):
        g.add_edge(a, b)
    g.add_edge(steps[-1], END)
    return g.compile()


def retrieve(jd: str, k: int = 15, lam: float = 0.6, with_narrative: bool = False) -> dict:
    """Entry point for other graphs (e.g. the pick node). Returns {"results", "grouped", "evidence"}."""
    out = build_graph().invoke({"jd": jd, "k": k, "lam": lam, "with_narrative": with_narrative})
    return {"results": out["results"], "grouped": out["grouped"], "evidence": out["evidence"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jd_file")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--lam", type=float, default=0.6, help="1.0 = pure relevance, 0.0 = pure diversity")
    ap.add_argument("--with-narrative", action="store_true")
    a = ap.parse_args()
    out = retrieve(Path(a.jd_file).read_text(encoding="utf-8"), a.k, a.lam, a.with_narrative)

    print("\nEVIDENCE PER REQUIREMENT")
    for n, e in enumerate(out["evidence"], 1):
        flag = "WEAK" if e["weak"] else "    "
        print(f"{n:>2}. [{e['mode']:<5}] {flag} {e['best_dist']:.3f}  {e['text'][:85]}")
        print(f"      -> {e['top'][0]['path']}: {e['top'][0]['text'][:70]}")
    print("\nTOP CHUNKS (MMR), GROUPED UNDER THEIR HEADINGS")
    for g in out["grouped"]:
        total = sum(len(v) for v in g["subsections"].values())
        print(f"\n[{g['section']}] {g['heading'][:70]}   ({total} linked items available)")
        for r in g["hits"]:
            where = "heading" if r["level"] == 1 else r["subsection"]
            print(f"   {r['relevance']:.2f}  {where:<18} reqs {','.join(map(str, r['covers'])) or '-':<10} {r['text'][:65]}")
