"""RAG step 1: ingest the profile JSON (backend/data/profile.json) into ChromaDB with structure-based chunking.

    python backend/rag/ingest.py              build the collection (re-builds from scratch)
    python backend/rag/ingest.py --dry-run    chunk only, print stats + a sample, no API calls

Graph:  load_profile -> chunk -> merge_near_dups -> store

  chunk            walks the JSON: one chunk per highlight / note / component / skills row / degree / cert,
                   each tagged with section, entry, field and its JSON path (a stable id)
                   exact repeats (normalized text) are dropped here
  merge_near_dups  chunks >= NEAR_DUP cosine to a kept chunk in the same entry are merged into it
  store            ChromaDB, embedding "[section | entry] text" so short items keep their context
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DB_DIR, PROFILE, RAG, TREE_FILE, load_env  # noqa: E402
from profile_schema import normalize  # noqa: E402

ENGINE = RAG          # old name kept: other modules import it as the base folder
COLLECTION = "profile"
CONTAINERS = {"skills", "certifications"}   # headings that only group items; never searched
NEAR_DUP = 0.97
MIN_CHARS = 25
SHORT_HIGHLIGHT = 60   # project highlights shorter than this are folded into the project's main chunk

# Not resume content (contact info, form answers, links) -> never embedded
SKIP_TOP = {"meta", "personal", "work_authorization", "standard_form_answers", "job_search", "headline"}
# Per-entity keys that are labels/links, or plain-English duplicates of other fields
SKIP_KEYS = {"name", "title", "company", "institution", "location", "type", "start", "end", "current",
             "repo", "live", "status", "source", "started", "nextwork_url", "pdf", "year",
             "simple_summary", "simple_highlights", "contribution_scope"}
# work_experience[].product (optional): only this is work context; one_liner / the_wedge / who_buys_it /
# competitive_frame are the company's marketing copy
PRODUCT_KEEP = {"what_it_does"}
# Lists that read as one line ("Tech: Python, PyTorch, ...") instead of one chunk per item
JOINED_LISTS = {"tech", "models_tested", "focus", "ctl_properties_verified", "federated_properties_verified",
                "limitations", "future_work"}


def embeddings():
    """OpenAI, Gemini or local embeddings, whichever providers.py picks from your keys."""
    from providers import Embedder
    return Embedder()


def check_index(col) -> None:
    """The stored vectors only match query vectors from the same embedding model."""
    from providers import embed_id
    built, now = (col.metadata or {}).get("embed", "google:models/gemini-embedding-001"), embed_id()
    if built != now:
        raise RuntimeError(f"The index was built with {built} embeddings but your keys now select {now}. "
                           "Re-index the profile (Re-index in the app, or python backend/rag/ingest.py).")


class IngestState(TypedDict, total=False):
    dry_run: bool
    profile: dict
    chunks: list[Document]
    vectors: list[list[float]]
    stored: int


# ---------- helpers ----------

def _label(key: str) -> str:
    return key.replace("_", " ")


def _flat(v) -> str:
    """Dicts/lists -> one readable line."""
    if isinstance(v, dict):
        return "; ".join(f"{_label(k)}: {_flat(x)}" for k, x in v.items() if x not in ("", None, [], {}))
    if isinstance(v, list):
        return ", ".join(_flat(x) for x in v)
    return str(v)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%$.+#]", " ", s.lower())).strip(" .")


def _walk_entity(section: str, entry: str, obj: dict, path: str):
    """Yield (text, field, path) for every fact inside one job / research entry / project."""
    for key, val in obj.items():
        if key in SKIP_KEYS or val in ("", None, [], {}):
            continue
        p = f"{path}.{key}"
        if isinstance(val, str):
            yield val, key, p
        elif isinstance(val, list) and (key in JOINED_LISTS or not all(isinstance(x, str) for x in val)):
            yield f"{_label(key).capitalize()}: {_flat(val)}", key, p
        elif isinstance(val, list):
            for i, item in enumerate(val):
                yield item, key, f"{p}[{i}]"
        elif isinstance(val, dict):             # components, agents, phases, key_results, product, ...
            short = []
            for k, v in val.items():
                if v in ("", None, [], {}):
                    continue
                line = f"{_label(k)}: {_flat(v)}"
                if len(line) < MIN_CHARS * 2:   # "name: Velociti", "bank bullets: 41" -> grouped below
                    short.append(line)
                else:
                    yield line, key, f"{p}.{k}"
            if short:
                yield f"{_label(key).capitalize()}: " + "; ".join(short), key, p


# ---------- nodes ----------

def load_profile(state: IngestState) -> IngestState:
    profile = normalize(json.loads(PROFILE.read_text(encoding="utf-8")))
    print(f"loaded {PROFILE.name}: {profile['personal']['full_name']}")
    return {"profile": profile}


def chunk(state: IngestState) -> IngestState:
    """Build the tree  section -> heading (level 1) -> subsection -> item (level 2).

    Level 1 = one heading per job / research entry / project / degree / skills / certifications / summary.
             Its text is the heading's main fact (description; for projects + short highlights + tech).
    Level 2 = every item under it, tagged with its subsection (highlights, components, engineering_notes, ...)
             and linked by parent_id. Both levels are embedded, so a query can hit a heading or a detail.
    """
    prof = state["profile"]
    raw = []   # (level, section, heading, subsection, path, parent_id, text)

    def entity(section, eid, heading, body, items):
        raw.append((1, section, heading, "main", eid, "", body))
        raw.extend((2, section, heading, f, p, eid, t) for t, f, p in items)

    for i, job in enumerate(prof.get("work_experience", [])):
        eid = f"work_experience[{i}]"
        heading = f"{job['title']} · {job['company']} ({job['start']} – {job['end']})"
        if isinstance(job.get("product"), dict):   # keep what the product does; drop its sales pitch
            job = {**job, "product": {k: v for k, v in job["product"].items() if k in PRODUCT_KEEP}}
        rest = {k: v for k, v in job.items() if k != "description"}
        entity("experience", eid, heading, job.get("description", heading), _walk_entity("experience", heading, rest, eid))

    for i, r in enumerate(prof.get("research_experience", [])):
        eid = f"research_experience[{i}]"
        heading = f"{r['title']} · {r['institution']} ({r['start']} – {r['end']})"
        if r.get("advisor"):
            heading += f" · advisor {r['advisor']}"
        rest = {k: v for k, v in r.items() if k not in ("description", "advisor")}
        entity("research", eid, heading, r.get("description", heading), _walk_entity("research", heading, rest, eid))

    if paper := prof.get("research_paper"):
        rest = {k: v for k, v in paper.items() if k != "abstract_summary"}
        entity("research", "research_paper", paper["title"], paper.get("abstract_summary", paper["title"]),
               _walk_entity("research", paper["title"], rest, "research_paper"))

    for i, proj in enumerate(prof.get("projects", [])):
        eid = f"projects[{i}]"
        hl = proj.get("highlights", [])
        short = [h.rstrip(".") for h in hl if len(h) < SHORT_HIGHLIGHT]
        main = proj.get("description", "")
        if short:
            main += " Highlights: " + "; ".join(short) + "."
        if proj.get("tech"):
            main += " Tech: " + ", ".join(proj["tech"]) + "."
        rest = {k: v for k, v in proj.items() if k not in ("description", "tech", "highlights")}
        items = list(_walk_entity("projects", proj["name"], rest, eid))
        items += [(h, "highlights", f"{eid}.highlights[{j}]") for j, h in enumerate(hl) if len(h) >= SHORT_HIGHLIGHT]
        entity("projects", eid, proj["name"], main.strip() or proj["name"], items)

    if skills := prof.get("skills"):
        rows = [(f"{_label(c).title()}: {', '.join(v)}", _label(c), f"skills.{c}") for c, v in skills.items()]
        entity("skills", "skills", "Technical skills", "Technical skills: " + "; ".join(_label(c) for c in skills), rows)

    for i, e in enumerate(prof.get("education", [])):
        heading = f"{', '.join(x for x in (e['degree'], e['field']) if x)} · {e['institution']}"
        body = f"{heading}. " + _flat({k: v for k, v in e.items() if k not in ("degree", "field", "institution", "graduated")})
        entity("education", f"education[{i}]", heading, body, [])

    if certs := prof.get("certifications"):
        rows = [(" · ".join(x for x in (c["name"], c["issuer"], c.get("date", "")) if x), "certification",
                 f"certifications[{i}]") for i, c in enumerate(certs)]
        entity("certifications", "certifications", "Certifications", "Certifications: " + "; ".join(r[0] for r in rows), rows)

    if s := prof.get("summary"):
        entity("summary", "summary", "Summary", s, [])
    if nar := prof.get("career_narrative"):
        rest = {k: v for k, v in nar.items() if k != "identity"}
        entity("narrative", "career_narrative", "Career narrative", nar.get("identity", "Career narrative"),
               _walk_entity("narrative", "", rest, "career_narrative"))

    chunks, seen = [], set()
    for level, section, heading, sub, path, parent, text in raw:
        key = _norm(text)
        if not key or (level == 2 and key in seen):   # exact repeat of an earlier item (headings always kept)
            continue
        seen.add(key)
        chunks.append(Document(page_content=text.strip(), metadata={
            "level": level, "section": section, "heading": heading, "entry": heading, "subsection": sub,
            "path": path, "parent_id": parent, "also": ""}))
    n1 = sum(d.metadata["level"] == 1 for d in chunks)
    print(f"made {len(chunks)} chunks: {n1} headings + {len(chunks) - n1} linked items")
    return {"chunks": chunks}


def build_tree(docs: list[Document]) -> dict:
    """section -> headings -> subsections -> item ids, plus every node's text (for expanding a hit)."""
    tree = {"sections": {}, "nodes": {}}
    for d in docs:
        m = d.metadata
        node = {"level": m["level"], "section": m["section"], "heading": m["heading"],
                "subsection": m["subsection"], "parent_id": m["parent_id"], "text": d.page_content}
        if m["level"] == 1:
            node["children"] = {}
            tree["sections"].setdefault(m["section"], []).append(m["path"])
        tree["nodes"][m["path"]] = node
    for pid, node in tree["nodes"].items():
        if node["level"] == 2:
            tree["nodes"][node["parent_id"]]["children"].setdefault(node["subsection"], []).append(pid)
    return tree


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))


def _embed_text(d: Document) -> str:
    m = d.metadata
    sub = _label(m["subsection"]) if m["level"] == 2 else ""
    head = " | ".join(x for x in (m["section"], m["heading"], sub) if x)
    return f"[{head}] {d.page_content}"


def merge_near_dups(state: IngestState) -> IngestState:
    if state.get("dry_run"):
        return {}
    import numpy as np
    docs = state["chunks"]
    vecs = np.array(embeddings().embed_documents([_embed_text(d) for d in docs]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    kept_idx, kept, shown = [], [], 0
    for i, d in enumerate(docs):
        if kept_idx:
            sims = vecs[kept_idx] @ vecs[i]
            j = int(sims.argmax())
            if (sims[j] >= NEAR_DUP and d.metadata["level"] == 2 and kept[j].metadata["level"] == 2
                    and kept[j].metadata["parent_id"] == d.metadata["parent_id"]                 # siblings only
                    and _numbers(kept[j].page_content) == _numbers(d.page_content)):   # different numbers = different fact
                if len(d.page_content) > len(kept[j].page_content):                  # keep the fuller wording
                    kept[j], d = d, kept[j]
                    kept_idx[j] = i
                m = kept[j].metadata
                m["also"] = "|".join(x for x in (m["also"], d.metadata["path"], d.metadata["also"]) if x)
                if shown < 10:
                    print(f"  merged ({sims[j]:.3f}):\n    KEEP {kept[j].page_content[:90]}\n    DROP {d.page_content[:90]}")
                    shown += 1
                continue
        kept_idx.append(i)
        kept.append(d)
    print(f"near-dup merge (>= {NEAR_DUP}): {len(docs)} -> {len(kept)} chunks")
    return {"chunks": kept, "vectors": vecs[kept_idx].tolist()}


def store(state: IngestState) -> IngestState:
    docs = state["chunks"]
    tree = build_tree(docs)
    if state.get("dry_run"):
        for sec, heads in tree["sections"].items():
            print(f"{sec}")
            for h in heads[:3]:
                n = tree["nodes"][h]
                subs = ", ".join(f"{s} {len(ids)}" for s, ids in n["children"].items())
                print(f"  {h:<24} {n['heading'][:45]:<45} [{subs}]")
        return {"stored": 0}
    TREE_FILE.write_text(json.dumps(tree, indent=1, ensure_ascii=False), encoding="utf-8")
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    from providers import embed_id
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embed": embed_id()})
    # container headings ("Technical skills", "Certifications") only group their items: tree yes, search no
    keep = [n for n, d in enumerate(docs) if not (d.metadata["level"] == 1 and d.metadata["path"] in CONTAINERS)]
    col.add(
        ids=[docs[n].metadata["path"] for n in keep],   # the JSON path is the id: stable, readable, = tree node id
        documents=[docs[n].page_content for n in keep],
        metadatas=[docs[n].metadata for n in keep],
        embeddings=[state["vectors"][n] for n in keep],
    )
    print(f"stored {len(keep)} chunks in {DB_DIR} ({len(docs) - len(keep)} container headings tree-only), "
          f"tree in {TREE_FILE.name}")
    return {"stored": len(keep)}


def load_tree() -> dict:
    return json.loads(TREE_FILE.read_text(encoding="utf-8"))


def expand(node_id: str, tree: dict | None = None) -> dict:
    """A hit -> its heading with every subsection and item under it (the linked context)."""
    tree = tree or load_tree()
    node = tree["nodes"][node_id]
    head_id = node_id if node["level"] == 1 else node["parent_id"]
    head = tree["nodes"][head_id]
    return {"id": head_id, "section": head["section"], "heading": head["heading"], "text": head["text"],
            "subsections": {s: [{"id": i, "text": tree["nodes"][i]["text"]} for i in ids]
                            for s, ids in head["children"].items()}}


def vectorstore():
    """Open the collection for retrieval (later steps)."""
    from langchain_chroma import Chroma
    return Chroma(collection_name=COLLECTION, embedding_function=embeddings(), persist_directory=str(DB_DIR))


def build_graph():
    g = StateGraph(IngestState)
    steps = ["load_profile", "chunk", "merge_near_dups", "store"]
    for name in steps:
        g.add_node(name, globals()[name])
    g.add_edge(START, steps[0])
    for a, b in zip(steps, steps[1:]):
        g.add_edge(a, b)
    g.add_edge(steps[-1], END)
    return g.compile()


if __name__ == "__main__":
    if not PROFILE.exists():
        sys.exit(f"missing profile: {PROFILE}")
    build_graph().invoke({"dry_run": "--dry-run" in sys.argv})
