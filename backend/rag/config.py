"""All paths and settings in one place. Everything personal lives in backend/data/ (gitignored).

    backend/.env          ONE of OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY (more = fallbacks)
                          <- copy .env.example; routing lives in providers.py
    backend/data/         profile.json (your uploaded resume, normalized), source file, ChromaDB, caches, runs

Optional env vars:
    RAG_DATA_DIR   where your data lives (default backend/data)
    RAG_ENV        a different .env file to load
    RAG_BROWSER    path to Chrome / Edge / Chromium for PDF printing (auto-detected otherwise)
    RAG_API_PORT   backend port (default 8710)
"""
import os
from pathlib import Path

RAG = Path(__file__).resolve().parent            # backend/rag
BACKEND = RAG.parent                             # backend
APP = BACKEND.parent                             # repo root

DATA = Path(os.environ.get("RAG_DATA_DIR", BACKEND / "data"))
PROFILE = DATA / "profile.json"
SOURCE_META = DATA / "source.json"
DB_DIR = DATA / "chroma_db"
TREE_FILE = DATA / "tree.json"
CACHE_DIR = DATA / "cache"
RUNS = DATA / "runs"
ALIASES = RAG / "aliases.json"                   # shared, generic tech synonyms (not personal)
API_PORT = int(os.environ.get("RAG_API_PORT", "8710"))

for d in (DATA, RUNS, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_env():
    from dotenv import load_dotenv
    for f in (os.environ.get("RAG_ENV"), BACKEND / ".env"):
        if f and Path(f).exists():
            load_dotenv(f, override=False)
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPEN_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]


def missing_keys() -> list[str]:
    from providers import missing_keys as _missing   # lazy: providers imports this module
    return _missing()
