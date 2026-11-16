"""Which LLM / embedding provider to use, from the API keys you put in backend/.env.

One key is enough. Add any of these (1 is required, the others are optional fallbacks):

    OPENAI_API_KEY      -> openai/gpt-5.4                 + text-embedding-3-small
    ANTHROPIC_API_KEY   -> anthropic/claude-sonnet-5-5    (no embedding API: uses another key's, or local)
    GOOGLE_API_KEY      -> gemini/gemini-2.5-pro          + gemini-embedding-001

Routing: every call goes to ONE provider, the first available in priority order (OpenAI, Anthropic,
Google). If that call fails (bad key, rate limit, outage), the next configured provider takes over.

Optional overrides:
    LLM_PROVIDER     openai | anthropic | google   put this provider first
    EMBED_PROVIDER   openai | google | local        pick the embedding backend
    OPENAI_MODEL / ANTHROPIC_MODEL / GEMINI_MODEL   any LiteLLM model string for that provider
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_env  # noqa: E402

ORDER = ("openai", "anthropic", "google")
KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
CHAT_MODEL = {"openai": ("OPENAI_MODEL", "openai/gpt-5.4"),
              "anthropic": ("ANTHROPIC_MODEL", "anthropic/claude-sonnet-5-5"),
              "google": ("GEMINI_MODEL", "gemini/gemini-2.5-pro")}
EMBED_MODEL = {"openai": "text-embedding-3-small",
               "google": "models/gemini-embedding-001",
               "local": "all-MiniLM-L6-v2"}          # ChromaDB's built-in ONNX model, downloaded once (~80 MB)


def _key(p: str) -> str:
    v = os.environ.get(KEY[p], "").strip()
    if not v and p == "google":
        v = os.environ.get("GEMINI_API_KEY", "").strip()
    return v


def available() -> list[str]:
    """Providers with a key, in routing order (LLM_PROVIDER first if set)."""
    load_env()
    order = list(ORDER)
    pref = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if pref in order:
        order.remove(pref)
        order.insert(0, pref)
    return [p for p in order if _key(p)]


def missing_keys() -> list[str]:
    """[] when at least one LLM key is set; otherwise one line naming the choices."""
    return [] if available() else ["one of OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY"]


def chat_model(p: str) -> str:
    env, default = CHAT_MODEL[p]
    return os.environ.get(env, "").strip() or default


def chat_chain() -> list[tuple[str, str, str]]:
    """[(provider, litellm model, api key), ...] primary first, then fallbacks."""
    return [(p, chat_model(p), _key(p)) for p in available()]


def primary_model() -> str:
    chain = chat_chain()
    return chain[0][1] if chain else "no LLM key set"


def embed_provider() -> str:
    load_env()
    forced = os.environ.get("EMBED_PROVIDER", "").strip().lower()
    if forced == "local" or (forced in ("openai", "google") and _key(forced)):
        return forced
    for p in available():
        if p in ("openai", "google"):
            return p
    return "local"                                   # e.g. only an Anthropic key


def embed_id() -> str:
    """Identifies the vector space. The index and query cache are only valid for the id they were built with."""
    p = embed_provider()
    return f"{p}:{EMBED_MODEL[p]}"


class Embedder:
    """One interface over the three backends. task_type is a Gemini hint; the others ignore it."""

    def __init__(self):
        self.provider = embed_provider()
        self.id = embed_id()
        p = self.provider
        if p == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._impl = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL[p], google_api_key=_key(p))
        elif p == "openai":
            from langchain_openai import OpenAIEmbeddings
            self._impl = OpenAIEmbeddings(model=EMBED_MODEL[p], api_key=_key(p))
        else:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._impl = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "google":
            return self._impl.embed_documents(texts, task_type=task_type) if task_type \
                else self._impl.embed_documents(texts)
        if self.provider == "openai":
            return self._impl.embed_documents(texts)
        return [list(map(float, v)) for v in self._impl(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text], task_type="RETRIEVAL_QUERY")[0]


def describe() -> dict:
    chain = chat_chain()
    return {"llm": chain[0][1] if chain else None, "fallbacks": [m for _, m, _ in chain[1:]],
            "embeddings": embed_id()}
