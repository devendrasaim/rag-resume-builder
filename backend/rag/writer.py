"""The RAG's only door to an LLM: LiteLLM + an EXACT-MATCH disk cache + a cost trace.

    from writer import structured, reset_trace, trace_summary
    out = structured(SectionOut, messages, tag="experience: Larkspur Logistics")

* Provider routing (providers.py): the first provider with a key in backend/.env (OpenAI, Anthropic,
  Google) gets every call. If that call errors, the next configured provider retries it.
* Cache key = sha256(model + reasoning + every message + the output schema), stored with diskcache.
  (LiteLLM 1.109 dropped its own disk backend; this is the same exact-match idea.)
  Same JD rerun -> same prompt -> cache hit, $0. A different JD changes the prompt -> miss (by design).
  Semantic caching is deliberately NOT used: it would hand back bullets written for a similar JD.
* Cost trace: every call is priced by LiteLLM (litellm.completion_cost, its model price table) from the
  real token usage. A cache hit costs $0 now; the cache also stores what the call originally cost, so the
  trace reports how much the cache SAVED. trace_summary() is written to runs/<slug>/cost.json per build.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import diskcache
import litellm
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR as _CACHE, load_env  # noqa: E402
import providers  # noqa: E402

REASONING = "low"                                        # OpenAI reasoning models only
CACHE_DIR = _CACHE / "llm"

_trace: list[dict] = []
_ready = False
_cache: diskcache.Cache | None = None


def _setup():
    global _ready, _cache
    if _ready:
        return
    load_env()
    _cache = diskcache.Cache(str(CACHE_DIR))
    litellm.suppress_debug_info = True
    litellm.drop_params = True                           # skip params a provider doesn't support
    _ready = True


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt": 0, "completion": 0, "reasoning": 0, "cached_input": 0}
    det_out = getattr(u, "completion_tokens_details", None)
    det_in = getattr(u, "prompt_tokens_details", None)
    return {"prompt": u.prompt_tokens or 0, "completion": u.completion_tokens or 0,
            "reasoning": (getattr(det_out, "reasoning_tokens", 0) or 0) if det_out else 0,
            "cached_input": (getattr(det_in, "cached_tokens", 0) or 0) if det_in else 0}


def current_model() -> str:
    """The model that gets calls right now (the primary provider's)."""
    return providers.primary_model()


def _call(provider: str, model: str, api_key: str, schema: type[BaseModel], messages: list[dict]):
    kw = {"reasoning_effort": REASONING} if provider == "openai" else {}
    return litellm.completion(model=model, messages=messages, response_format=schema, api_key=api_key, **kw)


def structured(schema: type[BaseModel], messages: list[dict], model: str | None = None, tag: str = "") -> BaseModel:
    """Call the model, force JSON matching `schema`, return the parsed object. Cached + cost-traced.
    Goes to one provider; falls back to the next configured provider only if that one fails."""
    _setup()
    chain = providers.chat_chain()
    if not chain:
        raise RuntimeError("No LLM key: add OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY to backend/.env")
    if model:                                            # explicit model: put it first, keep the fallbacks
        prov = model.split("/", 1)[0].replace("gemini", "google")
        key = dict((p, k) for p, _, k in chain).get(prov, chain[0][2])
        chain = [(prov, model, key)] + [c for c in chain if c[1] != model]

    errors = []
    for provider, mdl, api_key in chain:
        key = hashlib.sha256(json.dumps([mdl, REASONING, messages, schema.model_json_schema()],
                                        sort_keys=True).encode()).hexdigest()
        t0 = time.time()
        hit = _cache.get(key)
        if hit is not None:
            entry = json.loads(hit) if hit.startswith('{"_v"') else {"content": hit, "cost": None, "tokens": None}
            _trace.append({"tag": tag, "model": mdl, "cached": True, "cost": 0.0, "saved": entry["cost"],
                           "tokens": entry["tokens"], "ms": int((time.time() - t0) * 1000)})
            return schema.model_validate_json(entry["content"])
        try:
            resp = _call(provider, mdl, api_key, schema, messages)
            content = resp.choices[0].message.content
            out = schema.model_validate_json(content)    # only valid JSON gets cached
        except Exception as e:                           # auth, rate limit, outage, bad JSON -> next provider
            errors.append(f"{mdl}: {type(e).__name__}: {str(e)[:200]}")
            if len(errors) < len(chain):
                print(f"  LLM {mdl} failed ({type(e).__name__}); trying the next provider", flush=True)
            continue
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost = 0.0                                   # model missing from LiteLLM's price table
        tokens = _usage(resp)
        _cache.set(key, json.dumps({"_v": 2, "content": content, "cost": cost, "tokens": tokens}))
        _trace.append({"tag": tag, "model": mdl, "cached": False, "cost": cost, "saved": 0.0,
                       "tokens": tokens, "ms": int((time.time() - t0) * 1000)})
        return out
    raise RuntimeError("Every configured LLM provider failed:\n  " + "\n  ".join(errors))


def reset_trace():
    _trace.clear()


def trace_summary() -> dict:
    """This build's spend: totals, per stage (the part of the tag before ':'), and every call."""
    stages: dict[str, dict] = {}
    for c in _trace:
        st = stages.setdefault(c["tag"].split(":")[0].strip() or "other", {"calls": 0, "cost": 0.0, "saved": 0.0})
        st["calls"] += 1
        st["cost"] += c["cost"]
        st["saved"] += c["saved"] or 0.0
    tok = lambda k: sum((c["tokens"] or {}).get(k, 0) for c in _trace if not c["cached"])
    return {"at": time.time(), "model": current_model(), "calls": len(_trace),
            "cache_hits": sum(c["cached"] for c in _trace),
            "cost": round(sum(c["cost"] for c in _trace), 6),
            "saved": round(sum(c["saved"] or 0.0 for c in _trace), 6),
            "saved_unknown": sum(c["cached"] and c["saved"] is None for c in _trace),
            "tokens": {"prompt": tok("prompt"), "completion": tok("completion"), "reasoning": tok("reasoning")},
            "stages": {k: {**v, "cost": round(v["cost"], 6), "saved": round(v["saved"], 6)} for k, v in stages.items()},
            "trace": list(_trace)}


def usage() -> str:
    s = trace_summary()
    return (f"LLM: {s['calls']} calls, {s['cache_hits']} cache hits, ${s['cost']:.4f} spent, "
            f"${s['saved']:.4f} saved by cache ({current_model()})")
