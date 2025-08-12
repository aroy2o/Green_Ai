"""Product enrichment & sustainability orchestration module.

Stages:
1. extract_product_name: Heuristic brand/product name extraction from OCR text.
2. enrich_product_data: Async lightweight web search & scraping (DuckDuckGo HTML + selected domains).
3. build_enrichment_block: Produce structured enrichment text section for LLM prompt.
4. orchestrate_sustainability_analysis: Combine OCR + enrichment and call enriched LLM analysis.

Design Goals:
- Fail fast & silently on network / parsing errors (never block core OCR pipeline).
- Respect tight time budget (default total enrichment timeout ~6-8s).
- Do NOT store secrets. Only store enrichment + OCR text (truncated) in Mongo via store_sustainability_result if desired.
- Keep dependencies optional: degrade gracefully if httpx / bs4 / rapidfuzz / spaCy not installed.

Security / Compliance:
- Simple heuristic filtering of domains; no login / JS execution.
- Honor robots implicitly by only fetching public HTML pages; (Production: integrate robots.txt check if needed.)

Enhancements:
- Centralized config usage (app.config)
- Simple in-process TTL cache for enrichment to avoid repeated network calls for same product
- Structured logging improvements with context
- Lightweight quality scoring of enrichment signals
- Stricter type hints and defensive programming
"""
from __future__ import annotations
import os, re, asyncio, time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, OrderedDict
from loguru import logger
from datetime import datetime
from dataclasses import dataclass
import json

# Central config
try:  # Local import to avoid circulars during tests
    from app.config import (
        ANALYSIS_DATE, SUSTAINABILITY_MODEL as _MODEL, SUSTAINABILITY_TEMPERATURE as _TEMPERATURE,
        SUSTAINABILITY_MAX_TOKENS as _MAX_TOKENS, ENRICH_ENABLE as ENABLE_ENRICH,
        ENRICH_MAX_PAGES as MAX_PAGES, ENRICH_SEARCH_TIMEOUT as SEARCH_TIMEOUT,
        ENRICH_FETCH_TIMEOUT as FETCH_TIMEOUT, ENRICH_TOTAL_TIMEOUT as TOTAL_BUDGET,
        ENRICH_CACHE_TTL, ENRICH_CACHE_MAX
    )
except Exception:  # pragma: no cover
    ANALYSIS_DATE = datetime.utcnow().strftime("%B %d, %Y")
    _MODEL = os.getenv("SUSTAINABILITY_MODEL", "gpt-3.5-turbo")
    _TEMPERATURE = float(os.getenv("SUSTAINABILITY_TEMPERATURE", "0.25"))
    _MAX_TOKENS = int(os.getenv("SUSTAINABILITY_MAX_TOKENS", "1536"))
    ENABLE_ENRICH = os.getenv("ENABLE_ENRICH", "1") not in ("0","false","False")
    MAX_PAGES = int(os.getenv("ENRICH_MAX_PAGES", "6"))
    SEARCH_TIMEOUT = float(os.getenv("ENRICH_SEARCH_TIMEOUT", "4.0"))
    FETCH_TIMEOUT = float(os.getenv("ENRICH_FETCH_TIMEOUT", "4.0"))
    TOTAL_BUDGET = float(os.getenv("ENRICH_TOTAL_TIMEOUT", "8.0"))
    ENRICH_CACHE_TTL = 3600.0
    ENRICH_CACHE_MAX = 64
    MIN_SCORE_AVOID_LIMIT = float(os.getenv("ENRICH_MIN_SCORE_TO_AVOID_LIMIT", "0.12"))

# Optional deps
try: import httpx  # type: ignore
except Exception: httpx = None  # type: ignore
try: from bs4 import BeautifulSoup  # type: ignore
except Exception: BeautifulSoup = None  # type: ignore
try: from rapidfuzz import fuzz  # type: ignore
except Exception: fuzz = None  # type: ignore
try: import spacy  # type: ignore
except Exception: spacy = None  # type: ignore
try: from openai import OpenAI  # type: ignore
except Exception: OpenAI = None  # type: ignore

try:
    from app.db import store_sustainability_result  # type: ignore
except Exception:  # pragma: no cover
    async def store_sustainability_result(ocr_text: str, analysis: dict):  # type: ignore
        return None

# --- Runtime NLP (optional) ---
_NLP = None
if spacy is not None:
    try:
        _NLP = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
    except Exception:  # pragma: no cover
        _NLP = None

SAFE_DOMAINS = [
    "amazon.", "flipkart.", "jiomart.", "walmart.", "wikipedia.org", "fssai", "iso.org", "usda.gov",
    "europa.eu", "who.int", "un.org", "ewg.org", "rainforest", "official", "sustainability", "recycle",
    # Added brand & nutrition related domains (heuristic, public info only)
    "pepsico", "nutritionvalue", "fooducate", "myfitnesspal", "healthline", "livestrong"
]

# Precompiled patterns
ING_PAT = re.compile(r"ingredi(?:ent|ents)[:\s]*([^\n]{0,200})", re.IGNORECASE)
ALT_ING_PAT = re.compile(r"(?:contains|made with|made from)[:\s]*([^\n]{0,160})", re.IGNORECASE)
PACK_PAT = re.compile(r"(packag|bottle|jar|pouch|plastic|recycl|glass|metal|tin|carton|foil|laminat)[^\n]{0,120}", re.IGNORECASE)
CERT_PAT = re.compile(r"(FSSAI|ISO\s?\d{3,5}|USDA Organic|Fair ?Trade|BPA[- ]?free|Rainforest Alliance|RSPO|Non-GMO|GMP|FDA|Plastic[- ]?Free)[^\n]{0,60}", re.IGNORECASE)
RECYCLE_PAT = re.compile(r"(dispose|recycl|biodegrad|compost|reuse|eco[- ]?friendly|circular|zero[- ]?waste)[^\n]{0,160}", re.IGNORECASE)
NUTRI_LINE_PAT = re.compile(r"(energy|calories|protein|carbo(?:hydrate|hydrates)|fat|sodium|sugar)\s*[:=-]?\s*([0-9]+\s*(?:kcal|g|mg)?)", re.IGNORECASE)

# In-process enrichment cache (simple LRU with TTL)
@dataclass
class _CacheEntry:
    value: Dict[str, Any]
    ts: float

_enrich_cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()

# ---------------- Cache Introspection Helpers ----------------

def get_enrichment_cache_stats() -> Dict[str, Any]:
    """Return current enrichment cache statistics (no sensitive data)."""
    now = time.time()
    entries: List[Dict[str, Any]] = []
    for k, e in list(_enrich_cache.items())[:32]:  # cap detail
        entries.append({
            "key": k,
            "age_sec": round(now - e.ts, 3),
            "has_signals": bool(e.value.get("signals")),
            "score": e.value.get("score", 0.0),
        })
    return {
        "size": len(_enrich_cache),
        "max": ENRICH_CACHE_MAX,
        "ttl_sec": ENRICH_CACHE_TTL,
        "entries_preview": entries,
    }

def clear_enrichment_cache() -> Dict[str, Any]:
    """Clear the enrichment cache and return prior size."""
    prior = len(_enrich_cache)
    _enrich_cache.clear()
    return {"cleared": prior, "now": 0}

# Public cache stats helpers
def get_enrichment_cache_stats() -> dict:
    now = time.time()
    alive = [k for k,v in _enrich_cache.items() if now - v.ts <= ENRICH_CACHE_TTL]
    return {
        "size": len(_enrich_cache),
        "alive_entries": len(alive),
        "ttl_seconds": ENRICH_CACHE_TTL,
        "max_entries": ENRICH_CACHE_MAX,
        "keys": alive[:25],
    }

def clear_enrichment_cache() -> dict:
    removed = len(_enrich_cache)
    _enrich_cache.clear()
    return {"cleared": removed}

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    e = _enrich_cache.get(key)
    if not e:
        return None
    if now - e.ts > ENRICH_CACHE_TTL:
        _enrich_cache.pop(key, None)
        return None
    # Move to end (LRU)
    _enrich_cache.move_to_end(key)
    return e.value

def _cache_put(key: str, value: Dict[str, Any]) -> None:
    _enrich_cache[key] = _CacheEntry(value=value, ts=time.time())
    _enrich_cache.move_to_end(key)
    while len(_enrich_cache) > ENRICH_CACHE_MAX:
        _enrich_cache.popitem(last=False)

# ---------------- Utility ----------------

def _normalize_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

# ---------------- Product Name Extraction ----------------

def extract_product_name(ocr_text: str) -> str:
    text = (ocr_text or "").strip()
    if not text:
        return ""
    if _NLP is not None:
        try:
            doc = _NLP(text[:5000])
            cands = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART")]
            if cands:
                cands.sort(key=lambda c: (text.find(c), len(c)))
                best = cands[0]
                if 2 <= len(best.split()) <= 6:
                    return _normalize_whitespace(best)
        except Exception:  # pragma: no cover
            pass
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cand_scores: Counter[str] = Counter()
    for idx, line in enumerate(lines[:30]):
        clean = re.sub(r"[^A-Za-z0-9 &' -]", " ", line).strip()
        if not clean:
            continue
        words = clean.split()
        if len(words) > 8 or len(words) < 1:
            continue
        cap_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        title_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
        score = cap_ratio * 1.2 + title_ratio + (1.0 / (1 + abs(idx)))
        if clean.endswith('.'):
            score *= 0.6
        cand_scores[clean] += score
    if not cand_scores:
        first = text.split()
        return first[0][:40] if first else ""
    best, _ = cand_scores.most_common(1)[0]
    return _normalize_whitespace(best)

# ---------------- Search & Fetch ----------------
async def _search_queries(product: str) -> List[str]:
    """Concurrent multi-variant search producing filtered candidate URLs.

    Strategy:
    1. Generate query variants (base + ingredients/packaging/sustainability/nutrition).
    2. Fire all queries concurrently (bounded by MAX_PAGES heuristic).
    3. Parse result HTML for <a href> targets; filter SAFE_DOMAINS and dedupe.
    4. Return up to MAX_PAGES unique URLs preserving discovery order.
    """
    if not httpx:
        return []
    core_tokens = product.split()
    core_clean = [t for t in core_tokens if len(t) > 1]
    core = " ".join(core_clean[:6])

    def _variants(base: str) -> List[str]:
        stems = [
            f"{base} ingredients",
            f"{base} packaging sustainability",
            f"{base} nutrition facts",
            f"{base} certifications eco",
            f"{base} recyclability",
            f"{base} environmental impact",
        ]
        # Brand-only fallback if base phrase long
        if len(base.split()) > 3:
            brand = " ".join(base.split()[:2])
            stems += [
                f"{brand} chips ingredients", f"{brand} chips packaging", f"{brand} chips sustainability"
            ]
        return stems

    queries = _variants(core)

    # Fire all searches concurrently
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (enrichment bot)"}) as client:
        tasks = [client.get("https://duckduckgo.com/html/", params={"q": q}) for q in queries]
        results: List[Any] = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
    urls: List[str] = []
    for r in results:
        if not hasattr(r, "status_code") or getattr(r, "status_code", 0) != 200:
            continue
        html = getattr(r, "text", "")
        for m in re.finditer(r"<a[^>]+href=\"(http[s]?://[^\" >]+)\"", html):
            url = m.group(1).split("&")[0]
            low = url.lower()
            if any(d in low for d in SAFE_DOMAINS):
                urls.append(url)
    # Stable dedupe
    seen = set(); dedup: List[str] = []
    for u in urls:
        if u in seen: continue
        seen.add(u); dedup.append(u)
        if len(dedup) >= MAX_PAGES: break
    return dedup

async def _fetch_page(url: str) -> str:
    if not httpx:
        return ""
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (enrichment bot)"}) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return ""
            return r.text[:200_000]
    except Exception:
        return ""

# ---------------- Signal Extraction ----------------

def _extract_signals(html: str) -> Dict[str, List[str]]:
    if not html:
        return {k: [] for k in ("ingredients","packaging","certifications","claims","concerns","recycling","nutrition")}
    text = html
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            # JSON-LD parsing (nutrition / ingredient lists)
            for script in soup.find_all("script", type=lambda v: v and "ld+json" in v.lower()):
                try:
                    data = json.loads(script.string or "{}")
                    # Normalize to list
                    objs = data if isinstance(data, list) else [data]
                    for obj in objs:
                        if not isinstance(obj, dict):
                            continue
                        ing_list = obj.get("ingredients") or obj.get("recipeIngredient")
                        if isinstance(ing_list, str):
                            ingredients.append(_normalize_whitespace(ing_list)[:200])
                        elif isinstance(ing_list, list):
                            for it in ing_list[:25]:
                                if isinstance(it, str):
                                    ingredients.append(_normalize_whitespace(it)[:120])
                        nutr = obj.get("nutrition")
                        if isinstance(nutr, dict):
                            for k,v in nutr.items():
                                if isinstance(v, str) and any(ch.isdigit() for ch in v):
                                    nutrition.append(f"{k}: {v}"[:60])
                except Exception:
                    continue
            for tag in soup(["script","style","noscript"]):
                tag.extract()
            text = soup.get_text("\n")
            # Meta tags (description) quick addition
            for meta in soup.find_all('meta'):
                if meta.get('name','').lower() in ("description","keywords"):
                    val = (meta.get('content') or '').strip()
                    if val:
                        text += "\n" + val
        except Exception:
            pass
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)
    ingredients = [_normalize_whitespace(m.group(1).split('.')[:1][0]) for m in ING_PAT.finditer(joined) if m.group(1)]
    # Alternate ingredient phrasing
    ingredients += [_normalize_whitespace(m.group(1).split('.')[:1][0]) for m in ALT_ING_PAT.finditer(joined) if m.group(1)]
    packaging = [_normalize_whitespace(m.group(0)) for m in PACK_PAT.finditer(joined)]
    certs = [_normalize_whitespace(m.group(0)) for m in CERT_PAT.finditer(joined)]
    recycling = [_normalize_whitespace(m.group(0)) for m in RECYCLE_PAT.finditer(joined)]
    nutrition = [f"{m.group(1).title()}: {m.group(2)}" for m in NUTRI_LINE_PAT.finditer(joined)]
    claims, concerns = [], []
    for line in lines:
        low = line.lower()
        if any(k in low for k in ["eco","sustain","recycl","organic","biodegrad","compost","bpa","gluten free","no trans fat"]):
            claims.append(_normalize_whitespace(line)[:180])
        if any(k in low for k in ["plastic","waste","non-recycl","hazard","toxic","palm oil"]):
            concerns.append(_normalize_whitespace(line)[:180])
    def uniq(seq: List[str]) -> List[str]:
        seen = set(); out=[]
        for s in seq:
            key = s.lower()
            if key in seen or not s:
                continue
            seen.add(key); out.append(s)
        return out
    return {
        "ingredients": uniq(ingredients)[:10],
        "packaging": uniq(packaging)[:12],
        "certifications": uniq(certs)[:12],
        "claims": uniq(claims)[:18],
        "concerns": uniq(concerns)[:18],
        "recycling": uniq(recycling)[:12],
        "nutrition": uniq(nutrition)[:15],
    }

# ---------------- Enrichment Aggregation ----------------
async def enrich_product_data(product_name: str) -> Dict[str, Any]:
    if not ENABLE_ENRICH or not product_name:
        return {"product_name": product_name, "sources": [], "signals": {}, "score": 0.0}
    cached = _cache_get(product_name.lower())
    if cached:
        logger.debug(f"enrichment cache hit product='{product_name}'")
        return cached
    t0 = time.time()
    try:
        urls = await asyncio.wait_for(_search_queries(product_name), timeout=SEARCH_TIMEOUT + 1)
    except Exception:
        urls = []
    pages: List[str] = []
    if urls:
        tasks = [_fetch_page(u) for u in urls]
        try:
            pages = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=False), timeout=FETCH_TIMEOUT + 1)
        except Exception:
            pages = []
    aggregated = {k: [] for k in ("ingredients","packaging","certifications","claims","concerns","recycling","nutrition")}
    for html in pages:
        sig = _extract_signals(html)
        for k, arr in sig.items():
            aggregated[k].extend(arr)
    # SECOND PASS: If still weak signals, broaden queries (drop flavor words etc.)
    score_initial = _score_enrichment(aggregated)
    if score_initial < 0.15:
        base = product_name.split()
        core_short = " ".join([w for w in base if w.lower() not in {"treat","spicy","flavour","flavor","lips","se"}][:3])
        if core_short and core_short.lower() != product_name.lower():
            try:
                extra_urls = await asyncio.wait_for(_search_queries(core_short), timeout=SEARCH_TIMEOUT)
            except Exception:
                extra_urls = []
            for u in extra_urls:
                if u not in urls:
                    urls.append(u)
            new_pages: List[str] = []
            if extra_urls:
                try:
                    new_pages = await asyncio.wait_for(asyncio.gather(*[_fetch_page(u) for u in extra_urls], return_exceptions=False), timeout=FETCH_TIMEOUT)
                except Exception:
                    new_pages = []
            for html in new_pages:
                sig2 = _extract_signals(html)
                for k, arr in sig2.items():
                    aggregated[k].extend(arr)
    # Dedup aggregated
    for k, arr in aggregated.items():
        out=[]; seen=set()
        for item in arr:
            low = item.lower()
            if low in seen:
                continue
            seen.add(low); out.append(item)
        aggregated[k] = out
    score = _score_enrichment(aggregated)
    payload = {"product_name": product_name, "sources": urls, "signals": aggregated, "elapsed": round(time.time()-t0,3), "score": score}
    _cache_put(product_name.lower(), payload)
    return payload

# ---------------- Enrichment Scoring ----------------

def _score_enrichment(sig: Dict[str, List[str]]) -> float:
    if not sig:
        return 0.0
    weights = {"ingredients":3, "packaging":2, "certifications":3, "claims":1, "concerns":2, "recycling":2, "nutrition":2}
    total = 0.0; max_total = 0.0
    for k, w in weights.items():
        n = len(sig.get(k, []))
        total += w * min(n, 5)
        max_total += w * 5
    return round((total / max_total) if max_total else 0.0, 3)

# ---------------- Profile & Prompt Building ----------------

def build_enrichment_block(enrichment: Dict[str, Any]) -> str:
    if not enrichment:
        return ""
    sig = enrichment.get("signals", {}) or {}
    lines = ["== Enriched Product Data =="]
    lines.append(f"Product Name: {enrichment.get('product_name','')}")
    def add_section(title: str, items: List[str]):
        if items:
            lines.append(f"{title}:")
            for it in items[:10]:
                lines.append(f"- {it}")
            lines.append("")
    add_section("Ingredients", sig.get("ingredients", []))
    add_section("Packaging", sig.get("packaging", []))
    add_section("Certifications", sig.get("certifications", []))
    add_section("Manufacturer / Retailer Claims", sig.get("claims", []))
    add_section("Sustainability Concerns", sig.get("concerns", []))
    add_section("Recycling / Disposal", sig.get("recycling", []))
    add_section("Nutrition", sig.get("nutrition", []))
    if enrichment.get("sources"):
        lines.append("Source URLs:")
        for u in enrichment["sources"][:12]:
            lines.append(f"- {u}")
    return "\n".join(lines)

def _build_structured_profile(enrichment: Dict[str, Any], product_name: str, ocr_text: str) -> Dict[str, Any]:
    sig = (enrichment or {}).get("signals", {}) or {}
    return {
        "product": product_name,
        "ingredients": sig.get("ingredients", []),
        "packaging": sig.get("packaging", []),
        "certifications_claims": list({*sig.get("certifications", []), *sig.get("claims", [])}),
        "sustainability_concerns_reviews": list({*sig.get("concerns", []), *sig.get("recycling", [])}),
        "nutrition_facts": sig.get("nutrition", []),
        "sources": (enrichment or {}).get("sources", []),
        "raw_ocr_excerpt": ocr_text[:1000],
        "enrichment_score": enrichment.get("score", 0.0),
    }

_ENRICHED_SYSTEM_PROMPT = (
    "You are EcoAnalyst AI, an impartial, evidence-based product sustainability expert. "
    "Use ONLY the provided structured product profile (extracted text + scraped signals). "
    "Never invent ingredients, packaging, certifications, claims, or sources not present. "
    "For every insight, cite a source URL from the provided list or embedded signals. "
    "If packaging contains plastic/composite terms (plastic, PET, pouch, laminate, foil) produce at least one negative packaging concern. "
    "If data is truly sparse (no ingredients, no packaging, no certifications, no claims, no concerns) then mark limited_analysis true and provide the standard limitation message referencing EWG. "
    "Return BOTH a readable markdown section AND a fenced JSON code block labeled JSON with schema specified."
)

_JSON_SCHEMA_SNIPPET = (
    '{\n'
    '  "product_name": string,\n'
    '  "date": string,\n'
    '  "score": string,\n'
    '  "summary": string,\n'
    '  "positives": [string, ...],\n'
    '  "negatives": [string, ...],\n'
    '  "explanations": {"positives": [{"title": string, "explanation": string, "source_url": string}], "negatives": [{"title": string, "explanation": string, "source_url": string}]},\n'
    '  "recommendations": [string, ...],\n'
    '  "limited_analysis": boolean\n'
    '}'
)

def _build_messages_from_profile(profile: Dict[str, Any]) -> List[Dict[str, str]]:
    bullets = []
    def add(label: str, items: List[str]):
        if items:
            bullets.append(f"{label}:\n" + "\n".join(f"- {i}" for i in items))
    add("Ingredients", profile.get("ingredients", []))
    add("Packaging", profile.get("packaging", []))
    add("Certifications/Claims", profile.get("certifications_claims", []))
    add("Concerns/Reviews", profile.get("sustainability_concerns_reviews", []))
    add("Nutrition", profile.get("nutrition_facts", []))
    # Encourage model not to mark limited if any enrichment
    if profile.get("enrichment_score", 0) > 0:
        bullets.append(f"Enrichment Score: {profile.get('enrichment_score')}")
    sources = profile.get("sources", [])
    src_block = "\n".join(f"- {u}" for u in sources) if sources else ""
    user = (
        f"Product: {profile.get('product','')}\nDate of Analysis: {ANALYSIS_DATE}\nEnrichment Score: {profile.get('enrichment_score',0.0)}\n\n" +
        "Structured Signals:\n" + ("\n\n".join(bullets) if bullets else "(No structured signals extracted)") +
        ("\n\nSources:\n" + src_block if src_block else "") +
        "\n\nTask: Produce lifecycle sustainability analysis with: Score (Low/Moderate/High Concern), Summary (1-2 sentences), at least 5 positive insights (if data permits), at least 1 negative insight if any packaging/concern signals, 2-3 recommendations (only if not limited), explanations with 2-3 sentences each, and cite direct source_url for every explanation. If any structured signals or sources are present DO NOT mark limited_analysis true. JSON schema: " + _JSON_SCHEMA_SNIPPET + "\nUse product_name exactly as provided."
    )
    return [
        {"role": "system", "content": _ENRICHED_SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]

_JSON_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def _extract_json(markdown: str) -> Dict[str, Any]:
    if not markdown:
        return {}
    m = _JSON_BLOCK_RE.search(markdown)
    if not m:
        return {}
    block = m.group(1)
    try:
        import json
        return json.loads(block)
    except Exception:
        return {}

MANDATORY_KEYS = {"product_name", "date", "score", "summary", "positives", "negatives", "explanations", "recommendations", "limited_analysis"}

def _coerce_payload(data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {
            "product_name": profile.get("product", ""),
            "date": ANALYSIS_DATE,
            "score": "",
            "summary": "",
            "positives": [],
            "negatives": [],
            "explanations": {"positives": [], "negatives": []},
            "recommendations": [],
            "limited_analysis": True,
            "error": "LLM parsing failed"
        }
    data.setdefault("product_name", profile.get("product", ""))
    data.setdefault("date", ANALYSIS_DATE)
    data.setdefault("positives", [])
    data.setdefault("negatives", [])
    data.setdefault("explanations", {"positives": [], "negatives": []})
    data.setdefault("recommendations", [])
    data.setdefault("limited_analysis", False)
    for k in ("positives","negatives","recommendations"):
        if not isinstance(data.get(k), list):
            data[k] = []
    ex = data.get("explanations") or {}
    if not isinstance(ex, dict):
        ex = {"positives": [], "negatives": []}
    ex.setdefault("positives", [])
    ex.setdefault("negatives", [])
    data["explanations"] = ex
    return data

async def _call_llm(messages: List[Dict[str, str]]) -> str:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        client = OpenAI()
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=_MODEL,
            messages=messages,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        return resp.choices[0].message.content  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.debug(f"LLM error: {e}")
        return ""

# ---------------- Orchestrator ----------------
async def orchestrate_sustainability_analysis(ocr_text: str) -> Dict[str, Any]:
    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        return {
            "product_name": "",
            "date": ANALYSIS_DATE,
            "score": "",
            "summary": "",
            "positives": [],
            "negatives": [],
            "recommendations": [],
            "explanations": {"positives": [], "negatives": []},
            "limited_analysis": True,
            "error": None,
        }
    product_name = extract_product_name(ocr_text)
    enrichment: Dict[str, Any] = {}
    try:
        if ENABLE_ENRICH and product_name:
            budget = max(2.0, TOTAL_BUDGET - 2.0)
            enrichment = await asyncio.wait_for(enrich_product_data(product_name), timeout=budget)
    except Exception as e:  # pragma: no cover
        logger.debug(f"Enrichment failed: {e}")
        enrichment = {"error": str(e)}
    profile = _build_structured_profile(enrichment, product_name, ocr_text)
    messages = _build_messages_from_profile(profile)
    raw_output = await _call_llm(messages)
    parsed = _extract_json(raw_output)
    coerced = _coerce_payload(parsed, profile)
    # Override limited flag if enrichment produced sufficient signals
    if coerced.get("limited_analysis") and profile.get("enrichment_score", 0) >= MIN_SCORE_AVOID_LIMIT:
        coerced["limited_analysis"] = False
        if not coerced.get("summary"):
            coerced["summary"] = "Auto-upgraded from limited analysis due to successful external enrichment signals."
    # ...existing code...
    try:
        await store_sustainability_result(ocr_text, {**coerced, "_profile": profile})
    except Exception:  # pragma: no cover
        pass
    public = {k: v for k, v in coerced.items() if not k.startswith("_")}
    return public

__all__ = [
    "extract_product_name",
    "enrich_product_data",
    "build_enrichment_block",
    "orchestrate_sustainability_analysis",
    "get_enrichment_cache_stats",
    "clear_enrichment_cache",
]
