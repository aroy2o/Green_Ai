"""
Sustainability Analysis Module (backend-only)

- Builds an enhanced prompt for rigorous, evidence-based lifecycle assessment
  using ONLY OCR-extracted text.
- Calls OpenAI (GPT-3.5-turbo) with temperature 0.25, max_tokens 1536.
- Parses markdown/JSON into a structured dict suitable for frontend rendering.
- Provides fallback parsing and retry logic when responses don't conform.
- Best-effort MongoDB logging for trace/compliance (API key never logged).

Keep all sustainability logic confined to this module.
"""
from __future__ import annotations

import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.config import (
    ANALYSIS_DATE,
    SUSTAINABILITY_MODEL as _MODEL,
    SUSTAINABILITY_TEMPERATURE as _TEMPERATURE,
    SUSTAINABILITY_MAX_TOKENS as _MAX_TOKENS,
)

# Optional: load .env at import-time if available (safe no-op otherwise)
try:
    from dotenv import load_dotenv  # type: ignore
    # Attempt to load from common locations (project root, cwd)
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    load_dotenv(os.path.join(_root, ".env"))
    load_dotenv()  # fallback to CWD
except Exception:  # pragma: no cover
    pass

# OpenAI Python SDK v1.x
try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    logger.warning(f"OpenAI SDK not available: {e}")

# Optional Mongo (best-effort logging)
try:
    from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
except Exception:  # pragma: no cover
    AsyncIOMotorClient = None  # type: ignore

ANALYSIS_DATE = ANALYSIS_DATE  # ensure legacy references use central config
_MODEL = _MODEL
_TEMPERATURE = float(os.getenv("SUSTAINABILITY_TEMPERATURE", "0.25"))
_MAX_TOKENS = int(os.getenv("SUSTAINABILITY_MAX_TOKENS", "1536"))
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_MONGO_DB = os.getenv("MONGO_DB", "ocr_pipeline")
_MONGO_COLL = os.getenv("SUSTAINABILITY_COLLECTION", "sustainability_analyses")

# --- Prompt ---
_SYSTEM_PROMPT = (
    "You are EcoAnalyst AI, an impartial, evidence-driven sustainability engine.\n"
    "Your task: Analyze the product using ONLY the input text from OCR extraction; never assume beyond provided info.\n\n"
    "Guiding principles:\n"
    "- All insights must be supported by info found in the OCR text.\n"
    "- Do NOT make unsupported guesses about ingredients, packaging, or process.\n"
    "- Structure the analysis in lifecycle stages:\n"
    "  1. Sourcing & Ingredients\n"
    "  2. Packaging\n"
    "  3. End-of-Life considerations\n\n"
    "For each stage:\n"
    "- List positives with a short, factual bullet/explanation.\n"
    "- List negatives/concerns if present (especially for plastics, non-recyclable materials, water-intensive/palm oil ingredients, etc).\n"
    "- For every point, provide a direct citation to a reputable source. Use this priority order: Tier 1: Certification bodies (e.g., USDA, RSPO), then Tier 2: Government/EU/UN agencies, then peer-reviewed literature/major NGOs (WWF, EWG), then respected eco-journalism (NatGeo, The Guardian).\n\n"
    "If input is insufficient for meaningful analysis (e.g., just a product name or insufficient text):\n"
    "- Immediately use the \"Limited Analysis Protocol\" section at end of output (includes a standard EWG guide link).\n\n"
    "Output format:\n"
    "Product Name: [from OCR]\n"
    f"Date of Analysis: {ANALYSIS_DATE}\n"
    "Overall Sustainability Score: [Very Low Concern / Low / Moderate / High Concern]\n"
    "Summary: [Key findings in 1-2 neutral sentences]\n\n"
    "Positive Sustainability Insights:\n"
    "Insight 1\n\n"
    "Insight 2\n\n"
    "Negative Sustainability Insights / Concerns:\n"
    "Concern 1\n\n"
    "(If no negatives: \"No credible negative sustainability concerns were found based on the provided text.\")\n\n"
    "Detailed Explanations & Sources:\n"
    "Positives:\n\n"
    "Insight 1: [Explanation; 1-3 sentences] [Source: Direct URL]\n"
    "...\n\n"
    "Negatives:\n\n"
    "Concern 1: [Explanation; 1-3 sentences] [Source: Direct URL]\n"
    "...\n\n"
    "Considerations for a More Sustainable Choice:\n"
    "For those prioritizing [aspect], you may also consider: [Product], [reason]. [Source: direct URL]\n"
    "...\n\n"
    "Limited Analysis Protocol (ONLY if insufficient info):\n"
    "Analysis Limited: The provided text was not sufficient for a detailed sustainability report. Generally, focus on ingredients, packaging, and corporate transparency. See EWG consumer guides for more: https://www.ewg.org/consumer-guides\n\n"
    "- Remain neutral and never use speculative or marketing language.\n"
    "- Never use competitor- or alternative-centric language in recommendations.\n"
    f"- Use only the date provided: {ANALYSIS_DATE}.\n\n"
    "Also include, after the human-readable section, a machine-readable JSON block in a fenced code block marked with `json` labeled exactly as \"JSON\" that conforms to the following minimal schema (use keys as given, keep strings concise):\n"
    "{\n"
    "  \"product_name\": string,\n"
    "  \"date\": string,\n"
    "  \"score\": string,\n"
    "  \"summary\": string,\n"
    "  \"positives\": [string, ...],\n"
    "  \"negatives\": [string, ...],\n"
    "  \"explanations\": {\n"
    "    \"positives\": [{\"title\": string, \"explanation\": string, \"source_url\": string}],\n"
    "    \"negatives\": [{\"title\": string, \"explanation\": string, \"source_url\": string}]\n"
    "  },\n"
    "  \"recommendations\": [string, ...],\n"
    "  \"limited_analysis\": boolean\n"
    "}\n"
)

# ---------- Pydantic response model (for FastAPI response_model if desired) ----------
try:
    from pydantic import BaseModel, Field  # type: ignore

    class ExplanationItem(BaseModel):
        title: str
        explanation: str
        source_url: str

    class Explanations(BaseModel):
        positives: List[ExplanationItem] = Field(default_factory=list)
        negatives: List[ExplanationItem] = Field(default_factory=list)

    class SustainabilityResult(BaseModel):
        product_name: str = ""
        date: str = ANALYSIS_DATE
        score: str = ""
        summary: str = ""
        positives: List[str] = Field(default_factory=list)
        negatives: List[str] = Field(default_factory=list)
        explanations: Explanations = Field(default_factory=Explanations)
        recommendations: List[str] = Field(default_factory=list)
        limited_analysis: bool = False
        raw_markdown: Optional[str] = None
except Exception:  # pragma: no cover
    SustainabilityResult = dict  # fallback typing if pydantic missing


def _looks_sparse(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # Heuristic: <= 4 words and no punctuation suggests a bare product name
    words = re.findall(r"\b\w+\b", t)
    return len(words) <= 4


def _build_messages(ocr_text: str) -> List[Dict[str, str]]:
    user_parts = ["OCR Product Text:\n<<<\n", (ocr_text or "").strip(), "\n>>>\n\n"]
    if _looks_sparse(ocr_text):
        user_parts.append(
            "Note: The OCR text appears sparse (possibly only a product name).\n"
            "Follow the 'Limited Analysis Protocol' as specified if you cannot substantiate claims from the text.\n"
        )
    user_parts.append(
        "Deliver both the formatted human-readable section AND a fenced `json` code block titled JSON with the machine-readable object.\n"
        "Ensure every claim has a direct source URL per the citation hierarchy."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "".join(user_parts)},
    ]


def _extract_json_block(markdown: str) -> Optional[Dict[str, Any]]:
    if not markdown:
        return None
    # Find a fenced code block with json
    pattern = re.compile(r"```(?:json|JSON)?\s*\n(\{[\s\S]*?\})\s*\n```", re.IGNORECASE)
    m = pattern.search(markdown)
    if not m:
        return None
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        # Try to find the outermost JSON object braces as fallback
        try:
            start = block.find("{")
            end = block.rfind("}")
            if start != -1 and end != -1:
                return json.loads(block[start : end + 1])
        except Exception:
            return None
    return None


def _parse_headings(markdown: str) -> Dict[str, Any]:
    # Fallback markdown parser if no JSON block is present
    data: Dict[str, Any] = {
        "product_name": "",
        "date": ANALYSIS_DATE,
        "score": "",
        "summary": "",
        "positives": [],
        "negatives": [],
        "explanations": {"positives": [], "negatives": []},
        "recommendations": [],
        "limited_analysis": False,
        "raw_markdown": markdown,
    }

    lines = [l.strip() for l in (markdown or "").splitlines()]

    def find_value(prefix: str) -> str:
        for ln in lines:
            if ln.lower().startswith(prefix.lower()):
                return ln.split(":", 1)[1].strip()
        return ""

    data["product_name"] = find_value("Product Name")
    data["date"] = find_value("Date of Analysis") or ANALYSIS_DATE
    data["score"] = find_value("Overall Sustainability Score")
    data["summary"] = find_value("Summary")

    # Gather bullet-like lines under known headers
    def collect_section(header: str) -> List[str]:
        res: List[str] = []
        try:
            start = next(i for i, l in enumerate(lines) if l.lower().startswith(header.lower())) + 1
        except StopIteration:
            return res
        for j in range(start, len(lines)):
            if not lines[j] or re.match(r"^[A-Za-z].*:|^#", lines[j]):
                # Stop at next top-level label-like line
                break
            if lines[j].startswith("-") or lines[j].startswith("•"):
                res.append(lines[j].lstrip("-• "))
            elif lines[j]:
                res.append(lines[j])
        return res

    data["positives"] = [s for s in collect_section("Positive Sustainability Insights") if s]
    data["negatives"] = [s for s in collect_section("Negative Sustainability Insights") if s]

    # Explanations & Sources
    def collect_explanations(prefix: str) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        try:
            start = next(i for i, l in enumerate(lines) if l.lower().startswith(prefix.lower())) + 1
        except StopIteration:
            return items
        for j in range(start, len(lines)):
            ln = lines[j]
            if not ln:
                continue
            if re.match(r"^[A-Za-z].*:|^#", ln) and not ln.lower().startswith("insight") and not ln.lower().startswith("concern"):
                break
            # Expected pattern: Title: explanation [Source: URL]
            m = re.match(r"([^:]+):\s*(.*?)(?:\s*\[\s*Source\s*:\s*(.*?)\s*\])?$", ln, re.IGNORECASE)
            if m:
                title = m.group(1).strip()
                rest = m.group(2).strip()
                url = (m.group(3) or "").strip()
                items.append({"title": title, "explanation": rest, "source_url": url})
        return items

    # Locate subsection anchors under "Detailed Explanations & Sources:"
    pos_items = collect_explanations("Insight")
    neg_items = collect_explanations("Concern")
    if not pos_items or not neg_items:
        # Try broader search by scanning entire section
        pass
    data["explanations"] = {"positives": pos_items, "negatives": neg_items}

    # Recommendations
    data["recommendations"] = collect_section("Considerations for a More Sustainable Choice")

    # Limited Analysis Protocol presence
    lim = any("Limited Analysis Protocol" in l for l in lines) and any(
        "Analysis Limited" in l for l in lines
    )
    data["limited_analysis"] = bool(lim)

    return data


async def _maybe_log_to_mongo(ocr_text: str, payload: Dict[str, Any]) -> None:
    if AsyncIOMotorClient is None:
        return
    try:
        client = AsyncIOMotorClient(_MONGO_URI)
        db = client[_MONGO_DB]
        doc = {
            "timestamp": datetime.utcnow(),
            "ocr_text": (ocr_text or "")[:20000],  # trim
            "analysis": payload,
            "model": _MODEL,
        }
        await db[_MONGO_COLL].insert_one(doc)
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug(f"Mongo logging skipped: {e}")


def _client_factory() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; sustainability analysis will not function until configured in .env or environment.")
        return None
    try:
        # Explicitly pass API key from env for clarity
        return OpenAI(api_key=api_key)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to init OpenAI client: {e}")
        return None


def _call_openai_sync(messages: List[Dict[str, str]]) -> str:
    client = _client_factory()
    if client is None:
        raise RuntimeError("OpenAI client unavailable. Set OPENAI_API_KEY in your environment or .env file.")

    # Prefer Chat Completions API for gpt-3.5-turbo
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    except Exception as e1:
        logger.debug(f"chat.completions failed, trying responses API: {e1}")
        # Fallback to Responses API
        try:
            resp2 = client.responses.create(
                model=_MODEL,
                input={
                    "messages": messages,
                },
                temperature=_TEMPERATURE,
                max_output_tokens=_MAX_TOKENS,
            )
            # Concatenate text output segments
            parts: List[str] = []
            for out in resp2.output or []:
                if hasattr(out, "content"):
                    for c in out.content:
                        if getattr(c, "type", "") == "output_text":
                            parts.append(getattr(c, "text", ""))
            return "".join(parts)
        except Exception as e2:
            raise RuntimeError(f"OpenAI call failed: {e2}")


def _coerce_to_model(data: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure minimal keys exist
    coerced = {
        "product_name": data.get("product_name", ""),
        "date": data.get("date", ANALYSIS_DATE),
        "score": data.get("score", ""),
        "summary": data.get("summary", ""),
        "positives": data.get("positives", []) or [],
        "negatives": data.get("negatives", []) or [],
        "explanations": data.get("explanations", {"positives": [], "negatives": []}) or {"positives": [], "negatives": []},
        "recommendations": data.get("recommendations", []) or [],
        "limited_analysis": bool(data.get("limited_analysis", False)),
    }
    # Validate explanations items shape
    for k in ("positives", "negatives"):
        items = coerced["explanations"].get(k, []) or []
        cleaned = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cleaned.append({
                "title": str(it.get("title", "")).strip(),
                "explanation": str(it.get("explanation", "")).strip(),
                "source_url": str(it.get("source_url", "")).strip(),
            })
        coerced["explanations"][k] = cleaned
    return coerced


async def analyze_sustainability(ocr_text: str) -> Dict[str, Any]:
    """Run the sustainability analysis given OCR text and return a structured JSON dict.

    This function is async; OpenAI call runs in a thread to avoid blocking the loop.
    """
    messages = _build_messages(ocr_text)

    # First attempt
    try:
        content = await asyncio.to_thread(_call_openai_sync, messages)
    except Exception as e:
        # Return concise, schema-compliant error for user follow-up
        result = _coerce_to_model({})
        result.update({
            "error": f"LLM call failed: {e}",
            "limited_analysis": _looks_sparse(ocr_text),
        })
        await _maybe_log_to_mongo(ocr_text, result)
        return result

    parsed = _extract_json_block(content)
    if not parsed:
        parsed = _parse_headings(content)
    result = _coerce_to_model(parsed)
    result["raw_markdown"] = content

    # Basic validation: must have product_name AND either positives or negatives or limited_analysis
    valid = bool(result.get("product_name")) and (
        result.get("positives") or result.get("negatives") or result.get("limited_analysis")
    )

    if not valid:
        # Retry with stricter instruction to return JSON only
        retry_messages = [
            messages[0],
            {
                "role": "user",
                "content": (
                    f"Reformat the previous analysis for this OCR text into JSON ONLY per the provided schema.\n"
                    f"Return nothing except a single fenced `json` block labeled JSON.\n\n"
                    f"OCR Text (unchanged):\n<<<\n{ocr_text}\n>>>\n"
                ),
            },
        ]
        try:
            retry_content = await asyncio.to_thread(_call_openai_sync, retry_messages)
            parsed2 = _extract_json_block(retry_content)
            if parsed2:
                result = _coerce_to_model(parsed2)
                result["raw_markdown"] = content
        except Exception:
            # keep previous result
            pass

    # Best-effort logging (no API key exposure)
    try:
        await _maybe_log_to_mongo(ocr_text, result)
    except Exception:
        pass

    return result


__all__ = [
    "analyze_sustainability",
    "SustainabilityResult",
]
