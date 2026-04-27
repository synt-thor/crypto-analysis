"""Real-time news brief auto-builder.

Pipeline:
  RSS feeds (no key) → keyword filter → Gemini 2.0 Flash JSON-mode batch call
  → news_brief dict matching the schema consumed by signals/news.py.

Falls back to None on any failure so the caller (streamlit_app.py) can show
a "📁 Disk baseline" badge instead of crashing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Free, no-key RSS sources covering crypto, macro, and politics/geopolitics.
RSS_SOURCES: dict[str, str] = {
    # Crypto-native
    "coindesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    # US macro / Fed
    "fed_press":     "https://www.federalreserve.gov/feeds/press_all.xml",
    "bbc_business":  "http://feeds.bbci.co.uk/news/business/rss.xml",
    # Politics & geopolitics (Trump policy, sanctions, Middle East, China, etc.)
    "bbc_world":     "http://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_politics":  "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "reuters_world": "https://feeds.reuters.com/reuters/worldNews",
}

# Headlines must contain at least one whitelist token to make the cut.
KEYWORD_WHITELIST: tuple[str, ...] = (
    # Crypto direct
    "bitcoin", "btc", "crypto", "ether", "eth", "etf", "halving",
    "stablecoin", "blackrock", "fidelity", "ibit", "spot etf", "mstr", "strategy",
    # Trump / geopolitics (user emphasised volatility here)
    "trump", "tariff", "sanction", "executive order", "white house",
    "china", "iran", "russia", "north korea", "israel", "gaza",
    "ukraine", "taiwan", "treasury secretary",
    # Macro
    "fed", "fomc", "powell", "rate cut", "rate hike", "rate decision",
    "cpi", "pce", "ppi", "gdp", "unemployment", "payroll",
    "yield", "treasury", "recession", "inflation",
)

# Try newer models first, fall back to older if not available on the user's tier.
# As of 2026 Google rotates names rapidly; the auto-tracking aliases (`-latest`)
# are usually the safest. We probe through specific versions too in case the
# alias isn't available for a particular project.
MODEL_FALLBACK_CHAIN: tuple[str, ...] = (
    # Current GA series
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    # Older but often still enabled
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
    # Last-resort 1.5 series (may be sunset)
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
)
DEFAULT_MODEL = MODEL_FALLBACK_CHAIN[0]


# ─── 1. Fetch ────────────────────────────────────────────────────────────────


def fetch_headlines(window_hours: int = 24, max_per_source: int = 15) -> list[dict[str, Any]]:
    """Pull recent items from each RSS source. Resilient to single-source failures."""
    import feedparser

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
    out: list[dict[str, Any]] = []
    for name, url in RSS_SOURCES.items():
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", name, e)
            continue
        for entry in parsed.entries[:max_per_source]:
            ts = _parse_entry_time(entry)
            if ts is None or ts < cutoff:
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            out.append({
                "source": name,
                "title": title,
                "summary": summary[:300],
                "published": ts.isoformat(),
                "link": entry.get("link", ""),
            })
    return out


def _parse_entry_time(entry: Any) -> datetime | None:
    parsed_t = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_t:
        try:
            return datetime(*parsed_t[:6], tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _filter_relevant(headlines: list[dict[str, Any]], cap: int = 50) -> list[dict[str, Any]]:
    """Keep items whose title or summary contains a whitelist keyword."""
    kept: list[dict[str, Any]] = []
    for h in headlines:
        text = (h["title"] + " " + h.get("summary", "")).lower()
        if any(kw in text for kw in KEYWORD_WHITELIST):
            kept.append(h)
            if len(kept) >= cap:
                break
    return kept


# ─── 2. Score with Gemini ────────────────────────────────────────────────────


_SYSTEM_INSTRUCTION = (
    "You are a quantitative analyst for BTC futures markets. You read recent "
    "headlines (crypto-native, US macro, geopolitics) and decide which ones "
    "materially affect BTC price over the next 6-72 hours. For each material "
    "headline, assign a directional bias and a weight in [0, 1] reflecting "
    "estimated short-term price impact magnitude. Be honest: many headlines "
    "are noise — exclude them. Cover events on both sides (long and short) "
    "rather than picking a directional narrative.\n\n"
    "Trump-era foreign policy is volatile and material — tariffs, sanctions, "
    "executive orders, and Middle East / China / Russia developments often "
    "move risk assets including BTC. Score them carefully, considering whether "
    "the news is escalation (typically risk-off → BTC short) or de-escalation "
    "(risk-on → BTC long).\n\n"
    "Return STRICT JSON conforming to this schema (no prose):\n"
    "{\n"
    '  "events": [\n'
    '    {"headline": str, "bias": "long"|"short"|"neutral", '
    '"weight": float (0-1), "rationale": str (1 sentence)}\n'
    "  ],\n"
    '  "macro_calendar": [str (event with date), ...]\n'
    "}\n\n"
    "Aim for 5-15 events. Skip pure noise (celebrity gossip, generic market "
    "summaries, daily price recaps)."
)


def score_with_gemini(
    headlines: list[dict[str, Any]],
    api_key: str,
    model: str | None = None,
    window_hours: int = 24,
) -> dict[str, Any]:
    """Single-batch JSON-mode call (google-genai SDK).

    Always returns a dict. On failure the dict has a `_error` key
    describing the exact failure reason; on success it has `events`.
    """
    if not headlines:
        return {"window_hours": window_hours, "events": [], "macro_calendar": []}

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as e:
        return {"events": [], "macro_calendar": [],
                "_error": f"google-genai 패키지 import 실패: {e}"}

    # Numbered, source-tagged headline list keeps prompt compact.
    lines = []
    for i, h in enumerate(headlines, start=1):
        lines.append(
            f"{i}. [{h['source']}] {h['title']}"
            + (f" — {h['summary'][:120]}" if h.get("summary") else "")
        )
    user_prompt = (
        f"Recent headlines (last {window_hours}h, UTC):\n\n"
        + "\n".join(lines)
        + "\n\nReturn the JSON brief now."
    )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {"events": [], "macro_calendar": [],
                "_error": f"Gemini 클라이언트 생성 실패: {type(e).__name__}: {e}"}

    config = genai_types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0.2,
    )

    # Try the preferred model, fall back through the chain on model-specific
    # failures (404 NOT_FOUND, "model not supported", etc.).
    candidates: tuple[str, ...] = (
        (model,) if model else MODEL_FALLBACK_CHAIN
    )
    last_error: str | None = None
    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate, contents=user_prompt, config=config,
            )
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}"
            last_error = f"[{candidate}] {err_str}"
            # Only retry next model on availability errors. Auth / quota errors
            # are global and won't be helped by trying other models.
            lower = err_str.lower()
            if any(tok in lower for tok in ("not found", "not supported", "404", "unavailable")):
                logger.info("Model %s unavailable, trying next: %s", candidate, err_str)
                continue
            return {"events": [], "macro_calendar": [], "_error": last_error,
                    "_model_tried": candidate}

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            last_error = f"[{candidate}] empty response (안전 필터 가능성)"
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            last_error = f"[{candidate}] JSON 파싱 실패: {e}; raw[:200]={text[:200]!r}"
            continue
        # Success path: jump out of fallback loop with a model attribution.
        parsed["_model"] = candidate
        break
    else:
        # Final fallback: query ListModels so the user sees what their key can
        # actually call, instead of guessing at names.
        available_hint = ""
        try:
            available = []
            for m in client.models.list():
                name = getattr(m, "name", "") or ""
                # Only models that support generateContent are useful here.
                actions = getattr(m, "supported_actions", None) or \
                          getattr(m, "supported_generation_methods", None) or []
                if "generateContent" in actions or not actions:
                    available.append(name.replace("models/", ""))
                if len(available) >= 12:
                    break
            if available:
                available_hint = f" 이 키로 사용 가능한 모델 (상위 12): {', '.join(available)}"
        except Exception as list_e:
            available_hint = f" (ListModels도 실패: {list_e})"
        return {"events": [], "macro_calendar": [],
                "_error": f"모든 모델 후보 실패. last={last_error}.{available_hint}"}

    if not isinstance(parsed, dict) or "events" not in parsed:
        return {"events": [], "macro_calendar": [],
                "_error": f"Gemini 응답에 'events' 키 없음: {str(parsed)[:200]}"}

    # Sanitise + clamp.
    events = []
    for ev in parsed.get("events", []) or []:
        try:
            bias = str(ev.get("bias", "neutral")).lower()
            if bias not in ("long", "short", "neutral"):
                bias = "neutral"
            weight = float(ev.get("weight", 0.0) or 0.0)
            weight = max(0.0, min(1.0, weight))
            events.append({
                "headline": str(ev.get("headline", ""))[:300],
                "bias": bias,
                "weight": weight,
                "rationale": str(ev.get("rationale", ""))[:300],
            })
        except (TypeError, ValueError):
            continue

    return {
        "window_hours": window_hours,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": "live_gemini",
        "model": parsed.get("_model", "unknown"),
        "events": events,
        "macro_calendar": [str(c)[:120] for c in (parsed.get("macro_calendar") or [])][:10],
    }


# ─── 3. Top-level pipeline ───────────────────────────────────────────────────


def build_news_brief(
    window_hours: int = 24,
    api_key: str | None = None,
    max_headlines: int = 50,
) -> dict[str, Any] | None:
    """Returns a news_brief dict ready for signals/news.py, or None on failure.

    On failure paths, attaches a `_diagnostic` key to a stub dict so the
    caller can surface the exact reason (RSS empty, filter empty, Gemini
    error, SDK missing, etc.) instead of silently falling back.
    """
    if not api_key:
        return None  # No key → skip live; caller uses disk baseline.

    raw = fetch_headlines(window_hours=window_hours)
    if not raw:
        return {
            "events": [],
            "macro_calendar": [],
            "_diagnostic": (
                "RSS 페치에서 0건 반환 — Streamlit Cloud에서 외부 RSS가 막혔거나 "
                "모든 소스가 일시 장애. 1분 후 재시도."
            ),
        }

    relevant = _filter_relevant(raw, cap=max_headlines)
    if not relevant:
        return {
            "events": [],
            "macro_calendar": [],
            "_diagnostic": (
                f"RSS는 {len(raw)}건 받았지만 키워드 화이트리스트 통과 0건 — "
                "최근 24시간에 BTC/Trump/Fed 관련 헤드라인이 정말 없을 수도 있음."
            ),
        }

    scored = score_with_gemini(relevant, api_key=api_key, window_hours=window_hours)
    if "_error" in scored:
        return {
            "events": [],
            "macro_calendar": [],
            "_diagnostic": (
                f"Gemini 호출 실패 (입력 {len(relevant)}건): {scored['_error']}"
            ),
        }
    if not scored.get("events"):
        scored["_diagnostic"] = (
            f"Gemini 응답에 events 0개 — 입력 {len(relevant)}건이 "
            "모두 노이즈로 분류됐거나 모델 응답 파싱 이슈."
        )
    return scored
