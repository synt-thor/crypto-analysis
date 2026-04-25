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

DEFAULT_MODEL = "gemini-2.0-flash"


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
    model: str = DEFAULT_MODEL,
    window_hours: int = 24,
) -> dict[str, Any] | None:
    """Single-batch JSON-mode call (google-genai SDK). Returns brief dict or None."""
    if not headlines:
        return {"window_hours": window_hours, "events": [], "macro_calendar": []}

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("google-genai not installed; skipping live scoring")
        return None

    try:
        client = genai.Client(api_key=api_key)

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

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            logger.warning("Gemini returned empty response")
            return None
        parsed = json.loads(text)
    except Exception as e:
        logger.exception("Gemini scoring failed: %s", e)
        return None

    if not isinstance(parsed, dict) or "events" not in parsed:
        logger.warning("Gemini response missing 'events' key")
        return None

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

    None signals "use disk baseline" — caller should fall back gracefully.
    """
    if not api_key:
        return None  # No key → skip live; caller uses disk baseline.

    raw = fetch_headlines(window_hours=window_hours)
    if not raw:
        logger.info("No headlines returned from any RSS source")
        return None

    relevant = _filter_relevant(raw, cap=max_headlines)
    if not relevant:
        logger.info("No relevant headlines after keyword filter (raw=%d)", len(raw))
        return None

    return score_with_gemini(relevant, api_key=api_key, window_hours=window_hours)
