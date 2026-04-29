"""Streamlit web UI for the BTC futures decision engine.

Wraps existing crypto_analysis modules — no new business logic.
Run locally: `streamlit run streamlit_app.py`
"""

from __future__ import annotations

# Ensure local src/ is importable FIRST, ahead of any stale pip-installed copy.
# Streamlit Cloud reuses its venv across deploys, which can pin an older
# crypto_analysis package. Prepending src/ guarantees the freshest source wins.
import sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from crypto_analysis.backtest import run as run_backtest
from crypto_analysis.collectors import deribit, exchanges, onchain
from crypto_analysis.collectors.macro import macro_panel
from crypto_analysis.config import PERPETUAL
from crypto_analysis.decision import decide, decide_multi
from crypto_analysis.engine import (
    DEFAULT_WEIGHTS,
    WEIGHTS_LT,
    WEIGHTS_MT,
    WEIGHTS_ST,
    fuse,
)
from crypto_analysis.signals import SignalScore
from crypto_analysis.signals.basis import compute as basis_compute
from crypto_analysis.signals.funding import compute as funding_compute
from crypto_analysis.signals.gex import compute as gex_compute
from crypto_analysis.signals.iv_skew import compute as iv_skew_compute
from crypto_analysis.signals.liquidations import compute as liquidations_compute
from crypto_analysis.signals.macro import compute as macro_compute
from crypto_analysis.signals.news import compute as news_compute
from crypto_analysis.signals.oi import compute as oi_compute
from crypto_analysis.signals.onchain import compute as onchain_compute
from crypto_analysis.signals.option_skew import compute as option_skew_compute
from crypto_analysis.signals.orderbook import compute as orderbook_compute
from crypto_analysis.signals.spot_futures import compute as spot_futures_compute

# ─── Defensive monkey-patches ────────────────────────────────────────────────
# Streamlit Cloud reuses its venv across deploys. If a previous build cached an
# older crypto_analysis package, newly added module attributes (introduced after
# that build) will be missing at runtime even though the source is up-to-date.
# Sys.path prepend usually fixes this, but if the package is also imported via
# editable .pth, attribute resolution can still hit the cached site-packages.
# These shims attach the functions directly to the module objects when missing,
# guaranteeing the live app works regardless of pip-install state.
if not hasattr(deribit, "index_price"):
    from crypto_analysis.config import DERIBIT_REST as _DERIBIT_REST
    from crypto_analysis.http import get_json as _get_json

    def _index_price_shim(index_name: str = "btc_usd") -> float:
        payload = _get_json(
            f"{_DERIBIT_REST}/public/get_index_price",
            params={"index_name": index_name},
        )
        result = payload.get("result", payload) if isinstance(payload, dict) else {}
        return float(result.get("index_price") or 0.0)

    deribit.index_price = _index_price_shim  # type: ignore[attr-defined]

if not hasattr(exchanges, "coinbase_candles_range"):
    from crypto_analysis.config import COINBASE_REST as _COINBASE_REST
    from crypto_analysis.http import get_json as _get_json2

    def _coinbase_candles_range_shim(
        product_id: str = "BTC-USD",
        granularity: int = 3600,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        from datetime import datetime as _dt, timezone as _tz
        if start_ms is None or end_ms is None:
            rows = _get_json2(
                f"{_COINBASE_REST}/products/{product_id}/candles",
                {"granularity": granularity},
            )
            df = pd.DataFrame(rows or [], columns=["time", "low", "high", "open", "close", "volume"])
            if not df.empty:
                df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df["product_id"] = product_id
            return df.sort_values("ts").reset_index(drop=True) if not df.empty else df
        chunk_secs = 300 * granularity
        cursor = start_ms // 1000
        end_s = end_ms // 1000
        frames: list[pd.DataFrame] = []
        while cursor < end_s:
            chunk_end = min(cursor + chunk_secs, end_s)
            rows = _get_json2(
                f"{_COINBASE_REST}/products/{product_id}/candles",
                {
                    "granularity": granularity,
                    "start": _dt.fromtimestamp(cursor, tz=_tz.utc).isoformat(),
                    "end": _dt.fromtimestamp(chunk_end, tz=_tz.utc).isoformat(),
                },
            )
            if rows:
                frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
            cursor = chunk_end
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["product_id"] = product_id
        return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    exchanges.coinbase_candles_range = _coinbase_candles_range_shim  # type: ignore[attr-defined]

NEWS_BRIEF_PATH = Path(__file__).parent / "data" / "news_brief.json"

VERDICT_COLORS = {"LONG": "#22c55e", "SHORT": "#ef4444", "NEUTRAL": "#9ca3af"}

# Plain-language signal names for beginner-friendly summaries.
SIGNAL_NAME_KR: dict[str, str] = {
    "macro": "거시 환경(주식·달러)",
    "option_skew": "옵션 헷지 수요",
    "iv_skew": "변동성 안도/공포",
    "basis": "선물 베이시스",
    "orderbook": "오더북 매수벽",
    "news": "뉴스 흐름",
    "gex": "옵션 포지셔닝",
    "onchain": "비트코인 네트워크",
    "spot_futures": "현물-선물 괴리",
    "funding": "펀딩(롱/숏 균형)",
    "oi": "미결제약정 변화",
    "liquidations": "청산 흐름",
}

# Group signals by what they measure for the overview.
SIGNAL_CATEGORIES: dict[str, list[str]] = {
    "시장 분위기": ["macro", "news", "onchain"],
    "트레이더 베팅": ["funding", "oi", "gex", "option_skew", "iv_skew", "liquidations"],
    "가격 압력": ["orderbook", "spot_futures", "basis"],
}

AUDIENCE_FOR_TF: dict[str, str] = {
    "ST": "데이트레이더 · 수 시간",
    "MT": "스윙 트레이더 · 며칠",
    "LT": "장기 투자자 · 수 주~",
}


def _score_indicator(score: float, confidence: float = 1.0) -> tuple[str, str]:
    """Returns (compact_label, css_color) for a bias indicator without emoji."""
    if confidence < 0.15:
        return ("—", "#6b7280")  # inactive / dim
    if score >= 0.30:
        return ("▲▲", "#22c55e")
    if score >= 0.10:
        return ("▲", "#22c55e")
    if score >= -0.10:
        return ("·", "#9ca3af")
    if score >= -0.30:
        return ("▼", "#ef4444")
    return ("▼▼", "#ef4444")


def _score_phrase(score: float) -> str:
    if score >= 0.30:
        return "강한 매수"
    if score >= 0.15:
        return "매수 우호"
    if score >= 0.05:
        return "약 매수 쪽"
    if score >= -0.05:
        return "기다리기"
    if score >= -0.15:
        return "약 매도 쪽"
    if score >= -0.30:
        return "매도 우호"
    return "강한 매도"


def _tf_advice(verdict: str, score: float) -> str:
    """One-line plain advice for a timeframe verdict + score."""
    if verdict == "LONG":
        return "분할 매수 또는 보유" if score >= 0.20 else "관망 후 눌림 매수 후보"
    if verdict == "SHORT":
        return "헷지 또는 단기 숏 검토" if score <= -0.20 else "포지션 축소 검토"
    return "방향 베팅 자제"

# Detailed tooltips for each signal's weight slider — explains what the signal
# measures, its data source, and how its score maps to LONG/SHORT bias.
SIGNAL_HELP: dict[str, str] = {
    "funding": (
        "**퍼펫추얼 펀딩레이트** (역추세)\n\n"
        "Deribit BTC-PERPETUAL의 8시간 펀딩을 APR로 환산 + 7일 z-score.\n\n"
        "- 극단적 **+펀딩** (롱→숏 지급) = 롱 과열 → **숏 바이어스**\n"
        "- 극단적 **−펀딩** (숏→롱 지급) = 숏 과열 → **롱 바이어스**\n"
        "- 기준: 30% APR = crowded, 80% APR = extreme\n"
        "- 소스: `deribit.funding_rate_history`"
    ),
    "spot_futures": (
        "**현물 vs 퍼펫추얼 프리미엄** (역추세)\n\n"
        "Deribit BTC 퍼프 마크 − Deribit BTC 인덱스(현물) 괴리.\n\n"
        "- 퍼프 > 현물 = 레버리지 롱 수요 → **숏 바이어스**\n"
        "- 퍼프 < 현물 = 디레버리징·스트레스 → **롱 바이어스**\n"
        "- 20bp(0.2%) 괴리를 풀 스케일로 tanh 매핑\n"
        "- 수 시간 단위 마이크로 구조 신호"
    ),
    "option_skew": (
        "**25Δ 리스크리버설 (RR)** (트렌드)\n\n"
        "`RR = 25Δ 콜 IV − 25Δ 풋 IV`, 최근접 만기(DTE ≥ 7일).\n\n"
        "- **+RR** (콜 프리미엄) = 상승 포지셔닝 → **롱 바이어스**\n"
        "- **−RR** (풋 프리미엄) = 헷지·공포 → **숏 바이어스**\n"
        "- 5vol pp를 풀 스케일로, Black-Scholes 델타 직접 계산 (r=0)\n"
        "- 소스: Deribit 옵션 체인 book_summary"
    ),
    "basis": (
        "**만기 선물 연율화 베이시스** (역추세)\n\n"
        "Front 만기 선물 프리미엄을 연율로 환산 (DTE ≥ 7일 필터).\n\n"
        "- **깊은 콘탱고** (APR 25%+) = 투기성 롱 프리미엄 → **숏 바이어스**\n"
        "- 정상 콘탱고 (0~15%) = 중립·캐리 약한 롱\n"
        "- **백워데이션** = 현물 강세 또는 스트레스 → **롱 바이어스**\n"
        "- 소스: Deribit book_summary + 인덱스 가격"
    ),
    "oi": (
        "**미결제약정(OI) × 가격 변화** (트렌드)\n\n"
        "OI·가격 변화를 4국면으로 분류:\n\n"
        "- 가격↑ + OI↑ = 신규 롱 빌드업 → **롱**\n"
        "- 가격↓ + OI↑ = 신규 숏 빌드업 → **숏**\n"
        "- 가격↑ + OI↓ = 숏 커버링 (약한 랠리) → 미미한 롱\n"
        "- 가격↓ + OI↓ = 롱 청산 페이딩 → 미미한 숏\n"
        "- 라이브 단일 스냅샷에선 표본 부족으로 신뢰도 낮음"
    ),
    "macro": (
        "**크로스 자산 매크로 모멘텀** (트렌드)\n\n"
        "5일 vs 20일 이동평균 모멘텀 × BTC 과거 상관 부호.\n\n"
        "- **DXY ↑** = BTC 부정적 (달러 강세)\n"
        "- **SPY / QQQ ↑** = BTC 긍정적 (위험선호 동조)\n"
        "- **VIX ↑** = BTC 부정적 (위험회피)\n"
        "- **10Y 수익률 ↑** = BTC 부정적 (유동성 축소)\n"
        "- 소스: yfinance (DX-Y.NYB, SPY, QQQ, ^VIX, ^TNX, GC=F)"
    ),
    "iv_skew": (
        "**IV − RV 스프레드** (역추세)\n\n"
        "ATM 임플라이드 변동성 − Deribit 실현 변동성.\n\n"
        "- **IV ≫ RV** = 옵션 프리미엄 비쌈·공포 헷지 → **롱 바이어스**\n"
        "- **IV ≪ RV** = 안도·콤플레이선시 → **숏 바이어스**\n"
        "- 20pp 스프레드를 풀 스케일로 매핑\n"
        "- 소스: Deribit 최근접 ATM 옵션 ticker + historical_volatility"
    ),
    "gex": (
        "**딜러 감마 편향 프록시** (트렌드)\n\n"
        "0~60일 옵션의 감마 가중 OI 비율:\n"
        "`net = (콜 γ-OI − 풋 γ-OI) / (콜 γ-OI + 풋 γ-OI)`\n\n"
        "- **콜 편중** = 상승 포지셔닝 → **롱 바이어스**\n"
        "- **풋 편중** = 헷지 포지셔닝 → **숏 바이어스**\n"
        "- 딜러 실제 재고는 불가지, \"포지셔닝 편향\" 프록시\n"
        "- Black-Scholes 감마 직접 계산"
    ),
    "liquidations": (
        "**청산 플로우** (역추세)\n\n"
        "최근 Deribit 선물 1000건 거래 중 liquidation 플래그 집계.\n\n"
        "- **롱 청산 우세** (강제 매도) = 단기 바닥 시그널 → **롱 바이어스**\n"
        "- **숏 청산 우세** (강제 매수) = 숏 스퀴즈 고점 → **숏 바이어스**\n"
        "- Deribit만 집계 → CEX 전체 플로우보다 표본 작음\n"
        "- 청산 0건이면 비활성 (conf=0)"
    ),
    "orderbook": (
        "**오더북 뎁스 임밸런스** (트렌드)\n\n"
        "퍼펫추얼 상위 10호가의 매수·매도 볼륨 비율.\n\n"
        "- **bid 볼륨 우세** = 단기 매수 압력 → **롱 바이어스**\n"
        "- **ask 볼륨 우세** = 단기 매도 압력 → **숏 바이어스**\n"
        "- 스푸핑 취약 → 점수 상한 ±0.6\n"
        "- 수 분~수 시간 타임프레임"
    ),
    "onchain": (
        "**비트코인 네트워크 컨텍스트** (약한 컨텍스트)\n\n"
        "mempool.space의 멤풀·수수료·해시레이트·난이도.\n\n"
        "- 극단적 수수료·백로그 = 투기 과열 → 약한 **숏**\n"
        "- 해시레이트·난이도 상승 = 불장 모멘텀 → 약한 **롱**\n"
        "- 단기 가격 예측력은 약함, 거시 컨텍스트 목적\n"
        "- 소스: `mempool.space/api` 무료 엔드포인트"
    ),
    "news": (
        "**구조화 뉴스 브리프** (컨텍스트)\n\n"
        "`data/news_brief.json`의 이벤트 리스트를 집계.\n\n"
        "- 각 이벤트: `bias` (long/short/neutral) × `weight` (0~1)\n"
        "- 최대 점수 ±0.6, 신뢰도 최대 0.5로 제한\n"
        "  (뉴스→숫자 변환이 노이지하므로)\n"
        "- FOMC·CPI 등 매크로 캘린더도 함께 표시\n"
        "- 편집: 본문 expander에서 JSON 직접 수정"
    ),
}


# ─── Cached data fetchers ────────────────────────────────────────────────────


@st.cache_data(ttl=60, show_spinner="Deribit / Binance / mempool 호출 중…")
def fetch_market_inputs() -> dict:
    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    book = deribit.book_summary_by_currency("BTC", "future")
    perp_ticker = deribit.ticker(PERPETUAL)
    perp_ob = deribit.order_book(PERPETUAL, depth=20)
    options = deribit.option_book_summary("BTC")
    rv = deribit.historical_volatility("BTC")
    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))
    liq_trades = deribit.last_liquidations("BTC", count=1000)
    # Spot reference: Deribit BTC index (volume-weighted; no geo-block).
    # Binance public API blocks US datacenter IPs which fails on Streamlit Cloud.
    spot_price = deribit.index_price("btc_usd")
    mempool = onchain.mempool_snapshot()
    fees = onchain.fees_recommended()
    hr = onchain.hashrate_3d()

    perp_mark = float(perp_ticker.get("mark_price") or perp_ticker.get("last_price") or 0)

    # Pick a near-ATM option for IV reference.
    atm_iv_val = None
    if not options.empty and spot_price:
        soon = options[options["expiry"] > now].sort_values("expiry").head(60)
        if not soon.empty:
            soon = soon.assign(abs_delta=(soon["strike"] - spot_price).abs())
            near_atm = soon.sort_values(["expiry", "abs_delta"]).iloc[0]
            atm_iv_val = float(near_atm.get("mark_iv") or 0) or None

    return {
        "fetched_at": now,
        "book": book,
        "perp_ticker": perp_ticker,
        "perp_mark": perp_mark,
        "perp_ob": perp_ob,
        "options": options,
        "rv": rv,
        "funding": funding,
        "liq_trades": liq_trades,
        "spot_price": spot_price,
        "mempool": mempool,
        "fees": fees,
        "hr": hr,
        "atm_iv_val": atm_iv_val,
    }


@st.cache_data(ttl=300, show_spinner="yfinance에서 매크로 패널 받는 중…")
def fetch_macro_panel(days: int = 45) -> pd.DataFrame:
    return macro_panel(days=days)


def _get_gemini_api_key() -> str | None:
    """Read GEMINI_API_KEY from Streamlit secrets first, then environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


@st.cache_data(ttl=900, show_spinner="실시간 뉴스 페치 + Gemini 점수화…")
def fetch_live_news_brief() -> dict | None:
    """Returns auto-built news brief or None to fall back to disk baseline."""
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    try:
        from crypto_analysis.news_fetcher import build_news_brief
        return build_news_brief(window_hours=24, api_key=api_key)
    except Exception as e:
        st.warning(f"실시간 뉴스 페치 실패: {e}. 디스크 baseline으로 폴백합니다.")
        return None


_RES_TO_BARS_PER_HOUR = {"1": 60, "5": 12, "15": 4, "60": 1}
_RES_TO_COINBASE_GRAN = {"1": 60, "5": 300, "15": 900, "60": 3600}


@st.cache_data(ttl=3600, show_spinner="과거 데이터 백필 + 백테스트 실행 중…")
def run_backtest_live(
    days: int,
    hold_hours: float,
    resolution: str = "60",
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
):
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    bars_per_hour = _RES_TO_BARS_PER_HOUR.get(resolution, 1)

    # Use chunked fetch for fine resolutions (1m / 5m over many days).
    if resolution in ("1", "5") and days > 3:
        perp = deribit.tradingview_chart_data_chunked(
            PERPETUAL, start_ms, end_ms, resolution=resolution
        )
    else:
        perp = deribit.tradingview_chart_data(
            PERPETUAL, start_ms, end_ms, resolution=resolution
        )

    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))

    # Coinbase candles at matching granularity (fallback to 1h if granularity
    # too fine — Coinbase only supports 60/300/900/3600/21600/86400).
    granularity = _RES_TO_COINBASE_GRAN.get(resolution, 3600)
    spot = exchanges.coinbase_candles_range(
        "BTC-USD", granularity=granularity, start_ms=start_ms, end_ms=end_ms
    )

    return run_backtest(
        perp_ohlcv=perp.sort_values("ts").reset_index(drop=True),
        spot_ohlcv=spot.sort_values("ts").reset_index(drop=True),
        funding_df=funding.sort_values("ts").reset_index(drop=True),
        ticker_hist=None,
        hold_hours=hold_hours,
        step_hours=max(1, hold_hours / 2),
        bars_per_hour=bars_per_hour,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


# ─── Signal computation ──────────────────────────────────────────────────────


def build_signals(inputs: dict, macro_df: pd.DataFrame, news_brief: dict | None) -> list[SignalScore]:
    now = inputs["fetched_at"]
    perp_mark = inputs["perp_mark"]
    spot_price = inputs["spot_price"]
    perp_ticker = inputs["perp_ticker"]

    ticker_hist = pd.DataFrame([{
        "ts": now,
        "mark_price": perp_mark,
        "open_interest": perp_ticker.get("open_interest", 0),
    }])

    return [
        funding_compute(inputs["funding"]),
        basis_compute(inputs["book"], spot_price),
        oi_compute(ticker_hist),
        iv_skew_compute(inputs["rv"], inputs["atm_iv_val"]),
        spot_futures_compute(perp_mark, spot_price),
        orderbook_compute(inputs["perp_ob"], levels=10),
        option_skew_compute(inputs["options"], spot_price),
        gex_compute(inputs["options"], spot_price, dte_max=60),
        liquidations_compute(inputs["liq_trades"]),
        onchain_compute(inputs["mempool"], inputs["fees"], inputs["hr"]),
        macro_compute(macro_df),
        news_compute(news_brief),
    ]


# ─── Renderers ───────────────────────────────────────────────────────────────


def render_header(fetched_at: datetime) -> None:
    title_col, time_col, btn_col = st.columns([4, 3, 1])
    with title_col:
        st.title("BTC Futures Decision")
    with time_col:
        kst = fetched_at.astimezone(tz=timezone(timedelta(hours=9)))
        st.caption(
            f"Last fetched: **{fetched_at:%Y-%m-%d %H:%M:%S} UTC**  "
            f"(KST {kst:%H:%M:%S})"
        )
    with btn_col:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_verdict_block(decision, perp_mark: float, spot_price: float) -> None:
    color = VERDICT_COLORS[decision.verdict]
    premium = (perp_mark - spot_price) / spot_price * 100 if spot_price else 0.0
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}08);
            border: 2px solid {color};
            border-radius: 14px;
            padding: 24px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 14px; opacity: 0.7;">VERDICT</div>
            <div style="font-size: 56px; font-weight: 700; color: {color}; line-height: 1.1;">
                {decision.verdict}
            </div>
            <div style="font-size: 16px; opacity: 0.85; margin-top: 8px;">
                score: <b>{decision.score:+.3f}</b> &nbsp;·&nbsp; confidence: <b>{decision.confidence:.2f}</b>
            </div>
            <div style="font-size: 13px; opacity: 0.65; margin-top: 8px;">
                {decision.reason}
            </div>
            <div style="font-size: 13px; opacity: 0.7; margin-top: 12px;
                        border-top: 1px solid #ffffff22; padding-top: 10px;">
                BTC perp <b>${perp_mark:,.2f}</b> &nbsp;·&nbsp;
                spot <b>${spot_price:,.2f}</b> &nbsp;·&nbsp;
                premium <b>{premium:+.3f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeframes(multi) -> None:
    st.subheader("타임프레임별 판정")
    cols = st.columns(3)
    for col, tf_key, label, dec, res in zip(
        cols,
        ["ST", "MT", "LT"],
        ["ST  (수 시간)", "MT  (수 일)", "LT  (수 주)"],
        [multi.st, multi.mt, multi.lt],
        [multi.st_result, multi.mt_result, multi.lt_result],
    ):
        color = VERDICT_COLORS[dec.verdict]
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid {color}88; border-radius:10px;
                            padding:14px; background: {color}11;">
                    <div style="font-size:12px; opacity:0.7;">{label}</div>
                    <div style="font-size:28px; font-weight:700; color:{color};">
                        {dec.verdict}
                    </div>
                    <div style="font-size:13px; opacity:0.85;">
                        score <b>{dec.score:+.3f}</b> · conf <b>{dec.confidence:.2f}</b>
                    </div>
                    <div style="font-size:11px; opacity:0.7; margin-top:8px;
                                border-top:1px solid #ffffff22; padding-top:8px;
                                letter-spacing:0.3px;">
                        {AUDIENCE_FOR_TF[tf_key]}
                    </div>
                    <div style="font-size:13px; margin-top:6px; font-weight:600;
                                color:{color};">
                        {_tf_advice(dec.verdict, dec.score)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            top3 = res.contributions[:3]
            for c in top3:
                if c["contribution"] > 0:
                    arrow, arrow_color = "▲", "#22c55e"
                elif c["contribution"] < 0:
                    arrow, arrow_color = "▼", "#ef4444"
                else:
                    arrow, arrow_color = "·", "#9ca3af"
                kr_name = SIGNAL_NAME_KR.get(c["name"], c["name"])
                st.markdown(
                    f'<div style="font-size:12px; opacity:0.85; margin:4px 0;">'
                    f'<span style="color:{arrow_color}; font-weight:600;">{arrow}</span> '
                    f'<b>{kr_name}</b> '
                    f'<span style="opacity:0.7;">{c["contribution"]:+.3f}</span>'
                    f' · <span style="opacity:0.65;">{c["rationale"][:70]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _top_news_html(news_brief: dict | None, n: int = 3) -> str:
    """Returns HTML for the 'top N news' subsection, or '' if no news."""
    if not news_brief or not news_brief.get("events"):
        return ""
    events = sorted(
        news_brief["events"],
        key=lambda e: float(e.get("weight", 0) or 0),
        reverse=True,
    )[:n]
    if not events:
        return ""

    bias_marker = {
        "long":    ("▲", "#22c55e"),
        "short":   ("▼", "#ef4444"),
        "neutral": ("·", "#9ca3af"),
    }

    rows = []
    for ev in events:
        bias = str(ev.get("bias", "neutral")).lower()
        marker, color = bias_marker.get(bias, bias_marker["neutral"])
        weight = float(ev.get("weight", 0) or 0)
        headline = str(ev.get("headline", ""))[:80]
        rows.append(
            f'<div style="margin:5px 0; font-size:13px; line-height:1.5;">'
            f'<span style="color:{color}; font-weight:600; '
            f'display:inline-block; min-width:18px;">{marker}</span>'
            f'<span style="margin-left:4px;">{headline}</span>'
            f'<span style="opacity:0.45; font-size:11px; margin-left:8px;">'
            f'w={weight:.1f}</span>'
            f'</div>'
        )

    return (
        f'<div style="margin-top:12px; padding-top:10px;'
        f' border-top:1px solid #ffffff10;">'
        f'<div style="font-size:10px; letter-spacing:1.5px;'
        f' text-transform:uppercase; opacity:0.6; margin-bottom:6px;">'
        f'주요 뉴스 (상위 {len(events)}건)</div>'
        f'{"".join(rows)}'
        f'</div>'
    )


def render_plain_summary(decision, result, news_brief: dict | None = None) -> None:
    """1-line plain interpretation + optional top-news subsection."""
    contribs = result.contributions
    if not contribs:
        return
    pos = [c for c in contribs if c["contribution"] > 0]
    neg = [c for c in contribs if c["contribution"] < 0]
    top_pos = max(pos, key=lambda c: c["contribution"], default=None)
    top_neg = min(neg, key=lambda c: c["contribution"], default=None)

    parts = []
    if top_pos:
        parts.append(f"**{SIGNAL_NAME_KR.get(top_pos['name'], top_pos['name'])}**가 매수 쪽으로 가장 크게 기여")
    if top_neg:
        parts.append(f"**{SIGNAL_NAME_KR.get(top_neg['name'], top_neg['name'])}**가 매도 쪽으로 가장 크게 누름")

    verdict_phrase = {
        "LONG":    "→ **매수 쪽이 우세**",
        "SHORT":   "→ **매도 쪽이 우세**",
        "NEUTRAL": "→ **두 힘이 비슷해 명확한 방향 없음**",
    }[decision.verdict]

    summary = " · ".join(parts) + f" {verdict_phrase}"
    news_section = _top_news_html(news_brief, n=3)

    st.markdown(
        f"""<div style="background:#1e3a5f1f; border-left:2px solid #60a5fa;
        padding:12px 16px; margin:10px 0; border-radius:2px;">
        <div style="font-size:10px; letter-spacing:1.5px; text-transform:uppercase;
                    opacity:0.6; margin-bottom:4px;">한줄 해석</div>
        <div style="font-size:14px; line-height:1.6;">{summary}</div>
        {news_section}
        </div>""",
        unsafe_allow_html=True,
    )


def render_action_guide(multi) -> None:
    """3-row TF-aware action recommendation box."""
    rows = []
    for tf_key, label, dec in [
        ("LT", "장기 · 수 주~", multi.lt),
        ("MT", "중기 · 며칠",   multi.mt),
        ("ST", "단기 · 수 시간", multi.st),
    ]:
        color = VERDICT_COLORS[dec.verdict]
        advice = _tf_advice(dec.verdict, dec.score)
        rows.append(
            f"""<tr>
                <td style="padding:8px 10px; white-space:nowrap; opacity:0.75;
                           font-size:12px; letter-spacing:0.3px;">{label}</td>
                <td style="padding:8px 10px; color:{color}; font-weight:600;
                           letter-spacing:0.5px;">{dec.verdict}</td>
                <td style="padding:8px 10px;">{advice}</td>
            </tr>"""
        )

    st.markdown(
        f"""<div style="background:#0f172a55; border:1px solid #ffffff1a;
        border-radius:8px; padding:18px 20px; margin:12px 0;">
        <div style="font-size:14px; font-weight:600; margin-bottom:10px;
                    letter-spacing:0.3px;">
            오늘의 행동 가이드
        </div>
        <table style="width:100%; font-size:13px; border-collapse:collapse;">
            {''.join(rows)}
        </table>
        <div style="font-size:11px; opacity:0.6; margin-top:14px;
                    padding-top:10px; border-top:1px solid #ffffff10;
                    font-style:italic;">
            신호는 확률입니다. 강한 신호도 30% 이상 틀립니다. 손절선 미리 정하고,
            잃어도 되는 돈으로만 거래하세요. 한 신호에 풀 사이즈 진입 금지.
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_key_messages(result, news_brief: dict | None) -> int:
    """Dynamic regime/event insights — only show ones that apply right now.

    Returns the number of messages displayed (so the caller can append a
    conclusion box).
    """
    contribs = {c["name"]: c for c in result.contributions}
    msgs: list[str] = []

    # 1) Macro vs option skew divergence (very common in current regime).
    macro = contribs.get("macro")
    skew = contribs.get("option_skew")
    if macro and skew and macro["score"] > 0.3 and skew["score"] < -0.3:
        msgs.append(
            "**거시 vs 옵션 충돌** — 매크로(주식·달러)는 매수 우호인데 옵션 시장은 "
            "하락 보호에 비싸게 베팅 중. 거시 호재가 가격에 이미 반영됐을 가능성, "
            "또는 트레이더의 헷지가 과해서 풀리면 추가 상승 연료."
        )

    # 2) Complacency warning (IV << RV).
    iv = contribs.get("iv_skew")
    if iv and iv["score"] < -0.5:
        msgs.append(
            "**콤플레이선시 경고** — 옵션 시장이 미래 변동성을 실제 변동성보다 "
            "훨씬 낮게 가격에 반영. 시장이 과하게 안도하고 있어 변동성 폭발 위험. "
            "역사적으로 이런 구간 직후 큰 가격 급변이 자주 발생."
        )

    # 3) Funding extreme (rare but worth flagging).
    funding = contribs.get("funding")
    if funding and abs(funding["score"]) > 0.4:
        side = "롱 과열" if funding["score"] < 0 else "숏 과열"
        msgs.append(
            f"**펀딩 극단치** — {side} 상태. 역추세 압력 누적 중 · 청산 캐스케이드 가능."
        )

    # 4) Macro calendar event imminent (from news_brief).
    if news_brief and news_brief.get("macro_calendar"):
        cal = news_brief["macro_calendar"][:3]
        if cal:
            msgs.append(
                f"**임박 이벤트** — {' · '.join(cal)}. 결과에 따라 점수가 급변할 수 있음."
            )

    # 5) Liquidation flow extreme.
    liq = contribs.get("liquidations")
    if liq and abs(liq["score"]) > 0.3:
        side = "롱 청산 우세 (단기 바닥 가능)" if liq["score"] > 0 else "숏 청산 우세 (스퀴즈 고점 가능)"
        msgs.append(f"**청산 플로우** — {side}.")

    if not msgs:
        return 0  # nothing notable — keep page clean

    items_html = "".join(
        f'<li style="margin:8px 0; padding-left:4px;">{text}</li>' for text in msgs
    )
    st.markdown(
        f"""<div style="background:#1e293b3d; border:1px solid #ffffff1a;
        border-radius:8px; padding:18px 20px; margin:12px 0;">
        <div style="font-size:14px; font-weight:600; margin-bottom:10px;
                    letter-spacing:0.3px;">
            시장의 핵심 메시지
        </div>
        <ul style="margin:0; padding-left:18px; font-size:13px; line-height:1.7;">
            {items_html}
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )
    return len(msgs)


_THRESHOLD_EXPLAINER = """
**점수 → 표현 매핑**

모든 신호 점수와 최종 종합 점수는 **−1.0 ~ +1.0** 범위.

| 표현 | 점수 범위 | 의미 |
|---|---|---|
| **강한 매수** | `≥ +0.30` | 매우 강한 상승 압력 |
| **매수 우호** | `+0.15 ~ +0.30` | **최종 LONG 판정 임계** |
| 약 매수 쪽 | `+0.05 ~ +0.15` | 약한 매수 편향 |
| 기다리기 | `−0.05 ~ +0.05` | 사실상 균형 |
| 약 매도 쪽 | `−0.15 ~ −0.05` | 약한 매도 편향 |
| **매도 우호** | `−0.30 ~ −0.15` | **최종 SHORT 판정 임계** |
| **강한 매도** | `≤ −0.30` | 매우 강한 하락 압력 |

---

**최종 판정 규칙** (`decision.py`)

- **LONG**: 종합 점수 `≥ +0.15` **AND** 신뢰도 `≥ 0.25`
- **SHORT**: 종합 점수 `≤ −0.15` **AND** 신뢰도 `≥ 0.25`
- **NEUTRAL**: 위 둘 다 아닐 때 (또는 신뢰도 `< 0.25`이면 강제 NEUTRAL)

---

**왜 이 숫자들인가**

- **±0.15 (verdict 임계)**: 신호들의 가중 평균이 작아도 한 방향으로 15% 이상
  쏠려야 행동에 가치가 있음. 더 작으면 노이즈, 더 크면 너무 보수적이라 기회 놓침.
- **±0.30 (강한 신호)**: 임계의 2배. 단일 약한 신호로는 도달 불가 — 여러 강한
  신호가 한 방향으로 정렬돼야 나오는 수치.
- **신뢰도 0.25 (최소 컷)**: 최소 하나의 고신뢰 신호 또는 여러 중간 신뢰 신호의 합.
  이 미만이면 데이터가 부족해 점수 자체가 의미 없으므로 강제 NEUTRAL.

---

**시각 indicator 임계** (신호 분포 박스의 `▲▲ ▲ · ▼ ▼▼` 마커)

| 마커 | 점수 / 신뢰도 |
|---|---|
| `—` | 신뢰도 `< 0.15` (비활성) |
| `▲▲` | 점수 `≥ +0.30` |
| `▲` | 점수 `+0.10 ~ +0.30` |
| `·` | 점수 `−0.10 ~ +0.10` |
| `▼` | 점수 `−0.30 ~ −0.10` |
| `▼▼` | 점수 `≤ −0.30` |

표현 매핑(7단계)과 시각 indicator(5단계)가 다른 이유 — 텍스트 라벨은 더 세밀한
구분이 필요하고, 시각 마커는 한눈에 잡히게 단순화.

---

**수정하려면**

- `decision.py` — `LONG_THRESHOLD` / `SHORT_THRESHOLD` / `MIN_CONFIDENCE`
- `streamlit_app.py` — `_score_phrase()` / `_score_indicator()`
"""


def render_threshold_guide() -> None:
    """Popover explaining score buckets, verdict thresholds, and confidence gate."""
    cols = st.columns([1, 4])
    with cols[0]:
        with st.popover("점수와 임계 기준", use_container_width=True):
            st.markdown(_THRESHOLD_EXPLAINER)


def _render_section_conclusion(text: str) -> None:
    """Subtle green-bordered box with a clean 'Summary' label."""
    st.markdown(
        f"""<div style="background:#22c55e0d; border-left:2px solid #22c55e;
        padding:10px 16px; margin:6px 0 20px 0; border-radius:2px; font-size:13px;
        line-height:1.6;">
        <span style="font-size:10px; letter-spacing:1.5px; text-transform:uppercase;
                     opacity:0.65; margin-right:10px;">정리</span>
        {text}
        </div>""",
        unsafe_allow_html=True,
    )


def _compute_scalp_plan(
    multi,
    current_price: float,
    sl_pct: float = 0.003,
    tp_pct: float = 0.006,
) -> dict:
    """Compute concrete scalping trade plan from ST decision.

    Returns dict with direction / tier / entry / SL / TP / sizing / reasoning.
    """
    score = multi.st.score
    conf = multi.st.confidence
    abs_score = abs(score)

    # Tier classification (matches plan file table).
    if abs_score < 0.05 or conf < 0.25:
        tier = "hold"
        direction = "HOLD"
    elif abs_score < 0.15:
        tier = "weak"
        direction = "LONG" if score > 0 else "SHORT"
    elif abs_score < 0.30:
        tier = "medium"
        direction = "LONG" if score > 0 else "SHORT"
    else:
        tier = "strong"
        direction = "LONG" if score > 0 else "SHORT"

    # Sizing & leverage by tier.
    sizing = {
        "strong": (0.05, "2~3x", "강한"),
        "medium": (0.03, "1~2x", "우호"),
        "weak":   (0.02, "1x",   "약한"),
        "hold":   (0.0,  "—",    "보류"),
    }
    position_pct, leverage, tier_label = sizing[tier]

    # Approximate worst-case loss = pos_pct × sl_pct × leverage.
    # For "2~3x" we use midpoint 2.5; for "1~2x" we use 1.5.
    lev_midpoint = {"strong": 2.5, "medium": 1.5, "weak": 1.0, "hold": 0.0}[tier]
    max_loss_pct = position_pct * sl_pct * lev_midpoint

    # Entry / SL / TP prices.
    if direction == "LONG":
        entry_price = current_price
        sl_price = current_price * (1 - sl_pct)
        tp_price = current_price * (1 + tp_pct)
    elif direction == "SHORT":
        entry_price = current_price
        sl_price = current_price * (1 + sl_pct)
        tp_price = current_price * (1 - tp_pct)
    else:
        entry_price = sl_price = tp_price = current_price

    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0.0

    # Top supporting (pros) and opposing (cons) signals from ST result.
    contribs = multi.st_result.contributions
    pros = sorted(
        (c for c in contribs if c["contribution"] > 0.005),
        key=lambda c: c["contribution"], reverse=True,
    )[:3]
    cons = sorted(
        (c for c in contribs if c["contribution"] < -0.005),
        key=lambda c: c["contribution"],
    )[:3]

    # Reasoning sentence.
    if direction == "LONG":
        if tier == "strong":
            reasoning = "단기 신호가 강하게 매수 쪽으로 정렬 — 작은 사이즈로 즉시 진입 가능."
        elif tier == "medium":
            reasoning = "단기 매수 편향이지만 신호 강도 보통 — 작은 사이즈, 좁은 손절 권장."
        else:
            reasoning = "단기 약 매수 — 진입 선택사항, 큰 베팅 자제."
    elif direction == "SHORT":
        if tier == "strong":
            reasoning = "단기 신호가 강하게 매도 쪽으로 정렬 — 숏 또는 매도 진입 가능."
        elif tier == "medium":
            reasoning = "단기 매도 편향이지만 신호 강도 보통 — 작은 사이즈, 좁은 손절 권장."
        else:
            reasoning = "단기 약 매도 — 진입 선택사항, 큰 베팅 자제."
    else:
        if conf < 0.25:
            reasoning = "신뢰도 너무 낮음 — 데이터가 부족하니 진입 X."
        else:
            reasoning = "강한 매수·매도 신호 모두 부족 — 균형 상태, 명확한 신호 대기."

    return {
        "direction": direction,
        "tier": tier,
        "tier_label": tier_label,
        "score": score,
        "confidence": conf,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "rr_ratio": rr_ratio,
        "position_pct": position_pct,
        "leverage": leverage,
        "max_loss_pct": max_loss_pct,
        "pros": pros,
        "cons": cons,
        "reasoning": reasoning,
    }


def render_scalp_decision_panel(multi, current_price: float) -> None:
    """Always-visible scalping decision card (LONG/SHORT/HOLD with full plan)."""
    plan = _compute_scalp_plan(multi, current_price)
    direction = plan["direction"]
    tier = plan["tier"]
    color = VERDICT_COLORS["LONG"] if direction == "LONG" else \
            VERDICT_COLORS["SHORT"] if direction == "SHORT" else \
            VERDICT_COLORS["NEUTRAL"]

    # Direction label
    if direction == "HOLD":
        direction_label = "진입 보류"
        marker = "—"
    elif direction == "LONG":
        direction_label = f"{plan['tier_label']} 매수 (LONG)"
        marker = "▲▲" if tier == "strong" else "▲"
    else:
        direction_label = f"{plan['tier_label']} 매도 (SHORT)"
        marker = "▼▼" if tier == "strong" else "▼"

    # Build HTML pieces conditionally on direction
    if direction == "HOLD":
        body_html = f"""
            <div style="font-size:14px; line-height:1.7; opacity:0.85;">
                {plan['reasoning']}
            </div>
            <div style="margin-top:12px; padding-top:10px;
                        border-top:1px solid #ffffff10; font-size:12px;
                        opacity:0.6;">
                현재 ST score <b>{plan['score']:+.3f}</b> ·
                conf <b>{plan['confidence']:.2f}</b>
            </div>
        """
    else:
        # Pros HTML
        pros_html = ""
        for p in plan["pros"]:
            kr = SIGNAL_NAME_KR.get(p["name"], p["name"])
            pros_html += (
                f'<div style="margin:3px 0; font-size:12px;">'
                f'<span style="color:#22c55e;">▲</span> '
                f'<b>{kr}</b> '
                f'<span style="opacity:0.65;">{p["contribution"]:+.3f}</span>'
                f'</div>'
            )
        if not pros_html:
            pros_html = '<div style="opacity:0.5; font-size:12px;">없음</div>'

        # Cons HTML
        cons_html = ""
        for c in plan["cons"]:
            kr = SIGNAL_NAME_KR.get(c["name"], c["name"])
            cons_html += (
                f'<div style="margin:3px 0; font-size:12px;">'
                f'<span style="color:#ef4444;">▼</span> '
                f'<b>{kr}</b> '
                f'<span style="opacity:0.65;">{c["contribution"]:+.3f}</span>'
                f'</div>'
            )
        if not cons_html:
            cons_html = '<div style="opacity:0.5; font-size:12px;">없음</div>'

        body_html = f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;
                        margin-top:6px;">
                <div>
                    <div style="font-size:10px; letter-spacing:1.5px;
                                text-transform:uppercase; opacity:0.6;
                                margin-bottom:6px;">가격 계획</div>
                    <table style="width:100%; font-size:13px; border-collapse:collapse;">
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">진입가</td>
                            <td style="padding:4px 0; text-align:right;">
                                <b>${plan['entry_price']:,.2f}</b>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">손절가</td>
                            <td style="padding:4px 0; text-align:right; color:#ef4444;">
                                ${plan['sl_price']:,.2f}
                                <span style="opacity:0.55; font-size:11px;">
                                    ({plan['sl_pct']*100:+.2f}%)
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">익절가</td>
                            <td style="padding:4px 0; text-align:right; color:#22c55e;">
                                ${plan['tp_price']:,.2f}
                                <span style="opacity:0.55; font-size:11px;">
                                    ({plan['tp_pct']*100:+.2f}%)
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">R:R</td>
                            <td style="padding:4px 0; text-align:right;">
                                1:{plan['rr_ratio']:.1f}
                            </td>
                        </tr>
                    </table>
                </div>
                <div>
                    <div style="font-size:10px; letter-spacing:1.5px;
                                text-transform:uppercase; opacity:0.6;
                                margin-bottom:6px;">포지션 사이징</div>
                    <table style="width:100%; font-size:13px; border-collapse:collapse;">
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">자본 노출</td>
                            <td style="padding:4px 0; text-align:right;">
                                <b>{plan['position_pct']*100:.1f}%</b>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">레버리지</td>
                            <td style="padding:4px 0; text-align:right;">
                                <b>{plan['leverage']}</b>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; opacity:0.7;">최악 손실</td>
                            <td style="padding:4px 0; text-align:right; color:#ef4444;">
                                자본의 <b>{plan['max_loss_pct']*100:.3f}%</b>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
            <div style="margin-top:14px; padding-top:10px;
                        border-top:1px solid #ffffff10;">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;">
                    <div>
                        <div style="font-size:10px; letter-spacing:1.5px;
                                    text-transform:uppercase; opacity:0.6;
                                    margin-bottom:4px;">매수 근거</div>
                        {pros_html}
                    </div>
                    <div>
                        <div style="font-size:10px; letter-spacing:1.5px;
                                    text-transform:uppercase; opacity:0.6;
                                    margin-bottom:4px;">매도 근거</div>
                        {cons_html}
                    </div>
                </div>
            </div>
            <div style="margin-top:12px; font-size:12px; opacity:0.75;
                        font-style:italic;">
                {plan['reasoning']}
            </div>
        """

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, {color}1a, #0f172a33);
        border:2px solid {color}88; border-radius:10px;
        padding:18px 22px; margin:10px 0;">
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start; margin-bottom:10px;">
                <div>
                    <div style="font-size:11px; letter-spacing:2px;
                                text-transform:uppercase; opacity:0.65;">
                        단타 의사결정
                    </div>
                    <div style="font-size:24px; font-weight:700; color:{color};
                                margin-top:4px;">
                        <span style="margin-right:8px;">{marker}</span>
                        {direction_label}
                    </div>
                </div>
                <div style="font-size:11px; opacity:0.55; text-align:right;
                            line-height:1.5;">
                    ST score <b>{plan['score']:+.3f}</b><br>
                    conf <b>{plan['confidence']:.2f}</b><br>
                    tier <b>{tier}</b>
                </div>
            </div>
            {body_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_scalp_alert(multi, threshold: float = 0.30) -> None:
    """Big banner at top when ST score crosses strong-signal threshold."""
    st_score = multi.st.score
    st_verdict = multi.st.verdict
    st_conf = multi.st.confidence

    if abs(st_score) < threshold or st_conf < 0.25:
        return  # not strong enough

    if st_score >= threshold:
        side = "강한 매수 임계 돌파"
        action = "지금 매수 진입 검토 — 짧은 익절(0.5~1%) + 좁은 손절(0.3%)"
        color = "#22c55e"
    else:
        side = "강한 매도 임계 돌파"
        action = "지금 매도/숏 진입 검토 — 짧은 익절(0.5~1%) + 좁은 손절(0.3%)"
        color = "#ef4444"

    st.markdown(
        f"""<div style="background: {color}26; border:2px solid {color};
        border-radius:8px; padding:14px 18px; margin:8px 0 16px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; letter-spacing:2px;
                                text-transform:uppercase; opacity:0.75;">
                        단타 알림 · ST 임계
                    </div>
                    <div style="font-size:18px; font-weight:700; color:{color};
                                margin-top:4px;">
                        {side} ({st_verdict} {st_score:+.3f})
                    </div>
                    <div style="font-size:13px; opacity:0.85; margin-top:6px;">
                        {action}
                    </div>
                </div>
                <div style="font-size:11px; opacity:0.55; text-align:right;">
                    threshold ±{threshold:.2f}<br>
                    conf {st_conf:.2f}
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_easy_summary(decision, multi, news_brief: dict | None) -> None:
    """Top-of-page big-box plain-Korean summary tying everything together."""
    # Headline (overall verdict)
    if decision.verdict == "LONG":
        if decision.score >= 0.20:
            head = "지금 BTC 선물은 **매수 쪽이 우세**한 상태입니다."
        else:
            head = "지금 BTC 선물은 **약하게 매수 쪽**으로 기울어 있습니다."
    elif decision.verdict == "SHORT":
        if decision.score <= -0.20:
            head = "지금 BTC 선물은 **매도 쪽이 우세**한 상태입니다."
        else:
            head = "지금 BTC 선물은 **약하게 매도 쪽**으로 기울어 있습니다."
    else:
        head = "지금 BTC 선물은 **명확한 방향이 없는 상태**입니다."

    # ST line
    if multi.st.verdict == "LONG":
        st_line = "**단기(수 시간)**: 매수 우호 — 짧은 매수 시도 가능."
    elif multi.st.verdict == "SHORT":
        st_line = "**단기(수 시간)**: 매도 우호 — 짧은 숏 또는 헷지 검토."
    else:
        st_line = "**단기(수 시간)**: 방향 없음 — 쉬는 게 답."

    # MT line
    if multi.mt.verdict == "LONG":
        mt_line = "**중기(며칠)**: 매수 우호 — 분할 매수 환경."
    elif multi.mt.verdict == "SHORT":
        mt_line = "**중기(며칠)**: 매도 우호 — 포지션 축소 검토."
    else:
        mt_line = "**중기(며칠)**: 방향 없음 — 양방향 변동성 매수가 합리적."

    # LT line
    if multi.lt.verdict == "LONG":
        lt_line = "**장기(수 주)**: 매수 우호 — 분할 매수 또는 보유."
    elif multi.lt.verdict == "SHORT":
        lt_line = "**장기(수 주)**: 매도 우호 — 보유 줄이기 검토."
    else:
        lt_line = "**장기(수 주)**: 방향 없음 — 추가 신호 대기."

    # Optional calendar nudge
    cal_line = ""
    if news_brief and news_brief.get("macro_calendar"):
        first = news_brief["macro_calendar"][0]
        cal_line = (
            f'<div style="margin-top:10px; padding-top:8px; '
            f'border-top:1px solid #ffffff10; font-size:13px; opacity:0.85;">'
            f'예정 이벤트 — <b>{first}</b> 결과 후 다시 점검하세요.</div>'
        )

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #1e3a5f1f, #0f172a33);
        border:1px solid #60a5fa55; border-radius:8px;
        padding:22px 26px; margin:10px 0 18px 0;">
            <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
                        opacity:0.65; margin-bottom:12px;">
                오늘의 결론
            </div>
            <div style="font-size:15px; line-height:1.85;">
                {head}<br>
                <span style="opacity:0.85;">· {st_line}</span><br>
                <span style="opacity:0.85;">· {mt_line}</span><br>
                <span style="opacity:0.85;">· {lt_line}</span>
            </div>
            {cal_line}
            <div style="font-size:11px; opacity:0.55; margin-top:14px;
                        padding-top:10px; border-top:1px solid #ffffff10;
                        font-style:italic;">
                신호는 확률입니다. <b>잃어도 되는 돈으로만</b> 거래하시고,
                손절선을 미리 정하세요. 한 신호에 풀 사이즈 진입 금지.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _conclude_key_messages_text(msg_count: int) -> str:
    if msg_count == 0:
        return "지금 시장에는 **특별히 경계할 큰 시그널이 없어요**. 평소 그대로 진행해도 됩니다."
    if msg_count >= 2:
        return (
            "여러 경고가 동시에 떠 있어요. **큰 베팅 자제하고** 포지션 사이즈를 줄이거나 "
            "관망하는 게 안전합니다."
        )
    return "주목할 시그널이 있어요. **풀 사이즈 진입은 피하고** 신중하게 행동하세요."


def _conclude_signal_categories_text(result) -> str:
    by_name = {c["name"]: c for c in result.contributions}
    pos = sum(1 for c in by_name.values() if c["score"] > 0.1 and c["confidence"] >= 0.15)
    neg = sum(1 for c in by_name.values() if c["score"] < -0.1 and c["confidence"] >= 0.15)
    flat = len(by_name) - pos - neg
    if pos > neg + 2:
        tag = "**매수 신호가 우세**"
    elif neg > pos + 2:
        tag = "**매도 신호가 우세**"
    else:
        tag = "**매수·매도 신호가 비슷한 균형**"
    return (
        f"매수 신호 {pos}개 · 매도 신호 {neg}개 · 균형/비활성 {flat}개 → {tag}. "
        f"임계({'+0.15' if pos > neg else '-0.15'}) 통과 여부는 가중치 합산 결과에 달려 있어요."
    )


def _conclude_signal_table_text(result) -> str:
    contribs = result.contributions
    if not contribs:
        return "신호 데이터가 비어 있어요."
    pos = [c for c in contribs if c["contribution"] > 0]
    neg = [c for c in contribs if c["contribution"] < 0]
    top_pos = max(pos, key=lambda c: c["contribution"], default=None)
    top_neg = min(neg, key=lambda c: c["contribution"], default=None)
    parts = []
    if top_pos:
        parts.append(
            f"오늘 가장 크게 **매수 쪽으로 민** 신호: "
            f"**{SIGNAL_NAME_KR.get(top_pos['name'], top_pos['name'])}**"
        )
    if top_neg:
        parts.append(
            f"가장 크게 **매도 쪽으로 누른** 신호: "
            f"**{SIGNAL_NAME_KR.get(top_neg['name'], top_neg['name'])}**"
        )
    return ". ".join(parts) + "."


def render_signal_categories(result) -> None:
    """3-category overview of all 12 signals — typography only, no emoji."""
    by_name = {c["name"]: c for c in result.contributions}

    rows_html = []
    for cat_label, sig_names in SIGNAL_CATEGORIES.items():
        items = []
        for name in sig_names:
            c = by_name.get(name)
            if c is None:
                continue
            indicator, color = _score_indicator(c["score"], c["confidence"])
            kr = SIGNAL_NAME_KR.get(name, name)
            phrase = _score_phrase(c["score"])
            items.append(
                f'<span style="display:inline-block; margin:2px 0;">'
                f'<span style="color:{color}; font-weight:600; '
                f'display:inline-block; min-width:24px;">{indicator}</span>'
                f'<span style="margin-left:6px;">{kr}</span> '
                f'<span style="color:{color}; opacity:0.85;">— {phrase}</span>'
                f'</span>'
            )
        body = "<br>".join(items) if items else (
            '<span style="opacity:0.5;">데이터 없음</span>'
        )
        rows_html.append(
            f'<div style="margin:14px 0;">'
            f'<div style="font-size:11px; letter-spacing:1.5px; '
            f'text-transform:uppercase; opacity:0.6; margin-bottom:6px;">{cat_label}</div>'
            f'<div style="font-size:13px; padding-left:2px;">{body}</div>'
            f'</div>'
        )

    st.markdown(
        f"""<div style="background:#0f172a55; border:1px solid #ffffff1a;
        border-radius:8px; padding:18px 20px; margin:12px 0;">
        <div style="font-size:14px; font-weight:600; margin-bottom:6px;
                    letter-spacing:0.3px;">
            신호 분포
        </div>
        {''.join(rows_html)}
        <div style="font-size:11px; opacity:0.55; margin-top:14px;
                    padding-top:10px; border-top:1px solid #ffffff10;">
            <span style="color:#22c55e;">▲▲</span> 강한 매수 ·
            <span style="color:#22c55e;">▲</span> 매수 우호 ·
            <span style="color:#9ca3af;">·</span> 균형 ·
            <span style="color:#ef4444;">▼</span> 매도 우호 ·
            <span style="color:#ef4444;">▼▼</span> 강한 매도 ·
            <span style="color:#6b7280;">—</span> 비활성
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_signal_table(result) -> None:
    st.subheader("12개 신호 기여 (현재 가중치 기준)")
    rows = []
    for c in result.contributions:
        rows.append({
            "신호": c["name"],
            "Score": c["score"],
            "Conf": c["confidence"],
            "기여": c["contribution"],
            "유효 가중": c["effective_weight"],
            "근거": c["rationale"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("신호가 없습니다.")
        return
    st.dataframe(
        df,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=-1.0, max_value=1.0, format="%+.2f"
            ),
            "Conf": st.column_config.ProgressColumn(
                "Conf", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "기여": st.column_config.NumberColumn("기여", format="%+.3f"),
            "유효 가중": st.column_config.NumberColumn("유효 가중", format="%.2f"),
            "근거": st.column_config.TextColumn("근거", width="large"),
        },
        hide_index=True,
        use_container_width=True,
    )


def render_news_editor() -> dict | None:
    """Returns the news_brief to use (live > manual override > disk baseline)."""
    st.subheader("뉴스 브리프")

    # 1) Try live fetch (Gemini auto-scoring of RSS).
    live_brief = fetch_live_news_brief()
    api_key_present = _get_gemini_api_key() is not None

    # 2) Initialise session textarea from disk baseline once.
    if "news_text" not in st.session_state:
        try:
            st.session_state.news_text = (
                NEWS_BRIEF_PATH.read_text() if NEWS_BRIEF_PATH.exists() else "{}"
            )
        except Exception:
            st.session_state.news_text = "{}"

    # 3) Default mode: live if available, else manual/disk.
    if "news_mode" not in st.session_state:
        st.session_state.news_mode = "live" if live_brief else "manual"

    # 4) Status badge + diagnostic on partial failures.
    diagnostic_msg: str | None = None
    if live_brief and live_brief.get("events"):
        n_events = len(live_brief["events"])
        gen_at = live_brief.get("generated_at_utc", "")
        try:
            gen_dt = datetime.fromisoformat(gen_at)
            mins_ago = max(0, int((datetime.now(tz=timezone.utc) - gen_dt).total_seconds() / 60))
            ts_label = f"{mins_ago}분 전"
        except Exception:
            ts_label = "방금"
        badge_color = "#ef4444"
        badge_text = f"LIVE  ·  {n_events} events  ·  {ts_label} 갱신"
    elif live_brief is not None and "_diagnostic" in live_brief:
        # Pipeline ran but no usable events — show why.
        badge_color = "#f59e0b"
        badge_text = "LIVE 페치 부분 실패 — Baseline 사용"
        diagnostic_msg = live_brief["_diagnostic"]
        live_brief = None  # treat as failure for downstream
    elif api_key_present:
        badge_color = "#f59e0b"
        badge_text = "LIVE 페치 실패 — Baseline 사용"
    else:
        badge_color = "#9ca3af"
        badge_text = "Baseline (Gemini API 키 미설정)"

    st.markdown(
        f"""<div style="display:inline-block; border:1px solid {badge_color}88;
        border-radius:6px; padding:4px 10px; background:{badge_color}11;
        font-size:12px; margin-bottom:8px;">{badge_text}</div>""",
        unsafe_allow_html=True,
    )
    if diagnostic_msg:
        st.caption(f"진단: {diagnostic_msg}")

    # 5) Manual override toggle.
    if live_brief:
        cols = st.columns([1, 4])
        with cols[0]:
            override = st.toggle(
                "수동 편집 사용",
                value=(st.session_state.news_mode == "manual"),
                help="끄면 LIVE Gemini 브리프 사용, 켜면 아래 텍스트영역의 JSON 사용.",
            )
            st.session_state.news_mode = "manual" if override else "live"

    # 6) Expander with JSON editor (always available as override).
    with st.expander("뉴스 브리프 보기 / 편집", expanded=False):
        # Show live brief preview in read-only block when in live mode.
        if live_brief and st.session_state.news_mode == "live":
            st.caption("현재 LIVE 브리프 (읽기 전용 미리보기)")
            st.json(live_brief, expanded=False)
            st.caption("─ 아래는 디스크 baseline / 수동 편집용 ─")

        col_load, col_apply, col_disk_warn = st.columns([1, 1, 4])
        with col_load:
            if st.button("디스크에서 로드"):
                if NEWS_BRIEF_PATH.exists():
                    st.session_state.news_text = NEWS_BRIEF_PATH.read_text()
                    st.rerun()
                else:
                    st.warning(f"{NEWS_BRIEF_PATH} 가 없습니다.")
        with col_apply:
            apply_clicked = st.button("적용 (수동 모드)")

        text = st.text_area(
            "JSON (수동 편집 모드에서만 사용됨)",
            value=st.session_state.news_text,
            height=260,
            key="news_editor",
        )
        if apply_clicked:
            st.session_state.news_text = text
            st.session_state.news_mode = "manual"
            st.rerun()

    # 7) Decide which brief to return.
    if st.session_state.news_mode == "live" and live_brief:
        return live_brief
    try:
        parsed = json.loads(st.session_state.news_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError as e:
        st.error(f"JSON 파싱 에러: {e}. 직전 유효 브리프로 계산합니다.")
        return None


def render_scalp_mode_sidebar() -> bool:
    """Sidebar toggle for scalping mode — adds 1m refresh option + threshold alert."""
    st.sidebar.header("거래 모드")
    is_scalp = st.sidebar.toggle(
        "단타 모드",
        value=False,
        help=(
            "켜면: 자동 갱신 1분 옵션 활성화, ST 점수 임계(±0.30) 돌파 시 화면 상단 "
            "알림 배너 표시. 분 단위 의사결정용."
        ),
    )
    if is_scalp:
        st.sidebar.caption("⚡ 단타 모드 — 1분 갱신 가능, ST 임계 알림 활성")
    return is_scalp


def render_autorefresh_sidebar(scalp_mode: bool = False) -> int:
    """Sidebar control for auto-refresh interval. Returns interval in minutes (0 = off)."""
    st.sidebar.header("자동 갱신")
    if scalp_mode:
        options = [0, 1, 2, 5, 10, 15]
        labels = {0: "끔 (수동)", 1: "1분 (단타)", 2: "2분", 5: "5분",
                  10: "10분", 15: "15분"}
        default_idx = options.index(1)
    else:
        options = [0, 5, 10, 15, 30, 60]
        labels = {0: "끔 (수동)", 5: "5분", 10: "10분", 15: "15분 (추천)",
                  30: "30분", 60: "60분"}
        default_idx = options.index(15)
    minutes = st.sidebar.selectbox(
        "갱신 주기",
        options=options,
        index=default_idx,
        format_func=lambda m: labels[m],
        help=(
            "선택한 주기로 페이지가 자동 재실행되어 모든 데이터를 새로고침합니다. "
            "단타 모드에선 1분 권장 (분봉 단위 진입/청산 판단용)."
        ),
    )
    if minutes > 0:
        st_autorefresh(interval=minutes * 60 * 1000, key=f"autorefresh_{minutes}m")
        st.sidebar.caption(f"{minutes}분마다 자동 갱신 중")
    else:
        st.sidebar.caption("자동 갱신 꺼짐 — 우상단 Refresh 버튼으로 수동 갱신")
    st.sidebar.markdown("---")
    return minutes


_WEIGHT_EXPLAINER = """
**비유: 날씨 / 계절 / 기후**

같은 "온도"라도 묻는 시간 단위마다 다른 도구가 필요해요.

- "오후에 비?" → 시간별 레이더
- "주말은?" → 주간 예보
- "다음 달은?" → 계절 통계

신호도 똑같이 각자 **"유용한 시간 범위"**가 다릅니다.

---

**ST (수 시간)** — 마이크로 구조 + 강제 플로우 강조

- 오더북 **18%** · 청산 16% · 현물-선물 16% · 펀딩 14%
- 매크로 **2%** · 뉴스 **0%** (수 시간엔 거의 무의미)

**MT (수 일)** — 포지셔닝 + 파생 구조 강조

- 펀딩 **17%** · 옵션스큐 15% · 베이시스 13% · GEX 9%
- 오더북 **2%** (며칠 후엔 책 다 바뀜)

**LT (수 주)** — 거시 + 온체인 + 심층 베이시스 강조

- 매크로 **24%** · 온체인 18% · 베이시스 14% · IV/RV 10%
- 청산 **0%** · 오더북 **1%** (장기 추세엔 의미 없음)

---

**예시**: 오더북에 매수벽 큼(+0.50) + 매크로도 매수 우호(+0.56)

| 시계 | 오더북 기여 | 매크로 기여 |
|---|---|---|
| ST | +0.09 (큼) | +0.011 (미미) |
| LT | +0.005 (무시) | +0.13 (큼) |

같은 데이터를 **시계마다 다르게 듣는 것**. 데이트레이더에게 매크로는 "오후엔 어쩐다는 거야?", 장기 투자자에게 오더북은 "한 시간 뒤면 다 바뀔 텐데?"

---

**한 마디로**

신호마다 정보가 살아있는 시간이 다릅니다. ST/MT/LT 세트는 각 시간 범위에서 신뢰할 만한 신호에 더 무게를 두고, 의미 없는 신호는 거의 무시하도록 설계됐어요. 그래서 같은 시점에 **ST는 NEUTRAL인데 LT는 LONG**이 나올 수 있습니다 — 서로 다른 시간대를 보고 있기 때문.

수정하려면 `src/crypto_analysis/engine.py`의 `WEIGHTS_ST/MT/LT` dict를 직접 편집.
"""


def render_weight_sidebar() -> dict[str, float]:
    st.sidebar.header("가중치 오버라이드")
    with st.sidebar.popover("ST/MT/LT 가중치가 왜 달라요?", use_container_width=True):
        st.markdown(_WEIGHT_EXPLAINER)

    if "weights" not in st.session_state:
        st.session_state.weights = dict(DEFAULT_WEIGHTS)

    preset_cols = st.sidebar.columns(4)
    if preset_cols[0].button("기본"):
        st.session_state.weights = dict(DEFAULT_WEIGHTS)
        st.rerun()
    if preset_cols[1].button("ST"):
        st.session_state.weights = dict(WEIGHTS_ST)
        st.rerun()
    if preset_cols[2].button("MT"):
        st.session_state.weights = dict(WEIGHTS_MT)
        st.rerun()
    if preset_cols[3].button("LT"):
        st.session_state.weights = dict(WEIGHTS_LT)
        st.rerun()

    st.sidebar.caption("프리셋 버튼으로 ST/MT/LT 가중치 세트를 적용하거나, 슬라이더로 직접 조정하세요. 합은 자동 정규화됩니다.")

    weights = {}
    for name in DEFAULT_WEIGHTS.keys():
        weights[name] = st.sidebar.slider(
            name,
            min_value=0.0,
            max_value=0.40,
            value=float(st.session_state.weights.get(name, DEFAULT_WEIGHTS[name])),
            step=0.01,
            key=f"w_{name}",
            help=SIGNAL_HELP.get(name),
        )
    st.session_state.weights = weights

    total = sum(weights.values())
    st.sidebar.caption(f"합계: {total:.2f}")
    return weights


def render_capital_simulation(result, start_btc: float, start_usd: float) -> None:
    """Concrete '1 BTC로 시작하면 얼마' panel."""
    final_mult = (
        float(result.equity_curve.iloc[-1]) if not result.equity_curve.empty else 1.0
    )
    final_btc = start_btc * final_mult
    final_usd = start_usd * final_mult
    pct_return = (final_mult - 1.0) * 100
    color = "#22c55e" if final_mult >= 1.0 else "#ef4444"

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #1e3a5f1f, #0f172a33);
        border:1px solid {color}55; border-radius:8px;
        padding:18px 22px; margin:12px 0;">
            <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
                        opacity:0.65; margin-bottom:10px;">자본 시뮬레이션</div>
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
                <tr>
                    <td style="padding:6px 0; opacity:0.7;">시작 자본</td>
                    <td style="padding:6px 0; text-align:right;">
                        <b>{start_btc:.4f} BTC</b>
                        <span style="opacity:0.55;">  ·  ${start_usd:,.0f}</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px 0; opacity:0.7;">종료 자본</td>
                    <td style="padding:6px 0; text-align:right; color:{color};">
                        <b>{final_btc:.4f} BTC</b>
                        <span style="opacity:0.55;">  ·  ${final_usd:,.0f}</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px 0; opacity:0.7;">순수익</td>
                    <td style="padding:6px 0; text-align:right; color:{color}; font-weight:700;">
                        {pct_return:+.2f}%
                        &nbsp;<span style="opacity:0.6; font-weight:400;">
                        ({(final_btc - start_btc):+.4f} BTC)
                        </span>
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px 0; opacity:0.7;">최대 낙폭 (MDD)</td>
                    <td style="padding:6px 0; text-align:right; color:#ef4444;">
                        {result.max_drawdown * 100:.2f}%
                    </td>
                </tr>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )


def render_backtest(current_spot_usd: float = 77000.0) -> None:
    st.subheader("백테스트 + 자본 시뮬레이션")
    with st.expander("과거 데이터로 신호 엔진 검증 + 단타 시뮬", expanded=False):
        st.caption(
            "backtestable subset: funding + spot_futures (+ oi if ticker hist 제공). "
            "옵션·매크로·뉴스는 historical book 부재로 미반영 — 라이브 결정엔 다 들어감."
        )

        # Row 1: window + resolution + hold
        c1, c2, c3 = st.columns([1, 1, 1])
        days = c1.slider("기간 (일)", 1, 90, 30)
        resolution = c2.selectbox(
            "해상도",
            options=["60", "15", "5", "1"],
            index=0,
            format_func=lambda r: {"1": "1분", "5": "5분", "15": "15분", "60": "1시간"}[r],
            help="분봉은 단타 검증용. 1분 + 30일은 4만 봉 → 3~5분 소요.",
        )
        hold_hours = c3.number_input(
            "Hold hours (최대 보유)", 0.25, 48.0, 8.0, step=0.25,
            help="단타: 0.25~2시간 / 스윙: 4~24시간",
        )

        # Row 2: SL/TP/fee
        c4, c5, c6 = st.columns([1, 1, 1])
        sl_pct = c4.number_input(
            "Stop-loss (%)", 0.0, 5.0, 0.5, step=0.1,
            help="0이면 비활성. 0.5%면 진입가 대비 0.5% 역방향 가면 자동 청산",
        ) / 100.0
        tp_pct = c5.number_input(
            "Take-profit (%)", 0.0, 10.0, 1.0, step=0.1,
            help="0이면 비활성. 1.0%면 진입가 대비 1.0% 정방향 가면 자동 청산",
        ) / 100.0
        fee_bps = c6.number_input(
            "수수료+슬리피지 (bps)", 0.0, 50.0, 10.0, step=1.0,
            help="round-trip 비용 = 2 × 이 값. Deribit perp taker 5bp 권장.",
        )

        # Row 3: capital + run
        c7, c8 = st.columns([2, 1])
        start_btc = c7.number_input(
            "시작 자본 (BTC)", 0.01, 100.0, 1.0, step=0.1,
            help="1 BTC 기준 가정. 결과는 비례 환산.",
        )
        run = c8.button("실행", use_container_width=True)

        if not run:
            return

        # None means disabled
        sl_arg = sl_pct if sl_pct > 0 else None
        tp_arg = tp_pct if tp_pct > 0 else None
        fee_arg = fee_bps / 2.0  # split equally between fee & slippage
        slip_arg = fee_bps / 2.0

        result = run_backtest_live(
            days=days,
            hold_hours=hold_hours,
            resolution=resolution,
            fee_bps=fee_arg,
            slippage_bps=slip_arg,
            stop_loss_pct=sl_arg,
            take_profit_pct=tp_arg,
        )
        if result.trades.empty:
            st.warning("거래가 0건. 데이터 부족이거나 모든 판정이 NEUTRAL.")
            return

        # ─── Capital simulation panel (most prominent)
        render_capital_simulation(result, start_btc=start_btc,
                                  start_usd=start_btc * current_spot_usd)

        # ─── Aggregate metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("거래 수", len(result.trades),
                  help=f"활성 {result.win_count + result.loss_count}건 + NEUTRAL {result.neutral_count}건")
        m2.metric("Hit rate",
                  f"{result.hit_rate:.2%}" if pd.notna(result.hit_rate) else "—")
        m3.metric("평균 PnL/거래",
                  f"{result.avg_fwd_return:+.3%}" if pd.notna(result.avg_fwd_return) else "—")
        m4.metric("Sharpe (연환산)",
                  f"{result.sharpe:.2f}" if pd.notna(result.sharpe) else "—")

        # ─── Exit-reason breakdown
        e1, e2, e3 = st.columns(3)
        e1.metric("Stop-loss 발동", result.sl_hits)
        e2.metric("Take-profit 발동", result.tp_hits)
        e3.metric("Max-hold 청산", result.timeout_exits)

        # ─── Equity curve
        if not result.equity_curve.empty:
            fig = go.Figure()
            curve_color = "#22c55e" if result.total_return >= 0 else "#ef4444"
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values * start_btc,
                mode="lines",
                name="자본 (BTC)",
                line=dict(color=curve_color, width=2),
            ))
            fig.update_layout(
                title=f"자본 곡선  ·  시작 {start_btc:.2f} BTC → 종료 "
                      f"{start_btc * float(result.equity_curve.iloc[-1]):.4f} BTC",
                yaxis_title="BTC",
                height=320,
                margin=dict(l=20, r=20, t=50, b=30),
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ─── Trade-direction breakdown
        with st.expander("거래 분포 / 샘플", expanded=False):
            st.dataframe(
                result.trades["verdict"].value_counts().reset_index().rename(
                    columns={"verdict": "판정", "count": "건수"}
                ),
                hide_index=True,
            )
            if "exit_reason" in result.trades.columns:
                st.dataframe(
                    result.trades["exit_reason"].value_counts().reset_index().rename(
                        columns={"exit_reason": "청산 사유", "count": "건수"}
                    ),
                    hide_index=True,
                )
            st.caption("최근 거래 10건:")
            st.dataframe(
                result.trades.tail(10)[
                    ["ts", "verdict", "score", "pnl", "exit_reason", "bars_held"]
                ],
                hide_index=True,
            )


def render_footer() -> None:
    st.markdown(
        """
        ---
        <div style="font-size:12px; opacity:0.6; text-align:center; padding:10px;">
            확률적 보조 신호입니다. 시장 방향을 확정 예측하지 않으며, 과거 성과는 미래를 보장하지 않습니다.
            자금 관리·손절은 본 시스템과 별개로 사용자가 책임집니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="BTC Futures Decision",
        page_icon=None,
        layout="wide",
    )

    scalp_mode = render_scalp_mode_sidebar()
    render_autorefresh_sidebar(scalp_mode=scalp_mode)
    weights = render_weight_sidebar()

    inputs = fetch_market_inputs()
    macro_df = fetch_macro_panel(days=45)

    render_header(inputs["fetched_at"])

    news_brief = render_news_editor()

    signals = build_signals(inputs, macro_df, news_brief)
    result = fuse(signals, weights=weights)
    decision = decide(result)
    multi = decide_multi(signals)

    # Scalp threshold alert (only if scalp_mode and ST score crosses ±0.30).
    if scalp_mode:
        render_scalp_alert(multi)
        # Always-visible scalping decision card with full trade plan.
        render_scalp_decision_panel(multi, current_price=inputs["perp_mark"])

    # Top-of-page plain conclusion (largest box, sets the frame).
    render_easy_summary(decision, multi, news_brief)

    render_verdict_block(decision, inputs["perp_mark"], inputs["spot_price"])
    render_plain_summary(decision, result, news_brief)
    render_threshold_guide()
    render_timeframes(multi)
    render_action_guide(multi)

    msg_count = render_key_messages(result, news_brief)
    _render_section_conclusion(_conclude_key_messages_text(msg_count))

    render_signal_categories(result)
    _render_section_conclusion(_conclude_signal_categories_text(result))

    with st.expander("12개 신호 자세한 데이터 보기", expanded=False):
        render_signal_table(result)
        _render_section_conclusion(_conclude_signal_table_text(result))

    render_backtest(current_spot_usd=inputs["spot_price"] or 77000.0)
    render_footer()


if __name__ == "__main__":
    main()
