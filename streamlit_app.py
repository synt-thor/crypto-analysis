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


@st.cache_data(ttl=900, show_spinner="📰 실시간 뉴스 페치 + Gemini 점수화…")
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


@st.cache_data(ttl=3600, show_spinner="과거 데이터 백필 + 백테스트 실행 중…")
def run_backtest_live(days: int, hold_hours: int):
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    perp = deribit.tradingview_chart_data(PERPETUAL, start_ms, end_ms, resolution="60")
    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))
    # Coinbase candles instead of Binance (Binance blocks US datacenter IPs).
    spot = exchanges.coinbase_candles_range("BTC-USD", granularity=3600,
                                            start_ms=start_ms, end_ms=end_ms)

    return run_backtest(
        perp_ohlcv=perp.sort_values("ts").reset_index(drop=True),
        spot_ohlcv=spot.sort_values("ts").reset_index(drop=True),
        funding_df=funding.sort_values("ts").reset_index(drop=True),
        ticker_hist=None,
        hold_hours=hold_hours,
        step_hours=max(2, hold_hours // 2),
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
        if st.button("🔄 Refresh", use_container_width=True):
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
    for col, label, dec, res in zip(
        cols,
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
                </div>
                """,
                unsafe_allow_html=True,
            )
            top3 = res.contributions[:3]
            for c in top3:
                arrow = "▲" if c["contribution"] > 0 else ("▼" if c["contribution"] < 0 else "•")
                st.caption(
                    f"{arrow} **{c['name']}** {c['contribution']:+.3f}  ·  {c['rationale'][:80]}"
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

    # 4) Status badge.
    if live_brief:
        n_events = len(live_brief.get("events", []))
        gen_at = live_brief.get("generated_at_utc", "")
        try:
            gen_dt = datetime.fromisoformat(gen_at)
            mins_ago = max(0, int((datetime.now(tz=timezone.utc) - gen_dt).total_seconds() / 60))
            ts_label = f"{mins_ago}분 전"
        except Exception:
            ts_label = "방금"
        badge_color = "#ef4444"
        badge_text = f"🔴 LIVE  ·  {n_events} events  ·  {ts_label} 갱신"
    elif api_key_present:
        badge_color = "#f59e0b"
        badge_text = "⚠️ LIVE 페치 실패 — 디스크 baseline 사용"
    else:
        badge_color = "#9ca3af"
        badge_text = "📁 Disk baseline (Gemini API 키 미설정)"

    st.markdown(
        f"""<div style="display:inline-block; border:1px solid {badge_color}88;
        border-radius:6px; padding:4px 10px; background:{badge_color}11;
        font-size:12px; margin-bottom:8px;">{badge_text}</div>""",
        unsafe_allow_html=True,
    )

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
            if st.button("📥 디스크에서 로드"):
                if NEWS_BRIEF_PATH.exists():
                    st.session_state.news_text = NEWS_BRIEF_PATH.read_text()
                    st.rerun()
                else:
                    st.warning(f"{NEWS_BRIEF_PATH} 가 없습니다.")
        with col_apply:
            apply_clicked = st.button("✅ 적용 (수동 모드)")

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


def render_weight_sidebar() -> dict[str, float]:
    st.sidebar.header("가중치 오버라이드")

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


def render_backtest() -> None:
    st.subheader("백테스트 (live API에서 즉시 다운로드)")
    with st.expander("과거 데이터로 신호 엔진 검증", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 1])
        days = c1.slider("기간 (일)", 7, 90, 30)
        hold = c2.selectbox("Hold hours", [4, 8, 24], index=1)
        run = c3.button("▶ 실행", use_container_width=True)
        if not run:
            st.caption("백테스트는 funding + spot_futures 신호의 서브셋만 사용합니다 — 옵션·매크로·뉴스 등은 미반영.")
            return

        result = run_backtest_live(days=days, hold_hours=hold)
        if result.trades.empty:
            st.warning("거래가 0건. 데이터가 부족하거나 모든 판정이 NEUTRAL.")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("거래 수", len(result.trades))
        m2.metric("Hit rate", f"{result.hit_rate:.2%}" if pd.notna(result.hit_rate) else "—")
        m3.metric("평균 PnL", f"{result.avg_fwd_return:+.3%}" if pd.notna(result.avg_fwd_return) else "—")
        m4.metric("Sharpe", f"{result.sharpe:.2f}" if pd.notna(result.sharpe) else "—")

        if not result.equity_curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode="lines",
                name="equity",
                line=dict(color="#22c55e", width=2),
            ))
            fig.update_layout(
                title="에쿼티 커브 (1.0 = 시작)",
                yaxis_title="multiple",
                height=320,
                margin=dict(l=20, r=20, t=40, b=30),
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            result.trades["verdict"].value_counts().reset_index().rename(
                columns={"verdict": "판정", "count": "건수"}
            ),
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
        page_icon="📈",
        layout="wide",
    )

    weights = render_weight_sidebar()

    inputs = fetch_market_inputs()
    macro_df = fetch_macro_panel(days=45)

    render_header(inputs["fetched_at"])

    news_brief = render_news_editor()

    signals = build_signals(inputs, macro_df, news_brief)
    result = fuse(signals, weights=weights)
    decision = decide(result)
    multi = decide_multi(signals)

    render_verdict_block(decision, inputs["perp_mark"], inputs["spot_price"])
    render_timeframes(multi)
    render_signal_table(result)
    render_backtest()
    render_footer()


if __name__ == "__main__":
    main()
