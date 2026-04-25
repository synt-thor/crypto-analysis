"""Streamlit web UI for the BTC futures decision engine.

Wraps existing crypto_analysis modules — no new business logic.
Run locally: `streamlit run streamlit_app.py`
"""

from __future__ import annotations

import json
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

NEWS_BRIEF_PATH = Path(__file__).parent / "data" / "news_brief.json"

VERDICT_COLORS = {"LONG": "#22c55e", "SHORT": "#ef4444", "NEUTRAL": "#9ca3af"}


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
    """Returns parsed news_brief dict (or None on parse error)."""
    st.subheader("뉴스 브리프")

    if "news_text" not in st.session_state:
        try:
            st.session_state.news_text = NEWS_BRIEF_PATH.read_text() if NEWS_BRIEF_PATH.exists() else "{}"
        except Exception:
            st.session_state.news_text = "{}"

    with st.expander("뉴스 브리프 보기 / 편집", expanded=False):
        col_load, col_apply, _ = st.columns([1, 1, 4])
        with col_load:
            if st.button("📥 디스크에서 로드"):
                if NEWS_BRIEF_PATH.exists():
                    st.session_state.news_text = NEWS_BRIEF_PATH.read_text()
                    st.rerun()
                else:
                    st.warning(f"{NEWS_BRIEF_PATH} 가 없습니다.")
        with col_apply:
            apply_clicked = st.button("✅ 적용")

        text = st.text_area(
            "JSON",
            value=st.session_state.news_text,
            height=300,
            key="news_editor",
        )
        if apply_clicked:
            st.session_state.news_text = text

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
