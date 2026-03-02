"""
app.py
──────
Streamlit dashboard for the Regime-Based Trading System.

Layout
──────
1. Top banner  – Current Signal, Detected Regime, Confirmation count
2. Metrics row – Total Return, Alpha, Win Rate, Max Drawdown, # Trades
3. Main chart  – Candlestick + EMA overlays + regime-coloured background
                 + trade entry/exit markers
4. Equity curve – Strategy vs Buy & Hold
5. Regime pie  + Strategy Rules card
6. Trade log   – Detailed per-trade table
"""

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester import INITIAL_CAPITAL, run_backtest
from data_loader import fetch_ohlcv_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Regime colour palette ─────────────────────────────────────────────────────
REGIME_BG = {
    "Bull Run":   "rgba(0, 210, 110, 0.13)",
    "Bear/Crash": "rgba(220, 50,  50, 0.13)",
    "Neutral":    "rgba(120, 120, 120, 0.05)",
}
REGIME_TEXT_COLOR = {
    "Bull Run":   "#00d26e",
    "Bear/Crash": "#ef5350",
    "Neutral":    "#ffa726",
}

# ── Streamlit caching helpers ─────────────────────────────────────────────────
POPULAR_TICKERS = [
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    # US Large-cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META",
    # ETFs
    "SPY", "QQQ", "GLD",
]


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv_data(ticker=ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_backtest(df: pd.DataFrame, ticker: str):
    return run_backtest(df)


# ── Chart builders ─────────────────────────────────────────────────────────────
def _add_regime_rects(fig: go.Figure, df: pd.DataFrame) -> None:
    """Colour the chart background according to the detected regime."""
    groups = (df["Regime"] != df["Regime"].shift()).cumsum()
    for _, grp in df.groupby(groups):
        regime = grp["Regime"].iloc[0]
        fig.add_vrect(
            x0=grp.index[0],
            x1=grp.index[-1],
            fillcolor=REGIME_BG.get(regime, REGIME_BG["Neutral"]),
            opacity=1,
            layer="below",
            line_width=0,
        )


def build_candlestick_chart(df: pd.DataFrame, trades, ticker: str = "") -> go.Figure:
    """Candlestick + EMA lines + regime backgrounds + trade markers."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{ticker} · Regime Background", "Volume"),
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker or "Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    # ── EMA overlays ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_50"],
            name="EMA 50",
            line=dict(color="#ff9800", width=1.2),
            opacity=0.85,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_200"],
            name="EMA 200",
            line=dict(color="#42a5f5", width=1.2),
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    # ── Trade markers ─────────────────────────────────────────────────────────
    if trades:
        entry_times = [t.entry_time for t in trades]
        entry_prices = [t.entry_price for t in trades]
        exit_times = [t.exit_time for t in trades]
        exit_prices = [t.exit_price for t in trades]

        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers",
                marker=dict(symbol="triangle-up", size=11, color="#00e676", line=dict(width=1, color="#fff")),
                name="Entry",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode="markers",
                marker=dict(symbol="triangle-down", size=11, color="#ff1744", line=dict(width=1, color="#fff")),
                name="Exit",
            ),
            row=1,
            col=1,
        )

    # ── Volume bars ───────────────────────────────────────────────────────────
    bar_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=bar_colors,
            opacity=0.65,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ── Regime background rectangles ─────────────────────────────────────────
    _add_regime_rects(fig, df)

    fig.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2130")

    return fig


def build_equity_chart(portfolio: pd.Series, df: pd.DataFrame) -> go.Figure:
    """Strategy equity curve vs normalised Buy-and-Hold."""
    bh = INITIAL_CAPITAL * (df["Close"] / float(df["Close"].iloc[0]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio,
            name="Strategy",
            line=dict(color="#00e676", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.05)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bh.index,
            y=bh,
            name="Buy & Hold",
            line=dict(color="#42a5f5", width=1.5, dash="dot"),
        )
    )
    fig.add_hline(
        y=INITIAL_CAPITAL,
        line=dict(color="#555", width=1, dash="dash"),
        annotation_text="Starting Capital",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Equity Curve — Strategy vs Buy & Hold",
        height=340,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
    return fig


def build_regime_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["Regime"].value_counts()
    color_map = {
        "Bull Run":   "#00d26e",
        "Bear/Crash": "#ef5350",
        "Neutral":    "#ffa726",
    }
    colors = [color_map.get(label, "#888") for label in counts.index]

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.45,
            marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Regime Distribution",
        height=300,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _trades_dataframe(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [
        {
            "Entry Time":    t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "Exit Time":     t.exit_time.strftime("%Y-%m-%d %H:%M"),
            "Entry $":       f"{t.entry_price:,.2f}",
            "Exit $":        f"{t.exit_price:,.2f}",
            "Capital In":    f"{t.capital_at_entry:,.2f}",
            "P&L ($)":       round(t.pnl, 2),
            "Return (lev)":  f"{t.return_pct:+.2f}%",
            "Votes":         t.votes,
            "Exit Reason":   t.exit_reason,
        }
        for t in trades
    ]
    return pd.DataFrame(rows)


# ── Metric card helper ────────────────────────────────────────────────────────
def _kpi_card(label: str, value: str, color: str = "#e0e0e0") -> str:
    return f"""
    <div style="background:#1a1d2e;border-radius:10px;padding:18px 14px;text-align:center;
                border:1px solid #2a2d40;">
        <div style="color:#7a7f99;font-size:12px;letter-spacing:1px;text-transform:uppercase">
            {label}
        </div>
        <div style="color:{color};font-size:28px;font-weight:700;margin-top:6px">{value}</div>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Sidebar: ticker selector ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Asset Selection")
        preset = st.selectbox("Popular tickers", POPULAR_TICKERS, index=0)
        custom = st.text_input(
            "Or enter any Yahoo Finance ticker",
            placeholder="e.g. GOLD, ^GSPC, EURUSD=X …",
        )
        ticker = custom.strip().upper() if custom.strip() else preset

        st.markdown("---")
        st.caption(
            "Accepts any symbol understood by Yahoo Finance: "
            "stocks, ETFs, crypto (`BTC-USD`), forex (`EURUSD=X`), "
            "indices (`^GSPC`), futures (`GC=F`), …"
        )

    st.markdown(
        f"<h1 style='text-align:center;color:#e0e0e0'>📈 Regime-Based Trading System</h1>"
        f"<p style='text-align:center;color:#7a7f99;margin-top:-10px'>"
        f"HMM · 7 States · 8-Confirmation Voting · <b style='color:#42a5f5'>{ticker}</b> Hourly</p>",
        unsafe_allow_html=True,
    )

    # ── Load data & run backtest ──────────────────────────────────────────────
    data_placeholder = st.empty()
    with data_placeholder.container():
        with st.spinner(f"Fetching {ticker} historical data …"):
            try:
                df = _load_data(ticker)
            except Exception as exc:
                st.error(f"Could not load data for **{ticker}**: {exc}")
                st.stop()

        if df.empty:
            st.error(f"No hourly data found for **{ticker}**. Try a different symbol.")
            st.stop()

        with st.spinner(f"Training HMM & running backtest for {ticker} … (first run takes ~30s)"):
            result = _run_backtest(df, ticker)
    data_placeholder.empty()

    # ── Signal banner ─────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    sig_color = "#00e676" if result.current_signal == "LONG" else "#ff5252"
    reg_color = REGIME_TEXT_COLOR.get(result.current_regime, "#e0e0e0")
    last_votes = int(result.df["Votes"].iloc[-1])
    last_price = float(result.df["Close"].iloc[-1])

    c1.markdown(_kpi_card("Current Signal",    result.current_signal,  sig_color), unsafe_allow_html=True)
    c2.markdown(_kpi_card("Detected Regime",   result.current_regime,  reg_color), unsafe_allow_html=True)
    c3.markdown(_kpi_card("Confirmations",     f"{last_votes}/8",      "#ffa726"), unsafe_allow_html=True)
    c4.markdown(_kpi_card(f"{ticker} Price",    f"${last_price:,.2f}",  "#42a5f5"), unsafe_allow_html=True)

    # ── Performance metrics ───────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)

    ret_color  = "#00e676" if result.total_return >= 0 else "#ef5350"
    alph_color = "#00e676" if result.alpha >= 0        else "#ef5350"

    m1.markdown(_kpi_card("Total Return",    f"{result.total_return:+.1%}",    ret_color),  unsafe_allow_html=True)
    m2.markdown(_kpi_card("Alpha vs B&H",    f"{result.alpha:+.1%}",           alph_color), unsafe_allow_html=True)
    m3.markdown(_kpi_card("Win Rate",        f"{result.win_rate:.1%}",         "#e0e0e0"),  unsafe_allow_html=True)
    m4.markdown(_kpi_card("Max Drawdown",    f"{result.max_drawdown:.1%}",     "#ef5350"),  unsafe_allow_html=True)
    m5.markdown(_kpi_card("Total Trades",    str(result.n_trades),             "#e0e0e0"),  unsafe_allow_html=True)

    # ── Time-range filter ─────────────────────────────────────────────────────
    st.markdown("---")
    _, col_sel = st.columns([4, 1])
    with col_sel:
        range_opt = st.selectbox(
            "Chart Range",
            ["All History", "Last 365 Days", "Last 180 Days", "Last 90 Days"],
        )

    slice_map = {
        "Last 90 Days":  -90  * 24,
        "Last 180 Days": -180 * 24,
        "Last 365 Days": -365 * 24,
    }
    df_view = result.df.iloc[slice_map.get(range_opt, 0):]
    port_view = result.portfolio_values.loc[df_view.index]

    # ── Candlestick chart ─────────────────────────────────────────────────────
    fig_candle = build_candlestick_chart(df_view, result.trades, ticker)
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    fig_equity = build_equity_chart(port_view, df_view)
    st.plotly_chart(fig_equity, use_container_width=True)

    # ── Regime pie + strategy rules ───────────────────────────────────────────
    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        fig_pie = build_regime_pie(result.df)
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.markdown("### Strategy Rules")
        st.markdown(
            """
**HMM Core**
- 7-state `GaussianHMM` on Returns, Range, Volume Volatility
- Bull Run = highest-mean-return state
- Bear/Crash = lowest-mean-return state

**Entry** — Bull Run regime **AND** ≥ 7/8 confirmations:

| # | Condition |
|---|-----------|
| 1 | RSI < 90 |
| 2 | Momentum (20-bar) > 1 % |
| 3 | Volatility (20-bar σ) < 6 % |
| 4 | Volume > 20-bar SMA |
| 5 | ADX > 25 |
| 6 | Price > EMA 50 |
| 7 | Price > EMA 200 |
| 8 | MACD > Signal |

**Exit** — Regime flips to Bear/Crash
**Cooldown** — 48 h hard block after every exit
**Leverage** — None (1×)
**Capital** — Starting $10 000
"""
        )

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Trade Log")
    trades_df = _trades_dataframe(result.trades)
    if trades_df.empty:
        st.info("No trades were executed during the backtest period.")
    else:
        def _style_pnl(val):
            if isinstance(val, (int, float)):
                color = "#00e676" if val >= 0 else "#ef5350"
                return f"color: {color}; font-weight: 600"
            return ""

        styled = (
            trades_df.style
            .applymap(_style_pnl, subset=["P&L ($)"])
            .format({"P&L ($)": "{:+,.2f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<br><p style='text-align:center;color:#3a3f55;font-size:12px'>"
        "Regime-Based Trading System · For educational purposes only · Not financial advice"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
