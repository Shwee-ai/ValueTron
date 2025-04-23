# ════════════════════════════════════════════════════════════════════
#  Quant-Sentiment Dashboard  –  WSJ / NewsAPI edition
#  Paste this file as app.py   (Python ≥ 3.9)
# ════════════════════════════════════════════════════════════════════
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, base64, textwrap
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── settings you may tweak ─────────────────────────────────────────
TICKERS    = ["NVDA","AAPL","MSFT","TSLA","AMD",
              "ADBE","SCHW","DE","FANG","PLTR"]
REFRESH_MS = 1_800_000      # full-page auto-refresh 30 min
CACHE_TTL  = 900            # price & news cache 15 min

# ─── page style (optional Tron background) ──────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
if os.path.exists("tron.png"):
    bg = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
          body,.stApp{{
            background:
              linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
              url("data:image/png;base64,{bg}") center/cover fixed;
            color:#fff;font-family:Arial}}
          h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
          .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
        </style>"""), unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)
st_autorefresh(interval=REFRESH_MS, key="refresh")

# ─── sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], 1)
    tkr = st.selectbox("Ticker", TICKERS, 0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",  True)
    show_rsi  = st.checkbox("RSI",   True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("---")
    show_pe = st.checkbox("P/E ratio",       True)
    show_de = st.checkbox("Debt / Equity",   True)
    show_ev = st.checkbox("EV / EBITDA",     True)

# ─── helper: date range -----------------------------------------------------
today = dt.date.today()
delta = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf, 365)
start = dt.date(today.year,1,1) if tf=="YTD" else today - dt.timedelta(days=delta)

# ─── price & technicals -----------------------------------------------------
@st.cache_data(ttl=CACHE_TTL)
def load_price(tkr, start, end):
    raw = yf.download(tkr, start=start, end=end+dt.timedelta(days=1),
                      progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(tkr, level=0, axis=1)
    if raw.empty:
        return None
    df = raw.copy()
    df["Adj Close"] = df.get("Adj Close", df["Close"])
    df["SMA_20"]    = df["Adj Close"].rolling(20).mean()
    df["MACD"]      = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs    = delta.clip(lower=0).rolling(14).mean() / (
            -delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]       = 100 - 100/(1+rs)
    std             = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]  = df["SMA_20"] + 2*std
    df["BB_Lower"]  = df["SMA_20"] - 2*std
    return df

price = load_price(tkr, start, today)
if price is None: st.error("No price data."); st.stop()
price = price.dropna(subset=["Adj Close"])
if price.empty:   st.error("Not enough rows."); st.stop()
last = price.iloc[-1]

# ─── fundamentals -----------------------------------------------------------
@st.cache_data(ttl=86_400)
def fundamentals(tkr):
    finfo = yf.Ticker(tkr).fast_info or {}
    pe = finfo.get("trailingPe",   np.nan)
    de = finfo.get("debtToEquity", np.nan)
    ev = finfo.get("evToEbitda",   np.nan)
    if np.isnan(pe) or np.isnan(de) or np.isnan(ev):
        try:
            info = yf.Ticker(tkr).info
            pe = info.get("trailingPE", pe)
            de = info.get("debtToEquity", de)
            ev = info.get("enterpriseToEbitda", ev)
        except Exception:
            pass
    return dict(pe=pe, de=de, ev=ev)

# ─── WSJ / fallback NewsAPI sentiment --------------------------------------
@st.cache_data(ttl=CACHE_TTL)
def news_sentiment(tkr: str):
    key = st.secrets.get("NEWSAPI_KEY", "f927b70db70a46cba0b04974864e181d")
    if not key:
        st.warning("NEWSAPI_KEY missing in secrets.")
        return 0.0, "B", pd.DataFrame()

    def fetch_news(query, domains=None):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "pageSize": 50,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": key
        }
        if domains:
            params["domains"] = domains
        try:
            return requests.get(url, params=params, timeout=10).json().get("articles", [])
        except Exception:
            return []

    # symbol OR company name improves hit-rate
    company = yf.Ticker(tkr).info.get("shortName", tkr)
    query   = f"{tkr} OR {company}"

    articles = fetch_news(query, domains="wsj.com")     # WSJ only first
    if not articles:
        articles = fetch_news(query)                    # fallback all sources

    if not articles:
        return 0.0, "B", pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    def hybrid(text):
        return ((TextBlob(text).sentiment.polarity +
                 sia.polarity_scores(text)["compound"]) / 2)

    rows = [{"title": a["title"], "source": a["source"]["name"]} for a in articles]
    avg  = sum(hybrid(r["title"]) for r in rows) / len(rows)
    rating = "A" if avg > 0.20 else "C" if avg < -0.20 else "B"

    return avg, rating, pd.DataFrame(rows)

fund = fundamentals(tkr)
sent_val, sent_rating, df_news = news_sentiment(tkr)

# ─── score blend ------------------------------------------------------------
tech = 0.0
if show_sma  and "SMA_20" in last: tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd and "MACD"   in last: tech += 1 if last["MACD"]>0 else -1
if show_rsi  and "RSI"    in last: tech += 1 if 40<last["RSI"]<70 else -1
if show_bb   and {"BB_Upper","BB_Lower"}.issubset(last.index):
    if last["Adj Close"]>last["BB_Upper"]: tech += 0.5
    if last["Adj Close"]<last["BB_Lower"]: tech -= 0.5

if show_pe and not np.isnan(fund["pe"]): tech += 1   if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += 0.5 if fund["de"]<1  else -0.5
if show_ev and not np.isnan(fund["ev"]): tech += 1   if fund["ev"]<12 else -1

blend = tech_w/100 * tech + sent_w/100 * sent_val
ver, color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── tabs -------------------------------------------------------------------
tab_v, tab_ta, tab_f, tab_n = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","📰 News"])

with tab_v:
    st.header("Overall Verdict")
    st.markdown(f"<h1 style='color:{color};text-align:center'>{ver}</h1>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",  f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score",  f"{sent_val:.2f}")
    c4.metric("Blended",     f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech + {sent_w}% Sentiment")

with tab_ta:
    df = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Price",
                             line=dict(color="#0ff")))
    if show_sma and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA-20",
                                 line=dict(color="#ff0", dash="dash")))
    if show_bb and {"BB_Upper","BB_Lower"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper BB",
                                 line=dict(color="#0f0", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower BB",
                                 line=dict(color="#0f0", dash="dot")))
    fig.update_layout(template="plotly_dark", height=350,
                      title="Price / SMA / Bollinger")
    st.plotly_chart(fig, use_container_width=True)
    if show_macd and "MACD" in df.columns: st.line_chart(df["MACD"], height=200)
    if show_rsi  and "RSI"  in df.columns: st.line_chart(df["RSI"], height=200)

with tab_f:
    st.header("Key Ratios")
    st.table(pd.DataFrame({
        "Metric": ["P/E","Debt / Equity","EV / EBITDA"],
        "Value":  [fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_n:
    st.header("Latest News Headlines")
    if not df_news.empty:
        st.dataframe(df_news, hide_index=True, use_container_width=True)
