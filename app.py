# ────────── imports ────────────────────────────────────────────────
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, base64
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ────────── constants (easy to edit later) ─────────────────────────
TICKERS    = ["NVDA","AAPL","MSFT","TSLA","AMD",
              "ADBE","SCHW","DE","FANG","PLTR"]
REFRESH_MS = 1_800_000    # page ↻ every 30 min
CACHE_TTL  = 900          # price & reddit cache = 15 min

# ────────── page config & optional background image ───────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
if os.path.exists("tron.png"):
    bg = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(f"""
    <style>
      body,.stApp{{background:
        linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
        url("data:image/png;base64,{bg}") center/cover fixed;
        color:#fff;font-family:Arial}}
      h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
      .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
    </style>""", unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)

# ────────── auto-refresh (browser side) ────────────────────────────
st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

# ────────── sidebar ------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr = st.selectbox("Ticker", TICKERS, index=0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD", True)
    show_rsi  = st.checkbox("RSI",  True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("---")
    show_pe = st.checkbox("P/E ratio",      True)
    show_de = st.checkbox("Debt / Equity",  True)
    show_ev = st.checkbox("EV / EBITDA",    True)

# ────────── date range --------------------------------------------
today  = dt.date.today()
delta  = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start  = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=delta)

# ────────── price & indicators ------------------------------------
@st.cache_data(ttl=CACHE_TTL)
def load_price(tkr, start, end):
    raw = yf.download(tkr, start=start, end=end+dt.timedelta(days=1),
                      progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(tkr, level=0, axis=1)
    if raw.empty: return None
    df = raw.copy()
    df["Adj Close"] = df.get("Adj Close", df["Close"])
    df["SMA_20"]    = df["Adj Close"].rolling(20).mean()
    df["MACD"]      = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (
         -delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2*std
    df["BB_Lower"] = df["SMA_20"] - 2*std
    return df

price = load_price(tkr, start, today)
if price is None: st.error("No price data."); st.stop()
price = price.dropna(subset=["Adj Close"])
if price.empty: st.error("Not enough rows."); st.stop()
last = price.iloc[-1]

# ────────── fundamentals (P/E, D/E, EV/EBITDA) ---------------------
@st.cache_data(ttl=86_400)
def fundamentals(tkr):
    finfo = yf.Ticker(tkr).fast_info or {}
    pe = finfo.get("trailingPe", np.nan)
    de = finfo.get("debtToEquity", np.nan)
    ev = finfo.get("evToEbitda", np.nan)
    if np.isnan(pe) or np.isnan(de) or np.isnan(ev):
        try:
            info = yf.Ticker(tkr).info
            pe = info.get("trailingPE", pe)
            de = info.get("debtToEquity", de)
            ev = info.get("enterpriseToEbitda", ev)
        except Exception: pass
    return dict(pe=pe, de=de, ev=ev)

# ────────── reddit sentiment (dual query + fallback) ---------------
@st.cache_data(ttl=CACHE_TTL)
def reddit_sentiment(tkr):
    hdr  = {"User-Agent":"QuantDash/0.1"}
    subs = ["stocks","investing","wallstreetbets"]
    rows = []
    for sub in subs:
        for q in (tkr, f"${tkr}"):                # plain + $symbol
            url = (f"https://api.reddit.com/r/{sub}/search"
                   f"?q={q}&restrict_sr=true&sort=new&limit=30")
            try:
                r = requests.get(url, headers=hdr, timeout=10)
                r.raise_for_status()
                js = r.json()
                rows += [{
                    "title":  c["data"].get("title",""),
                    "text":   c["data"].get("selftext",""),
                    "score":  c["data"].get("score",0)
                } for c in js.get("data",{}).get("children",[])]
            except Exception as e:
                st.warning(f"Reddit fetch failed /r/{sub} {q}: {e}")
                continue

    # ↩︎ fallback to Pushshift if still empty
    if not rows:
        url = (f"https://api.pushshift.io/reddit/search/submission/"
               f"?q={tkr}&subreddit=stocks,investing,wallstreetbets&sort=desc&size=50")
        try:
            ps = requests.get(url, timeout=10).json().get("data", [])
            rows = [{
                "title": p.get("title",""),
                "text":  p.get("selftext",""),
                "score": p.get("score",0)
            } for p in ps]
        except Exception:
            pass

    if not rows:
        return 0.0, "B", pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    def hybrid(r):
        txt   = f"{r['title']} {r['text']}"
        base  = (TextBlob(txt).sentiment.polarity +
                 sia.polarity_scores(txt)["compound"]) / 2
        return base * min(r["score"],100) / 100

    avg = sum(hybrid(r) for r in rows) / len(rows)
    rating = "A" if avg > 0.2 else "C" if avg < -0.2 else "B"
    df = pd.DataFrame([{"title":r["title"],"score":r["score"]} for r in rows])
    return avg, rating, df

# ---------- helper calls (must precede scoring) --------------------
fund = fundamentals(tkr)
sent_val, sent_rating, df_posts = reddit_sentiment(tkr)

# ────────── scoring -------------------------------------------------
tech = 0.0
if show_sma and "SMA_20" in last:   tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd and "MACD"  in last:   tech += 1 if last["MACD"]>0 else -1
if show_rsi and "RSI"   in last:    tech += 1 if 40<last["RSI"]<70 else -1
if show_bb and {"BB_Upper","BB_Lower"}.issubset(last.index):
    tech += 0.5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech -= 0.5 if last["Adj Close"]<last["BB_Lower"] else 0

if show_pe and not np.isnan(fund["pe"]): tech += 1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += 0.5 if fund["de"]<1 else -0.5
if show_ev and not np.isnan(fund["ev"]): tech += 1 if fund["ev"]<12 else -1

blend = tech_w/100 * tech + sent_w/100 * sent_val
ver, color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ────────── UI tabs -------------------------------------------------
tab_v, tab_ta, tab_f, tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with tab_v:
    st.header("Overall Verdict")
    st.markdown(f"<h1 style='color:{color};text-align:center'>{ver}</h1>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",   f"{tech:.2f}")
    c2.metric("Sent Rating",  sent_rating)
    c3.metric("Sent Score",   f"{sent_val:.2f}")
    c4.metric("Blended",      f"{blend:.2f}")
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
    if show_rsi and "RSI" in df.columns: st.line_chart(df["RSI"], height=200)
