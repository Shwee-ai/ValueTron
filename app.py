# ════════════════════════════════════════════════════════════════════
#   Quant-Sentiment  —  All-in-one Streamlit app
#   • fetches Reddit -> scores -> writes CSVs on a timed interval
#   • reads the same CSVs to power the dashboard
# ════════════════════════════════════════════════════════════════════

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, time, base64, textwrap
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ───────────── configuration ───────────────────────────────────────
TICKERS   = ["NVDA","AAPL","MSFT","TSLA","AMD",
             "ADBE","SCHW","DE","FANG","PLTR"]

REFRESH_MS       = 1_800_000      # whole page ↻ every 30 min
CACHE_TTL        = 900            # price cache 15 min
COLLECT_INTERVAL = 3 * 3600       # refetch Reddit every 3 hours
POST_LIMIT       = 50             # per ticker

POSTS_CSV   = "reddit_posts_scored.csv"
SENTS_CSV   = "reddit_sentiments.csv"

# ───────────── style (optional) ────────────────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
if os.path.exists("tron.png"):
    bg64 = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
         body,.stApp{{
           background:
             linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
             url("data:image/png;base64,{bg64}") center/cover fixed;
           color:#fff; font-family:Arial}}
         h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
         .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
        </style>"""), unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)
st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

# ───────────── sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], 1)
    tkr = st.selectbox("Ticker", TICKERS, 0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    show_sma  = st.checkbox("SMA-20",          True)
    show_macd = st.checkbox("MACD",            True)
    show_rsi  = st.checkbox("RSI",             True)
    show_bb   = st.checkbox("Bollinger Bands", True)
    st.markdown("---")
    show_pe   = st.checkbox("P/E ratio",       True)
    show_de   = st.checkbox("Debt / Equity",   True)
    show_ev   = st.checkbox("EV / EBITDA",     True)

# ───────────── 0. helper: collect+score if needed ──────────────────
def collect_and_score_if_needed():
    # run only if CSV missing or too old
    needs_run = (
        not os.path.exists(SENTS_CSV) or
        time.time() - os.path.getmtime(SENTS_CSV) > COLLECT_INTERVAL
    )
    if not needs_run:
        return  # nothing to do

    st.info("⏳ Fetching fresh Reddit posts…")
    rows = []
    subs = "stocks,investing,wallstreetbets"
    base = "https://api.pushshift.io/reddit/search/submission/"
    for sym in TICKERS:
        try:
            res = requests.get(
                base,
                params={"q": sym,
                        "subreddit": subs,
                        "after": "7d",
                        "size": POST_LIMIT,
                        "sort": "desc"},
                timeout=10)
            data = res.json().get("data", [])
        except Exception:
            data = []
        for d in data:
            rows.append({
                "ticker": sym,
                "title":  d.get("title",""),
                "text":   d.get("selftext",""),
                "score":  d.get("score",0)
            })
        time.sleep(1)  # polite pause

    if not rows:
        st.warning("Reddit fetch failed for all tickers.")
        return

    df = pd.DataFrame(rows)

    # sentiment scoring
    sia = SentimentIntensityAnalyzer()
    def hybrid(txt):
        return ((TextBlob(txt).sentiment.polarity +
                 sia.polarity_scores(txt)["compound"]) / 2)

    df["sentiment"] = (df["title"].fillna("") + " " + df["text"].fillna("")
                      ).apply(hybrid)

    df.to_csv(POSTS_CSV, index=False)

    avg = (df.groupby("ticker")["sentiment"]
              .mean().round(4).reset_index())
    avg.to_csv(SENTS_CSV, index=False)
    st.success("✅ Reddit CSVs updated.")

collect_and_score_if_needed()

# ───────────── 1. read sentiment CSVs ──────────────────────────────
try:
    sent_df = pd.read_csv(SENTS_CSV)
except Exception:
    sent_df = pd.DataFrame({"ticker":[], "sentiment":[]})

row = sent_df[sent_df["ticker"] == tkr]
if row.empty:
    sent_val   = 0.0
    sent_rating = "B"
else:
    sent_val = float(row["sentiment"])
    sent_rating = "A" if sent_val>0.20 else "C" if sent_val<-0.20 else "B"

# posts for table
try:
    df_posts = pd.read_csv(POSTS_CSV)
    df_posts = df_posts[df_posts["ticker"] == tkr][["title","score"]].head(20)
except Exception:
    df_posts = pd.DataFrame()

# ───────────── 2. price + indicators (cached) ──────────────────────
@st.cache_data(ttl=CACHE_TTL)
def load_price(tkr, start, end):
    raw = yf.download(tkr, start=start, end=end+dt.timedelta(days=1),
                      progress=False)
    if raw.empty: return None
    df = raw.copy()
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"]   = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta        = df["Adj Close"].diff()
    rs           = delta.clip(lower=0).rolling(14).mean() / (
                   -delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]    = 100 - 100/(1+rs)
    std          = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]=df["SMA_20"]+2*std
    df["BB_Lower"]=df["SMA_20"]-2*std
    return df

today = dt.date.today()
delta_days = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf, 365)
start = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=delta_days)

price = load_price(tkr, start, today)
if price is None: st.error("Price data error"); st.stop()
last = price.iloc[-1]

# fundamentals
@st.cache_data(ttl=86_400)
def fundamentals(t):
    info = yf.Ticker(t).info
    return dict(pe=info.get("trailingPE",np.nan),
                de=info.get("debtToEquity",np.nan),
                ev=info.get("enterpriseToEbitda",np.nan))
fund = fundamentals(tkr)

# ───────────── 3. scoring blend ────────────────────────────────────
tech = 0.0
if show_sma and "SMA_20" in last: tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd and "MACD" in last:  tech += 1 if last["MACD"]>0 else -1
if show_rsi and "RSI" in last:    tech += 1 if 40<last["RSI"]<70 else -1
if show_bb and {"BB_Upper","BB_Lower"}.issubset(last.index):
    tech += 0.5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech -= 0.5 if last["Adj Close"]<last["BB_Lower"] else 0
if show_pe and not np.isnan(fund["pe"]): tech += 1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += .5 if fund["de"]<1  else -.5
if show_ev and not np.isnan(fund["ev"]): tech += 1 if fund["ev"]<12 else -1

blend = tech_w/100*tech + sent_w/100*sent_val
ver,color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ───────────── 4. UI ───────────────────────────────────────────────
tab_v, tab_ta, tab_f, tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit / Preset"])

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
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(color="#0ff")))
    if show_sma: fig.add_trace(go.Scatter(x=df.index,y=df["SMA_20"],name="SMA-20", line=dict(color="#ff0",dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="Upper BB", line=dict(color="#0f0",dash="dot")))
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="Lower BB", line=dict(color="#0f0",dash="dot")))
    fig.update_layout(template="plotly_dark",height=350,title="Price / SMA / Bollinger")
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(df["MACD"],height=200)
    if show_rsi:  st.line_chart(df["RSI"],height=200)

with tab_f:
    st.header("Key Ratios")
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt / Equity","EV / EBITDA"],
        "Value":[fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    st.header("Latest Reddit Mentions (live or preset)")
    if not df_posts.empty:
        st.dataframe(df_posts, hide_index=True, use_container_width=True)
