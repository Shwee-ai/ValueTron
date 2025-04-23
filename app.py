# ═══════════════════════════════════════════════════════════════════
#  Quant-Sentiment Dashboard – Reddit-only (no presets)
#  Combine: fetch ▶︎ score ▶︎ CSV ▶︎ Streamlit UI
# ═══════════════════════════════════════════════════════════════════
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, time, base64, textwrap
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── settings ──────────────────────────────────────────────────────
TICKERS         = ["NVDA","AMD","ADBE","VRTX","SCHW",
                   "CROX","DE","FANG","TMUS","PLTR"]
SUBS            = ["stocks","investing","wallstreetbets"]
HEADERS         = {"User-Agent": "Mozilla/5.0 (StockDashBot/0.1)"}

REFRESH_MS      = 1_800_000      # page ↻ 30 min
PRICE_CACHE_TTL = 900            # price cache 15 min
COLLECT_EVERY   = 3 * 3600       # refetch Reddit every 3 h
POST_LIMIT      = 40             # per subreddit

POSTS_CSV  = "reddit_posts_scored.csv"
SENTS_CSV  = "reddit_sentiments.csv"

# ─── optional Tron background ──────────────────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
if os.path.exists("tron.png"):
    b64 = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(textwrap.dedent(f"""
      <style>
        body,.stApp{{
          background:linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
          url("data:image/png;base64,{b64}") center/cover fixed;
          color:#fff;font-family:Arial}}
        h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
        .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
      </style>"""), unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)
st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

# ─── sidebar ───────────────────────────────────────────────────────
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
    show_pe = st.checkbox("P/E ratio",         True)
    show_de = st.checkbox("Debt / Equity",     True)
    show_ev = st.checkbox("EV / EBITDA",       True)

# ─── 0 · fetch+score Reddit if cache stale ─────────────────────────
def fetch_reddit_posts(ticker: str, limit=POST_LIMIT):
    rows = []
    for sub in SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={ticker}&restrict_sr=1&sort=new&limit={limit}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                time.sleep(1); continue
            for child in resp.json().get("data", {}).get("children", []):
                d = child["data"]
                rows.append({
                    "ticker": ticker,
                    "title":  d.get("title",""),
                    "text":   d.get("selftext",""),
                    "score":  d.get("score",0),
                    "utc":    d.get("created_utc",0),
                    "sub":    sub
                })
        except Exception:
            pass
        time.sleep(1)                        # avoid 429
    return rows

def collect_if_needed():
    if os.path.exists(SENTS_CSV) and time.time() - os.path.getmtime(SENTS_CSV) < COLLECT_EVERY:
        return
    st.info("⏳ Collecting fresh Reddit posts…")
    rows = []
    for sym in TICKERS:
        rows += fetch_reddit_posts(sym)
    if not rows:
        st.warning("Reddit fetch failed (no posts)."); return
    df = pd.DataFrame(rows)
    sia = SentimentIntensityAnalyzer()
    def hybrid(row):
        txt = f'{row["title"]} {row["text"]}'
        return ((TextBlob(txt).sentiment.polarity +
                 sia.polarity_scores(txt)["compound"]) / 2)
    df["sentiment"] = df.apply(hybrid, axis=1)
    df.to_csv(POSTS_CSV, index=False)
    df.groupby("ticker")["sentiment"].mean().round(4).reset_index(
        ).to_csv(SENTS_CSV, index=False)
    st.success("✅ Reddit CSVs updated.")

collect_if_needed()

# ─── 1 · load sentiment CSV (neutral if missing) ───────────────────
try:
    sent_df = pd.read_csv(SENTS_CSV)
    sent_val = float(sent_df.loc[sent_df["ticker"] == tkr, "sentiment"].values[0])
except Exception:
    sent_val = 0.0
sent_rating = "A" if sent_val>0.20 else "C" if sent_val<-0.20 else "B"

try:
    df_posts = (pd.read_csv(POSTS_CSV)
                  .query("ticker == @tkr")[["title","score"]].head(20))
except Exception:
    df_posts = pd.DataFrame()

# ─── 2 · price & indicators (robust) ───────────────────────────────
@st.cache_data(ttl=PRICE_CACHE_TTL)
def load_price(sym: str, start: dt.date, end: dt.date):
    raw = yf.download(sym, start=start, end=end+dt.timedelta(days=1),
                      progress=False, group_by="ticker")
    if raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(sym, level=0, axis=1).rename(columns=str.strip)
    df = raw.copy()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"]   = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs    = delta.clip(lower=0).rolling(14).mean() / (
            -delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]    = 100 - 100/(1+rs)
    std          = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]=df["SMA_20"] + 2*std
    df["BB_Lower"]=df["SMA_20"] - 2*std
    return df

today = dt.date.today()
delta_days = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start = dt.date(today.year,1,1) if tf=="YTD" else today - dt.timedelta(days=delta_days)
price = load_price(tkr, start, today)
if price is None: st.error("Price data error"); st.stop()
last = price.iloc[-1]

# fundamentals (fast_info→info)
@st.cache_data(ttl=86_400)
def fundamentals(sym):
    info = yf.Ticker(sym).info
    return dict(pe=info.get("trailingPE",np.nan),
                de=info.get("debtToEquity",np.nan),
                ev=info.get("enterpriseToEbitda",np.nan))
fund = fundamentals(tkr)

# ─── 3 · blended score ─────────────────────────────────────────────
tech = 0.0
if show_sma: tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd: tech += 1 if last["MACD"]>0 else -1
if show_rsi: tech += 1 if 40<last["RSI"]<70 else -1
if show_bb:
    tech += 0.5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech -= 0.5 if last["Adj Close"]<last["BB_Lower"] else 0
if show_pe and not np.isnan(fund["pe"]): tech += 1   if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += 0.5 if fund["de"]<1  else -0.5
if show_ev and not np.isnan(fund["ev"]): tech += 1   if fund["ev"]<12 else -1

blend = tech_w/100*tech + sent_w/100*sent_val
ver,color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── 4 · UI tabs ───────────────────────────────────────────────────
tab_v, tab_ta, tab_f, tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

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
    fig.add_trace(go.Scatter(x=df.index,y=df["Adj Close"],name="Price",line=dict(color="#0ff")))
    if show_sma: fig.add_trace(go.Scatter(x=df.index,y=df["SMA_20"],name="SMA-20",line=dict(color="#ff0",dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="Upper BB",line=dict(color="#0f0",dash="dot")))
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="Lower BB",line=dict(color="#0f0",dash="dot")))
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
    st.header("Latest Reddit Mentions")
    if not df_posts.empty:
        st.dataframe(df_posts, hide_index=True, use_container_width=True)
    else:
        st.info("No recent posts for this ticker.")
