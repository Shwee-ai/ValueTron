# ═══════════════════════════════════════════════════════════════════
#  Quant-Sentiment Dashboard  —  Reddit + CSV silent fallback
# ═══════════════════════════════════════════════════════════════════
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── constants ─────────────────────────────────────────────────────
TICKERS = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
SUBS    = ["stocks","investing","wallstreetbets"]
UA      = {"User-Agent": "Mozilla/5.0 (StockDashBot/0.2)"}

PRICE_TTL       = 900          # 15 min
COLLECT_EVERY   = 3*3600       # 3 h
POST_LIMIT      = 40
POSTS_CSV_LIVE  = "reddit_posts_scored.csv"
SENTS_CSV_LIVE  = "reddit_sentiments.csv"
POSTS_CSV_FBK   = "reddit_posts.csv"        # ← your uploaded files
SENTS_CSV_FBK   = "reddit_sentiments.csv"

# ─── page setup ────────────────────────────────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
st.markdown("<h1 style='text-align:center'>⚡️ Quant Sentiment Dashboard</h1>",
            unsafe_allow_html=True)
st_autorefresh(interval=1_800_000, key="auto")     # 30 min

# ─── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], 1)
    tkr = st.selectbox("Ticker", TICKERS, 0)
    tech_w = st.slider("Technical %", 0, 100, 60)
    sent_w = 100 - tech_w
    show_sma = st.checkbox("SMA-20", True); show_macd = st.checkbox("MACD", True)
    show_rsi = st.checkbox("RSI", True);   show_bb   = st.checkbox("B-Bands", True)
    st.markdown("---")
    show_pe = st.checkbox("P/E", True); show_de = st.checkbox("D/E", True)
    show_ev = st.checkbox("EV/EBITDA", True)

# ─── 0 · collect live Reddit if needed (silent) ────────────────────
def fetch_reddit(sym: str):
    rows = []
    # --- Official JSON -------------------------------------------------
    for sub in SUBS:
        url = f"https://www.reddit.com/r/{sub}/search.json?q={sym}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1"
        try:
            r = requests.get(url, headers=UA, timeout=10)
            if r.status_code == 200:
                for c in r.json().get("data", {}).get("children", []):
                    d = c["data"]
                    rows.append({"ticker":sym,"title":d.get("title",""),
                                 "text":d.get("selftext",""),"score":d.get("score",0)})
        except Exception:
            pass
        time.sleep(0.5)
    # --- Pushshift fallback -------------------------------------------
    if rows: return rows
    base = "https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        url = f"{base}?q={sym}&subreddit={sub}&after=7d&sort=desc&size={POST_LIMIT}"
        try:
            for d in requests.get(url, timeout=10).json().get("data", []):
                rows.append({"ticker":sym,"title":d.get("title",""),
                             "text":d.get("selftext",""),"score":d.get("score",0)})
        except Exception:
            pass
        time.sleep(0.3)
    return rows

def collect_if_stale():
    if (os.path.exists(SENTS_CSV_LIVE) and
        time.time() - os.path.getmtime(SENTS_CSV_LIVE) < COLLECT_EVERY):
        return
    all_rows = []
    for sym in TICKERS:
        all_rows += fetch_reddit(sym)
    if not all_rows:      # keep old CSVs if fetch failed
        return
    df = pd.DataFrame(all_rows)
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = (df["title"].fillna("")+" "+df["text"].fillna("")
                      ).apply(lambda t:(TextBlob(t).sentiment.polarity+
                                        sia.polarity_scores(t)["compound"])/2)
    df.to_csv(POSTS_CSV_LIVE, index=False)
    df.groupby("ticker")["sentiment"].mean().reset_index(
        ).to_csv(SENTS_CSV_LIVE, index=False)

collect_if_stale()

# ─── 1 · load sentiment (live → fallback → neutral) ────────────────
def load_sentiment(sym: str):
    for path in (SENTS_CSV_LIVE, SENTS_CSV_FBK):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if sym in df["ticker"].values:
                    val = float(df.loc[df["ticker"]==sym,"sentiment"].values[0])
                    return val
            except Exception:
                pass
    return 0.0

sent_val = load_sentiment(tkr)
sent_rating = "A" if sent_val>0.20 else "C" if sent_val<-0.20 else "B"

def load_posts(sym: str):
    for path in (POSTS_CSV_LIVE, POSTS_CSV_FBK):
        if os.path.exists(path):
            try:
                p = pd.read_csv(path).query("ticker == @sym")[["title","score"]].head(20)
                if not p.empty: return p
            except Exception:
                pass
    return pd.DataFrame()

df_posts = load_posts(tkr)

# ─── 2 · price & indicators (robust) ───────────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def price(sym,start,end):
    raw = yf.download(sym,start=start,end=end+dt.timedelta(days=1),progress=False)
    if raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex):
        raw = raw.xs(sym,level=0,axis=1).rename(columns=str.strip)
    df = raw.copy()
    if "Adj Close" not in df: df["Adj Close"] = df["Close"]
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"]   = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"] = 100-100/(1+rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]=df["SMA_20"]+2*std; df["BB_Lower"]=df["SMA_20"]-2*std
    return df

today = dt.date.today()
d = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=d)
hist = price(tkr,start,today); last = hist.iloc[-1] if hist is not None else None

@st.cache_data(ttl=86_400)
def fundamentals(sym):
    inf = yf.Ticker(sym).info
    return {"pe":inf.get("trailingPE",np.nan),
            "de":inf.get("debtToEquity",np.nan),
            "ev":inf.get("enterpriseToEbitda",np.nan)}
fund = fundamentals(tkr)

# ─── 3 · blended score ─────────────────────────────────────────────
tech = 0.0
if hist is not None:
    if show_sma:  tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
    if show_macd: tech += 1 if last["MACD"]>0 else -1
    if show_rsi:  tech += 1 if 40<last["RSI"]<70 else -1
    if show_bb:
        tech += 0.5 if last["Adj Close"]>last["BB_Upper"] else 0
        tech -= 0.5 if last["Adj Close"]<last["BB_Lower"] else 0
if show_pe and not np.isnan(fund["pe"]): tech += 1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += .5 if fund["de"]<1 else -.5
if show_ev and not np.isnan(fund["ev"]): tech += 1 if fund["ev"]<12 else -1

blend = tech_w/100*tech + sent_w/100*sent_val
ver,color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── 4 · UI ─────────────────────────────────────────────────────────
t1,t2,t3,t4 = st.tabs(["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with t1:
    st.markdown(f"<h1 style='color:{color};text-align:center'>{ver}</h1>",unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech",f"{tech:.2f}"); c2.metric("Sent Rating",sent_rating)
    c3.metric("Sent Score",f"{sent_val:.2f}"); c4.metric("Blended",f"{blend:.2f}")

with t2:
    if hist is not None:
        h = hist.loc[start:today]
        fig = go.Figure(); fig.add_trace(go.Scatter(x=h.index,y=h["Adj Close"],name="Price",line=dict(color="#0ff")))
        if show_sma: fig.add_trace(go.Scatter(x=h.index,y=h["SMA_20"],name="SMA-20",line=dict(color="#ff0",dash="dash")))
        if show_bb:
            fig.add_trace(go.Scatter(x=h.index,y=h["BB_Upper"],name="Upper BB",line=dict(color="#0f0",dash="dot")))
            fig.add_trace(go.Scatter(x=h.index,y=h["BB_Lower"],name="Lower BB",line=dict(color="#0f0",dash="dot")))
        fig.update_layout(template="plotly_dark",height=350,title="Price / SMA / Bollinger")
        st.plotly_chart(fig,use_container_width=True)
        if show_macd: st.line_chart(h["MACD"],height=200)
        if show_rsi:  st.line_chart(h["RSI"],height=200)
    else:
        st.write("Price data unavailable.")

with t3:
    st.table(pd.DataFrame({"Metric":["P/E","Debt/Equity","EV/EBITDA"],
                           "Value":[fund["pe"],fund["de"],fund["ev"]]}
                          ).set_index("Metric"))

with t4:
    if not df_posts.empty:
        st.dataframe(df_posts,hide_index=True,use_container_width=True)
