# ═══════════════════════════════════════════════════════════════════
#  Quant-Sentiment Dashboard
#  • live Reddit  →  Pushshift fallback  →  local CSV fallback
#  • NO UI notifications during any fallback
# ═══════════════════════════════════════════════════════════════════
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, os, time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

# ─── core settings ─────────────────────────────────────────────────
TICKERS = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
SUBS    = ["stocks","investing","wallstreetbets"]
UA      = {"User-Agent": "Mozilla/5.0 (StockDashBot/0.2)"}

REFRESH_MS      = 1_800_000   # dashboard auto-refresh 30 min
PRICE_CACHE_TTL = 900         # yfinance cache
COLLECT_EVERY   = 3*3600      # rediscover Reddit every 3 h
POST_LIMIT      = 40          # per subreddit

POSTS_CSV = "reddit_posts.csv"          # your uploaded files
SENTS_CSV = "reddit_sentiments.csv"

# ─── page config ──────────────────────────────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
st.markdown("<h1 style='text-align:center'>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)
st_autorefresh(interval=REFRESH_MS, key="auto")

# ─── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], 1)
    tkr = st.selectbox("Ticker", TICKERS, 0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60); sent_w = 100-tech_w
    show_sma  = st.checkbox("SMA-20", True); show_macd = st.checkbox("MACD", True)
    show_rsi  = st.checkbox("RSI", True);   show_bb   = st.checkbox("Bollinger", True)
    st.markdown("---")
    show_pe = st.checkbox("P/E", True); show_de = st.checkbox("Debt/Equity", True); show_ev = st.checkbox("EV/EBITDA", True)

# ─── 0 • fetch + score if needed (no UI notices) ──────────────────
def fetch_reddit_posts(ticker):
    rows = []
    for sub in SUBS:
        url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1"
        try:
            r = requests.get(url, headers=UA, timeout=10)
            if r.status_code == 200:
                for c in r.json().get("data", {}).get("children", []):
                    d = c["data"]; rows.append({"ticker":ticker,
                        "title":d.get("title",""), "text":d.get("selftext",""),
                        "score":d.get("score",0)})
        except: pass
        time.sleep(0.5)
    if rows: return rows                                       # success

    # Pushshift fallback
    base = "https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        try:
            ps = requests.get(base, params={
                "q": ticker, "subreddit": sub, "after":"7d",
                "size": POST_LIMIT, "sort":"desc"}, timeout=10).json()
            for d in ps.get("data", []):
                rows.append({"ticker":ticker,
                    "title":d.get("title",""), "text":d.get("selftext",""),
                    "score":d.get("score",0)})
        except: pass
        time.sleep(0.3)
    return rows

def collect_if_stale():
    if os.path.exists(SENTS_CSV) and time.time() - os.path.getmtime(SENTS_CSV) < COLLECT_EVERY:
        return                                              # recent enough
    all_rows = [row for sym in TICKERS for row in fetch_reddit_posts(sym)]
    if not all_rows:                                        # leave CSVs untouched
        return
    df = pd.DataFrame(all_rows)
    sia = SentimentIntensityAnalyzer()
    def score(r): txt=f"{r['title']} {r['text']}"; return((TextBlob(txt).sentiment.polarity+sia.polarity_scores(txt)["compound"])/2)
    df["sentiment"] = df.apply(score, axis=1)
    df.to_csv(POSTS_CSV, index=False)
    df.groupby("ticker")["sentiment"].mean().round(4).reset_index().to_csv(SENTS_CSV, index=False)

collect_if_stale()   # runs silently

# ─── 1 • load sentiment table (neutral if missing) ────────────────
try:   sent_val = float(pd.read_csv(SENTS_CSV).set_index("ticker").at[tkr,"sentiment"])
except: sent_val = 0.0
sent_rating = "A" if sent_val>0.20 else "C" if sent_val<-0.20 else "B"

try:   df_posts = pd.read_csv(POSTS_CSV).query("ticker==@tkr")[["title","score"]].head(20)
except: df_posts = pd.DataFrame()

# ─── 2 • price + TA (robust) ───────────────────────────────────────
@st.cache_data(ttl=PRICE_CACHE_TTL)
def price_df(sym, start, end):
    raw = yf.download(sym, start=start, end=end+dt.timedelta(days=1), progress=False)
    if raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(sym, level=0, axis=1).rename(columns=str.strip)
    df = raw.copy()
    if "Adj Close" not in df: df["Adj Close"] = df["Close"]
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"]   = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    rs = df["Adj Close"].diff().clip(lower=0).rolling(14).mean() / \
         (-df["Adj Close"].diff().clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2*std; df["BB_Lower"] = df["SMA_20"] - 2*std
    return df

today = dt.date.today()
start = dt.date(today.year,1,1) if tf=="YTD" else today - dt.timedelta(days={"1W":7,"1M":30,"6M":180,"1Y":365}[tf])
price = price_df(tkr, start, today)
if price is None: st.error("Price data error"); st.stop()
last = price.iloc[-1]

@st.cache_data(ttl=86_400)
def fundamentals(sym):
    info = yf.Ticker(sym).info
    return {"pe":info.get("trailingPE",np.nan),
            "de":info.get("debtToEquity",np.nan),
            "ev":info.get("enterpriseToEbitda",np.nan)}
fund = fundamentals(tkr)

# ─── 3 • blended score ─────────────────────────────────────────────
tech = 0.0
if show_sma:  tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd: tech += 1 if last["MACD"]>0 else -1
if show_rsi:  tech += 1 if 40<last["RSI"]<70 else -1
if show_bb:
    tech += .5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech -= .5 if last["Adj Close"]<last["BB_Lower"] else 0
if show_pe and not np.isnan(fund["pe"]): tech += 1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += .5 if fund["de"]<1 else -.5
if show_ev and not np.isnan(fund["ev"]): tech += 1 if fund["ev"]<12 else -1

blend = tech_w/100*tech + sent_w/100*sent_val
ver,color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── 4 • UI ────────────────────────────────────────────────────────
tab_v, tab_ta, tab_f, tab_r = st.tabs(["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with tab_v:
    st.markdown(f"<h2 style='color:{color};text-align:center'>{ver}</h2>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",f"{tech:.2f}")
    c2.metric("Sent Rating",sent_rating)
    c3.metric("Sent Score",f"{sent_val:.2f}")
    c4.metric("Blended",f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech  +  {sent_w}% Sentiment")

with tab_ta:
    df = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Adj Close"],name="Price"))
    if show_sma: fig.add_trace(go.Scatter(x=df.index,y=df["SMA_20"],name="SMA-20",line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="Upper BB",line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="Lower BB",line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark",height=340)
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(df["MACD"],height=180)
    if show_rsi:  st.line_chart(df["RSI"],height=180)

with tab_f:
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt/Equity","EV/EBITDA"],
        "Value":[fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    if not df_posts.empty:
        st.dataframe(df_posts,hide_index=True,use_container_width=True)
