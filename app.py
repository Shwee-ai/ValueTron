# ─────────────────────────  ValueTron  ──────────────────────────
# single-file Streamlit dash (ticker TA + Reddit sentiment)

import base64, pathlib, os, time, datetime as dt, requests
import pandas as pd, numpy as np, yfinance as yf, streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

# ─── 1.  PAGE CONFIG MUST BE FIRST ───────────────────────────────
st.set_page_config(page_title="📈 ValueTron",
                   page_icon="⚡️",
                   layout="wide")

# ─── 2.  TRON BACKGROUND (lighter overlay 80 % ) ─────────────────
img_path = pathlib.Path("tron.png")
if img_path.exists():
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    st.markdown(f"""
    <style>
    body, .stApp {{
        background: linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.80)),
                    url("data:image/png;base64,{b64}") center/cover fixed no-repeat;
    }}
    header, footer {{visibility:hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ─── 3.  STATIC CONFIG  ──────────────────────────────────────────
TICKERS = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
FULLNAME = {
    "NVDA":"NVIDIA","AMD":"Advanced Micro Devices","ADBE":"Adobe","VRTX":"Vertex Pharma",
    "SCHW":"Charles Schwab","CROX":"Crocs","DE":"Deere & Co.","FANG":"Diamondback Energy",
    "TMUS":"T-Mobile US","PLTR":"Palantir"
}
SUBS = ["stocks","investing","wallstreetbets"]
UA   = {"User-Agent":"Mozilla/5.0 (ValueTron/1.5)"}

REFRESH_SEC = 3*3600             # re-scrape every 3 h
POST_LIMIT  = 40
POSTS_CSV   = "reddit_posts.csv"
PRICE_TTL   = 900                # yfinance cache 15 min

# ─── 4.  TITLE & AUTO-REFRESH  ───────────────────────────────────
st.markdown("<h1 style='text-align:center'>⚡️ ValueTron</h1>", unsafe_allow_html=True)
st_autorefresh(interval=30*60*1000, key="reload")          # full reload every 30 min

# ─── 5.  SIDEBAR  ────────────────────────────────────────────────
with st.sidebar:
    tf = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)

    label_list = [f"{sym} ({FULLNAME[sym]})" for sym in TICKERS]
    choice     = st.selectbox("Ticker", label_list, index=0)
    tkr        = choice.split()[0]                 # raw ticker

    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    st.markdown("### Technical Indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("### Fundamental Ratios")
    show_pe = st.checkbox("P/E ratio",     True)
    show_de = st.checkbox("Debt/Equity",   True)
    show_ev = st.checkbox("EV/EBITDA",     True)

# ─── 6.  REDDIT FETCH (background cache) ─────────────────────────
def fetch_reddit(ticker:str):
    rows=[]
    for sub in SUBS:
        url=(f"https://www.reddit.com/r/{sub}/search.json"
             f"?q={ticker}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1")
        try:
            r=requests.get(url,headers=UA,timeout=8)
            for c in r.json().get("data",{}).get("children",[]):
                d=c["data"]; rows.append({"ticker":ticker,
                    "title":d.get("title",""),"text":d.get("selftext",""),
                    "score":d.get("score",0)})
        except: pass
        time.sleep(0.3)
    return rows

def refresh_reddit():
    if os.path.exists(POSTS_CSV) and time.time()-os.path.getmtime(POSTS_CSV)<REFRESH_SEC:
        return
    all_posts=[]
    for sym in TICKERS: all_posts+=fetch_reddit(sym)
    if all_posts: pd.DataFrame(all_posts).to_csv(POSTS_CSV,index=False)

refresh_reddit()

# ─── 7.  LOAD & SCORE SENTIMENT  ─────────────────────────────────
posts_all=pd.read_csv(POSTS_CSV) if os.path.exists(POSTS_CSV) else pd.DataFrame(columns=["ticker","title","text","score"])
posts_all["text"].fillna("",inplace=True)

sia=SentimentIntensityAnalyzer()
def hybrid(txt,up):
    tb=TextBlob(txt).sentiment.polarity
    vd=sia.polarity_scores(txt)["compound"]
    return ((tb+vd)/2)*min(up,100)/100

posts_all["sentiment_score"]=posts_all.apply(lambda r: hybrid(r.title+" "+r.text, r.score), axis=1)
letter = lambda x: "A" if x>0.05 else "C" if x<-0.05 else "B"
posts_all["rating"]=posts_all["sentiment_score"].apply(letter)

posts_df = posts_all.loc[posts_all.ticker==tkr,["title","rating"]].head(20)
sub      = posts_all.loc[posts_all.ticker==tkr,"sentiment_score"]
sent_val = sub.mean() if not sub.empty else 0.0
sent_rating = letter(sent_val)

# ─── 8.  PRICE + TECH INDICATORS  ────────────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym,start,end):
    df=yf.download(sym,start=start,end=end+dt.timedelta(days=1),progress=False)
    if df.empty: return None
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(-1)
    df.columns=df.columns.str.lower().str.replace(" ","_")
    df["Adj Close"]=df.get("adj_close",df.get("close",df.select_dtypes("number").iloc[:,0]))
    df["sma_20"]=df["Adj Close"].rolling(20).mean()
    df["macd"]=df["Adj Close"].ewm(span=12).mean()-df["Adj Close"].ewm(span=26).mean()
    delta=df["Adj Close"].diff()
    rs=delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["rsi"]=100-100/(1+rs)
    std=df["Adj Close"].rolling(20).std()
    df["bb_upper"]=df["sma_20"]+2*std
    df["bb_lower"]=df["sma_20"]-2*std
    return df

today=dt.date.today()
days={"1W":7,"1M":30,"6M":180,"1Y":365}[tf]
start=dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=days)
price=load_price(tkr,start,today)
if price is None: st.error("No price data."); st.stop()
last=price.iloc[-1]

@st.cache_data(ttl=86400)
def load_fund(sym):
    info=yf.Ticker(sym).info
    return {"pe":info.get("trailingPE",np.nan),
            "de":info.get("debtToEquity",np.nan),
            "ev":info.get("enterpriseToEbitda",np.nan)}
fund=load_fund(tkr)

# scores
tech=0
if show_sma and not pd.isna(last.sma_20): tech+=1 if last["Adj Close"]>last.sma_20 else -1
if show_macd and not pd.isna(last.macd): tech+=1 if last.macd>0 else -1
if show_rsi and not pd.isna(last.rsi): tech+=1 if 40<last.rsi<70 else -1
if show_bb and not (pd.isna(last.bb_upper)|pd.isna(last.bb_lower)):
    tech+=0.5 if last["Adj Close"]>last.bb_upper else 0
    tech-=0.5 if last["Adj Close"]<last.bb_lower else 0
if show_pe and not pd.isna(fund["pe"]): tech+=1 if fund["pe"]<18 else -1
if show_de and not pd.isna(fund["de"]): tech+=0.5 if fund["de"]<1 else -0.5
if show_ev and not pd.isna(fund["ev"]): tech+=1 if fund["ev"]<12 else -1

blend=tech_w/100*tech + sent_w/100*sent_val
ver,clr=("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── 9.  TABS  ────────────────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with tab_v:
    st.markdown(f"<h2 style='color:{clr};text-align:center'>{ver}</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",  f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score",  f"{sent_val:.2f}")
    c4.metric("Blended",     f"{blend:.2f}")

    expl=[]
    expl.append("Technical indicators **bullish**" if tech>0 else
                "Technical indicators **bearish**" if tech<0 else
                "Technical indicators **neutral**")
    expl.append("Reddit sentiment **positive**" if sent_val>0.05 else
                "Reddit sentiment **negative**" if sent_val<-0.05 else
                "Reddit sentiment **neutral**")
    expl.append(f"Blend = {tech_w}%×Tech + {sent_w}%×Sent → **{blend:.2f}**")
    st.caption(" · ".join(expl))

with tab_ta:
    dfp=price.loc[start:today]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Adj Close"],name="Price"))
    if show_sma: fig.add_trace(go.Scatter(x=dfp.index,y=dfp.sma_20,name="SMA-20",line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp.bb_upper,name="Upper BB",line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp.bb_lower,name="Lower BB",line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark",height=350)
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(dfp.macd,height=180)
    if show_rsi:  st.line_chart(dfp.rsi,height=180)

with tab_f:
    rat=pd.DataFrame({"Metric":["P/E","Debt/Equity","EV/EBITDA"],
                      "Value":[fund["pe"],fund["de"],fund["ev"]]}).set_index("Metric")
    st.table(rat)

with tab_r:
    st.markdown("### Community Pulse (Reddit)")
    counts=posts_df.rating.value_counts().reindex(["A","B","C"],fill_value=0)
    cA,cB,cC=st.columns(3)
    cA.metric("A ➜ Positive", counts["A"])
    cB.metric("B ➜ Neutral",  counts["B"])
    cC.metric("C ➜ Negative", counts["C"])
    st.caption("Each post scored via TextBlob + VADER, weighted by up-votes.")
    if posts_df.empty: st.info("No Reddit posts.") 
    else:
        st.dataframe(posts_df.rename(columns={"title":"Reddit Post Title"}),
                     hide_index=True,use_container_width=True)
