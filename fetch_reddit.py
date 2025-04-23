#!/usr/bin/env python
"""
Pull recent Reddit submissions for a list of tickers
and save them into reddit_posts.csv
Pushshift = no API keys required.
"""

import requests, time, csv, datetime as dt

TICKERS = ["NVDA","AAPL","MSFT","TSLA","AMD",
           "ADBE","SCHW","DE","FANG","PLTR"]
SUBS    = "stocks,investing,wallstreetbets"
LIMIT   = 50           # per ticker
OUTFILE = "reddit_posts.csv"

rows = []
for tkr in TICKERS:
    url = (f"https://api.pushshift.io/reddit/search/submission/"
           f"?q={tkr}&subreddit={SUBS}&sort=desc&size={LIMIT}")
    try:
        data = requests.get(url, timeout=10).json().get("data", [])
    except Exception as e:
        print("Pushshift error:", e); data = []

    for d in data:
        rows.append({
            "ticker":  tkr,
            "title":   d.get("title", ""),
            "text":    d.get("selftext", ""),
            "score":   d.get("score", 0),
            "date":    dt.datetime.utcfromtimestamp(d["created_utc"]).strftime("%Y-%m-%d")
        })
    time.sleep(1)      # polite pause

# write / overwrite CSV
with open(OUTFILE, "w", newline='', encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f"✅ Saved {len(rows)} posts → {OUTFILE}")
