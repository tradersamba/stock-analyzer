from fastapi import FastAPI, HTTPException
import numpy as np
import requests
import os
import json
import re

app = FastAPI(title="Peer Valuation Engine v10.6.3 (Finnhub Industry Authority)", version="10.6.3")

# ===================================================
# ENV
# ===================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not POLYGON_API_KEY:
    raise Exception("POLYGON_API_KEY not set")

if not FINNHUB_API_KEY:
    print("⚠️ WARNING: FINNHUB_API_KEY not set")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set (LLM fallback only)")


# ===================================================
# PEER UNIVERSE
# ===================================================
INDUSTRY_PEERS = {
    "Airlines": ["DAL","AAL","LUV","UAL","ALK","JBLU","SKYW","CPA","RYAAY","ULCC","ALGT","HA"],
    "Auto Manufacturers": ["TSLA","TM","GM","F","HMC","STLA","RACE","LCID","RIVN","NIO","XPEV","LI"],
    "Semiconductors": ["NVDA","AMD","AVGO","QCOM","MRVL","INTC","TSM","TXN","ADI","NXPI","ON","MU","LRCX","AMAT","KLAC"],
    "Software - Infrastructure": ["MSFT","ORCL","ADBE","CRM","NOW","PLTR","SNOW","DDOG","PANW","CRWD","ZS","OKTA"],
    "Cybersecurity": ["CRWD","PANW","ZS","OKTA","FTNT","CYBR","QLYS","VRNS","TENB"],
    "Banks - Diversified": ["JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF","SCHW"],
    "Insurance": ["BRK-B","PGR","ALL","TRV","CB","MET","PRU","AIG","HIG","AFL","L"],
    "Payments & Fintech": ["V","MA","PYPL","SQ","AXP","COIN","FIS","GPN","SOFI","AFRM"],
    "Aerospace & Defense": ["LMT","RTX","NOC","BA","GD","HII","LHX","AXON","KTOS"],
    "Industrials": ["CAT","HON","UNP","UPS","GE","DE","MMM","ETN","EMR","PH","ITW","ROP"],
    "Retail - Defensive": ["WMT","COST","TGT","HD","LOW","KR","DG","DLTR","CVS","BJ","TJX"],
    "Consumer Discretionary": ["AMZN","TSLA","NKE","SBUX","MCD","BKNG","ETSY","EBAY","GM","F","LULU"],
    "Consumer Staples": ["PG","KO","PEP","WMT","COST","CL","KMB","GIS","MDLZ","HSY","PM"],
    "Pharmaceuticals": ["LLY","JNJ","PFE","MRK","ABBV","BMY","AMGN","NVO","AZN","SNY"],
    "Biotech": ["REGN","VRTX","BIIB","GILD","MRNA","BNTX","ALNY","SRPT"],
    "Clean Energy": ["ENPH","SEDG","FSLR","PLUG","BE","RUN","NEE"],
    "Materials": ["LIN","APD","SHW","FCX","NEM","DOW","NUE","ALB"],
    "Real Estate (REITs)": ["AMT","PLD","CCI","EQIX","PSA","O","SPG","DLR"],
    "Communication Services": ["GOOGL","META","NFLX","DIS","T","VZ","CMCSA","SPOT","SNAP","TTWO","RBLX"],
    "Consumer Electronics": ["AAPL","SONY","HPQ","DELL","NTDOY","LOGI","QCOM","AMD","INTC","MU"],
    "Information Technology Services": ["IBM","ACN","DXC","CTSH","FICO"],
    "Internet Platforms": ["GOOGL","META","AMZN","RDDT","SNAP","BIDU"],
    "Internet Commerce": ["AMZN","EBAY","ETSY","DASH","SHOP","MELI","PDD","JD","BABA","W","CHWY","BKNG","EXPE","CPNG"]
}


# ===================================================
# HELPERS
# ===================================================
def compute_pe(price, eps):
    if price is None or eps is None or eps == 0:
        return None
    return price / eps


def median(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if len(vals) < 2:
        return None
    return float(np.median(np.array(vals)))


# ===================================================
# 🔵 FINNHUB = ONLY SOURCE OF INDUSTRY
# ===================================================
def get_finnhub_industry(symbol: str):
    try:
        url = "https://finnhub.io/api/v1/stock/profile2"

        resp = requests.get(
            url,
            params={
                "symbol": symbol,
                "token": FINNHUB_API_KEY
            },
            timeout=5
        ).json()

        industry = resp.get("finnhubIndustry")  # IMPORTANT
        sector = resp.get("sector")

        # hard debug safety
        if not industry:
            industry = "Unknown"
        if not sector:
            sector = "Unknown"

        return industry, sector

    except Exception:
        return "Unknown", "Unknown"


# ===================================================
# LLM INDUSTRY NORMALIZER
# ===================================================
def map_industry_llm(raw_industry: str, sector: str):
    try:
        import openai

        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
Map this company industry into ONE of these categories:

{list(INDUSTRY_PEERS.keys())}

Raw industry: {raw_industry}
Sector: {sector}

Return ONLY JSON:
{{
  "industry": "...",
  "confidence": 0.0-1.0
}}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return json.loads(resp.choices[0].message.content.strip())

    except Exception:
        return {"industry": None, "confidence": 0.0}


# ===================================================
# INDUSTRY RESOLVER
# ===================================================
def resolve_industry(raw_industry, sector, llm_result):

    llm_industry = llm_result.get("industry")
    confidence = float(llm_result.get("confidence", 0))

    # 1. Trust LLM if strong
    if llm_industry in INDUSTRY_PEERS and confidence >= 0.6:
        return llm_industry, confidence

    # 2. Finnhub fallback (IMPORTANT FIX YOU NEED)
    raw = f"{raw_industry} {sector}".lower()

    for industry in INDUSTRY_PEERS:
        if industry.lower() in raw:
            return industry, 0.7

    return "Unknown", 0.0


# ===================================================
# FINNHUB EPS
# ===================================================
EPS_CACHE = {}

def get_eps(symbol: str):
    if symbol in EPS_CACHE:
        return EPS_CACHE[symbol]

    try:
        url = "https://finnhub.io/api/v1/stock/metric"

        resp = requests.get(
            url,
            params={
                "symbol": symbol,
                "metric": "all",
                "token": FINNHUB_API_KEY
            },
            timeout=5
        ).json()

        eps = resp.get("metric", {}).get("epsTTM")
        EPS_CACHE[symbol] = eps
        return eps

    except Exception:
        return None


# ===================================================
# SNAPSHOT (POLYGON PRICE ONLY)
# ===================================================
def get_snapshot(symbol):
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"

        resp = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=5).json()

        price = None
        if resp.get("results"):
            price = resp["results"][0].get("c")

        return {"price": price}

    except Exception:
        return {"price": None}


# ===================================================
# TICKER RESOLVER (POLYGON ONLY)
# ===================================================
def resolve_ticker(name: str):

    # already ticker input
    if name.isupper() and len(name) <= 6:
        return name

    url = "https://api.polygon.io/v3/reference/tickers"

    resp = requests.get(url, params={
        "search": name,
        "active": "true",
        "limit": 10,
        "apiKey": POLYGON_API_KEY
    }, timeout=5).json()

    results = resp.get("results", [])
    if not results:
        raise HTTPException(400, "No ticker found")

    candidates = []

    for r in results:
        symbol = r.get("ticker")
        if not symbol:
            continue

        # 🔥 HARD FILTER: reject garbage tickers early
        if len(symbol) > 6:
            continue

        if "." in symbol or "-" in symbol:
            continue

        # 🔥 CRITICAL FIX: must be a real tradable stock
        snapshot = get_snapshot(symbol)
        if snapshot["price"] is None:
            continue

        name_field = (r.get("name") or "").lower()

        score = 0

        if name.lower() == name_field:
            score += 10
        if name.lower() in name_field:
            score += 5
        if name.lower() in symbol.lower():
            score += 3

        if score > 0:
            candidates.append((symbol, score))

    if not candidates:
        raise HTTPException(400, "No valid ticker candidates")

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[0][0]

# ===================================================
# MAIN
# ===================================================
@app.get("/lookup")
def lookup(name: str):

    ticker = resolve_ticker(name)

    target = get_snapshot(ticker)

    price = target["price"]
    eps = get_eps(ticker)
    pe = compute_pe(price, eps)

    # 🔵 FINNHUB INDUSTRY IS NOW SINGLE SOURCE OF TRUTH
    fin_industry, fin_sector = get_finnhub_industry(ticker)

    llm_result = map_industry_llm(fin_industry, fin_sector)
    industry_used, confidence = resolve_industry(fin_industry, fin_sector, llm_result)

    peers = INDUSTRY_PEERS.get(industry_used, [])

    valuation_peers = []
    excluded_peers = []
    peer_pes = []

    for p in peers:
        s = get_snapshot(p)
        v = compute_pe(s["price"], get_eps(p))

        if v is not None and v > 0:
            valuation_peers.append(p)
            peer_pes.append(v)
        else:
            excluded_peers.append(p)

    peer_median = median(peer_pes)

    if pe is None:
        rating = "Unknown"
        explanation = "Insufficient data"

    elif peer_median is None:
        rating = "Unknown"
        explanation = "Insufficient peer data"

    else:
        ratio = pe / peer_median

        if ratio < 0.8:
            rating = "Undervalued"
        elif ratio > 1.2:
            rating = "Overvalued"
        else:
            rating = "Fairly Valued"

        explanation = f"PE {pe:.2f} vs peer median {peer_median:.2f}"

    return {
        "input": name,
        "ticker": ticker,
        "price": price,
        "eps": eps,
        "pe": pe,

        "industry_raw": fin_industry,
        "industry_sector": fin_sector,
        "industry_used": industry_used,
        "industry_llm_confidence": confidence,

        "peers": peers,
        "valuation_peers": valuation_peers,
        "excluded_peers": excluded_peers,

        "peer_median_pe": peer_median,

        "assessment": {
            "rating": rating,
            "explanation": explanation
        }
    }
