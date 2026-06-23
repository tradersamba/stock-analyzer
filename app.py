from fastapi import FastAPI, HTTPException
import numpy as np
import requests
import os

app = FastAPI(title="Peer Valuation Engine v10.2 (Robust PE)", version="10.2")

# ===================================================
# ENV
# ===================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not POLYGON_API_KEY:
    raise Exception("POLYGON_API_KEY not set")

if not FINNHUB_API_KEY:
    print("⚠️ WARNING: FINNHUB_API_KEY not set")


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
# 🔥 NEW: OUTLIER FILTER (TRIMMED MEDIAN)
# ===================================================
def trim_outliers(values, trim_ratio=0.1):
    """
    Removes top and bottom X% of PE values.
    Default = 10% trim.
    """
    if len(values) < 5:
        return values

    values = sorted(values)

    trim_count = int(len(values) * trim_ratio)

    if trim_count == 0:
        return values

    return values[trim_count: len(values) - trim_count]


# ===================================================
# EPS (FINNHUB)
# ===================================================
def get_eps(symbol: str):
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

        return resp.get("metric", {}).get("epsTTM")

    except Exception as e:
        print("EPS ERROR:", e)
        return None


# ===================================================
# SNAPSHOT
# ===================================================
def get_snapshot(symbol):
    try:
        price_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        price_resp = requests.get(price_url, params={"apiKey": POLYGON_API_KEY}, timeout=5).json()

        price = None
        if price_resp.get("results"):
            price = price_resp["results"][0].get("c")

        return {
            "price": price,
            "eps": get_eps(symbol),
            "industry": "Unknown",
            "exchange_code": None
        }

    except Exception:
        return {"price": None, "eps": None, "industry": "Unknown", "exchange_code": None}


# ===================================================
# MAIN
# ===================================================
@app.get("/lookup")
def lookup(name: str):

    ticker = name.upper() if name.isupper() and len(name) <= 6 else name.upper()

    cache = {}

    def snap(sym):
        if sym not in cache:
            cache[sym] = get_snapshot(sym)
        return cache[sym]

    target = snap(ticker)

    pe = compute_pe(target["price"], target["eps"])

    raw_industry = "Semiconductors"  # simplified for now
    peers = INDUSTRY_PEERS.get(raw_industry, [])

    valuation_peers = []
    excluded_peers = []
    peer_pes = []

    for p in peers:
        s = snap(p)
        v = compute_pe(s["price"], s["eps"])

        if v is not None and v > 0:
            valuation_peers.append(p)
            peer_pes.append(v)
        else:
            excluded_peers.append(p)

    # ===================================================
    # 🔥 OUTLIER CLEANING STEP
    # ===================================================
    peer_pes = trim_outliers(peer_pes, 0.1)

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

        explanation = f"PE {pe:.2f} vs trimmed peer median {peer_median:.2f}"

    return {
        "input": name,
        "ticker": ticker,
        "price": target["price"],
        "eps": target["eps"],
        "pe": pe,
        "industry": raw_industry,
        "peers": peers,
        "valuation_peers": valuation_peers,
        "excluded_peers": excluded_peers,
        "peer_median_pe": peer_median,
        "assessment": {
            "rating": rating,
            "explanation": explanation
        }
    }
