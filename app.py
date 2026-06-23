from fastapi import FastAPI, HTTPException
import numpy as np
import requests
import os

app = FastAPI(title="Peer Valuation Engine v10.2 (Hardened Resolver)", version="10.2")

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
# VALIDATION
# ===================================================
def is_valid_ticker(symbol: str):
    if not symbol:
        return False
    if len(symbol) > 6:
        return False
    if symbol.lower() == symbol:
        return False
    if " " in symbol:
        return False
    return True


# ===================================================
# SNAPSHOT (used for validation)
# ===================================================
def get_snapshot(symbol):
    try:
        price_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        resp = requests.get(price_url, params={"apiKey": POLYGON_API_KEY}, timeout=5).json()

        price = None
        if resp.get("results"):
            price = resp["results"][0].get("c")

        return {
            "price": price,
            "eps": None,
            "industry": "Unknown"
        }

    except Exception:
        return {"price": None, "eps": None, "industry": "Unknown"}


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

    except Exception:
        return None


# ===================================================
# RESOLVER (HARDENED)
# ===================================================
def resolve_ticker(name: str):
    try:
        # direct ticker input
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
            raise Exception("No results")

        candidates = []

        for r in results:
            symbol = r.get("ticker")
            name_match = (r.get("name") or "").lower()

            if not symbol:
                continue

            if len(symbol) > 6 or "." in symbol or "-" in symbol:
                continue

            score = 0

            if name.lower() == name_match:
                score += 5
            if name.lower() in name_match:
                score += 3
            if name.lower() in symbol.lower():
                score += 2
            if r.get("primary_exchange") in ["XNAS", "XNYS", "ARCX"]:
                score += 1

            candidates.append((symbol, score))

        if not candidates:
            raise Exception("No valid candidates")

        candidates.sort(key=lambda x: x[1], reverse=True)

        # ===================================================
        # 🔥 CRITICAL FIX: validate via snapshot
        # ===================================================
        for symbol, _ in candidates:
            if not is_valid_ticker(symbol):
                continue

            test = get_snapshot(symbol)

            if test["price"] is not None:
                print("🧠 RESOLVED & VALIDATED:", name, "→", symbol)
                return symbol

        raise Exception("No valid tradable ticker found")

    except Exception as e:
        print("RESOLVE ERROR:", e)
        raise HTTPException(status_code=400, detail=f"Cannot resolve '{name}'")


# ===================================================
# MAIN
# ===================================================
@app.get("/lookup")
def lookup(name: str):

    print("\n============== NEW REQUEST ================")
    print("[INPUT]:", name)

    ticker = resolve_ticker(name)
    print("[TICKER]:", ticker)

    cache = {}

    def snap(sym):
        if sym not in cache:
            cache[sym] = get_snapshot(sym)
        return cache[sym]

    target = snap(ticker)

    pe = compute_pe(target["price"], target["eps"])

    raw_industry = "Semiconductors"
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
