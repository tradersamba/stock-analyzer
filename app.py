from fastapi import FastAPI, HTTPException
import numpy as np
import requests
import os

app = FastAPI(title="Peer Valuation Engine v10 (Polygon Fundamentals)", version="10.0")

# ===================================================
# ENV
# ===================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    raise Exception("POLYGON_API_KEY not set in environment variables")

# ===================================================
# MARKET MAPPING
# ===================================================
MARKET_MAP = {
    "NMS": ("NSDQ", "Nasdaq"),
    "NGM": ("NSDQ", "Nasdaq"),
    "NCM": ("NSDQ", "Nasdaq"),
    "NYQ": ("NYSE", "New York Stock Exchange"),
    "ASE": ("AMEX", "NYSE American"),
}

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
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    return float(np.median(np.array(vals)))

# ===================================================
# EPS (POLYGON FUNDAMENTALS)
# ===================================================
def get_eps(symbol):
    try:
        url = f"https://api.polygon.io/vX/reference/financials"

        params = {
            "ticker": symbol,
            "timeframe": "ttm",
            "limit": 1,
            "apiKey": POLYGON_API_KEY
        }

        resp = requests.get(url, params=params, timeout=5)

        if resp.status_code != 200:
            return None

        results = resp.json().get("results", [])
        if not results:
            return None

        fin = results[0].get("financials", {})

        income = fin.get("income_statement", {})
        balance = fin.get("balance_sheet", {})

        net_income = income.get("net_income")
        shares = balance.get("basic_average_shares")

        if net_income is None or shares is None or shares == 0:
            return None

        return net_income / shares

    except Exception as e:
        print("EPS ERROR:", str(e))
        return None

# ===================================================
# RESOLVE TICKER
# ===================================================
def resolve_ticker(name: str):
    try:
        if name.isupper() and len(name) <= 6:
            return name

        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "search": name,
            "active": "true",
            "limit": 10,
            "apiKey": POLYGON_API_KEY
        }

        resp = requests.get(url, params=params, timeout=5)

        if resp.status_code != 200:
            raise Exception("Polygon API error")

        results = resp.json().get("results", [])

        if not results:
            raise Exception("No results")

        candidates = []

        for r in results:
            symbol = r.get("ticker")
            if not symbol:
                continue

            if len(symbol) > 6 or "." in symbol or "-" in symbol:
                continue

            score = 0

            if name.lower() in (r.get("name") or "").lower():
                score += 3

            if name.lower() in symbol.lower():
                score += 2

            candidates.append((symbol, score))

        if not candidates:
            raise Exception("No valid candidates")

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    except Exception as e:
        print("RESOLVE ERROR:", str(e))
        raise HTTPException(status_code=400, detail=f"Cannot resolve '{name}'")

# ===================================================
# SNAPSHOT
# ===================================================
def get_snapshot(symbol):
    try:
        print("🔵 POLYGON SNAPSHOT:", symbol)

        price_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        price_resp = requests.get(price_url, params={"apiKey": POLYGON_API_KEY}, timeout=5)

        price_json = price_resp.json() if price_resp.status_code == 200 else {}

        price = None
        if price_json.get("results"):
            price = price_json["results"][0].get("c")

        details_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        details_resp = requests.get(details_url, params={"apiKey": POLYGON_API_KEY}, timeout=5)

        details_json = details_resp.json() if details_resp.status_code == 200 else {}

        details = details_json.get("results", {}) if details_json else {}

        industry = (
            details.get("sic_description")
            or details.get("description")
            or "Unknown"
        )

        return {
            "price": price,
            "eps": get_eps(symbol),
            "industry": industry,
            "sector": None,
            "exchange_code": details.get("primary_exchange"),
            "market_code": None,
            "market_name": None
        }

    except Exception as e:
        print("SNAPSHOT ERROR:", str(e))
        return {
            "price": None,
            "eps": None,
            "industry": "Unknown",
            "sector": None,
            "exchange_code": None,
            "market_code": None,
            "market_name": None
        }

# ===================================================
# MAIN ENDPOINT
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

    raw_industry = target["industry"]

    YAHOO_MAP = {
        "Internet Content & Information": "Communication Services",
        "Online Media": "Communication Services",
        "Social Media": "Communication Services",
        "Internet Retail": "Internet Commerce",
        "Internet & Direct Marketing Retail": "Internet Commerce",
        "Computer Hardware": "Consumer Electronics",
        "Information Technology Services": "Information Technology Services"
    }

    industry = YAHOO_MAP.get(raw_industry, raw_industry)
    peers = INDUSTRY_PEERS.get(industry, [])

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

    elif pe < 0:
        rating = "Not Applicable"
        explanation = "Negative PE"

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
        "market": target.get("exchange_code") or "UNK",
        "price": target["price"],
        "eps": target["eps"],
        "pe": pe,
        "industry_raw": raw_industry,
        "industry_used": industry,
        "peers": peers,
        "valuation_peers": valuation_peers,
        "excluded_peers": excluded_peers,
        "peer_median_pe": peer_median,
        "assessment": {
            "rating": rating,
            "explanation": explanation
        }
    }
