from fastapi import FastAPI, HTTPException
import yfinance as yf
import numpy as np

app = FastAPI(title="Peer Valuation Engine v9.5", version="9.5")


# ===================================================
# MARKET MAPPING (NEW ADDITION)
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
# TICKER RESOLVER (UNCHANGED)
# ===================================================
def resolve_ticker(name: str):
    try:
        if name.isupper() and len(name) <= 6:
            return name

        search = yf.Search(name)
        results = getattr(search, "quotes", None) or getattr(search, "results", None)

        if not results:
            raise Exception("No results")

        candidates = []

        for r in results:
            symbol = r.get("symbol")
            if not symbol:
                continue

            if len(symbol) > 6 or "." in symbol or "-" in symbol:
                continue

            try:
                t = yf.Ticker(symbol)
                info = t.info

                price = info.get("regularMarketPrice")
                eps = info.get("trailingEps") or info.get("forwardEps")
                market_cap = info.get("marketCap")

                if price is None or price <= 0:
                    continue

                if market_cap is None and eps is None:
                    continue

                score = 0
                if eps is not None:
                    score += 3
                if market_cap and market_cap > 1e9:
                    score += 3
                score += (r.get("score") or 0) * 0.1
                if name.lower() in symbol.lower():
                    score += 1

                candidates.append((symbol, score))

            except:
                continue

        if not candidates:
            raise Exception("No valid candidates")

        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0]

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot find '{name}' in US stock market listings. "
                f"It may be listed under a different name, be private, "
                f"or traded in foreign markets."
            )
        )


# ===================================================
# SNAPSHOT (ONLY CHANGE = DEBUG LINE ADDED)
# ===================================================
def get_snapshot(symbol):
    print("ENTERED get_snapshot:", symbol)

    try:
        t = yf.Ticker(symbol)
        print("CREATED TICKER:", symbol)

        info = t.info

        exchange_code = info.get("exchange")

        market_code = None
        market_name = None

        if exchange_code in MARKET_MAP:
            market_code, market_name = MARKET_MAP[exchange_code]

        return {
            "price": info.get("regularMarketPrice"),
            "eps": info.get("trailingEps") or info.get("forwardEps"),
            "industry": info.get("industry") or "Unknown",
            "sector": info.get("sector"),
            "exchange_code": exchange_code,
            "market_code": market_code,
            "market_name": market_name
        }

    except:
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

    print("🔵 ABOUT TO CALL SNAP WITH:", ticker)
    target = snap(ticker)
    print("🟢 RETURNED FROM SNAP")

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
        explanation = (
            "The PE Ratio is negative, which means the company is currently not profitable "
            "and so the metric has no real diagnostic meaning"
        )

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

        explanation = (
            f"PE ratio of {pe:.2f} is {rating.lower()} compared with peer median PE of {peer_median:.2f}"
        )

    return {
        "input": name,
        "ticker": ticker,
        "market": (
            f"{target.get('market_code') or 'UNK'} - "
            f"{target.get('market_name') or 'Unknown Exchange'}"
        ),
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