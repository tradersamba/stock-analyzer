from fastapi import FastAPI, HTTPException
import numpy as np
import requests
import os
import json
import re

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(
    title="Peer Valuation Engine v10.6.4 (Stable Resolver)",
    version="10.6.4"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# ===================================================
# ENV
# ===================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not POLYGON_API_KEY:
    raise Exception("POLYGON_API_KEY not set")

if not FINNHUB_API_KEY:
    print("⚠️ WARNING: FINNHUB_API_KEY not set", flush=True)

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set (LLM fallback disabled)", flush=True)


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
    if not price or not eps or eps == 0:
        return None
    return price / eps


def median(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if len(vals) < 2:
        return None
    return float(np.median(np.array(vals)))


def clean_name(name: str):
    return re.sub(r"[^a-zA-Z ]", "", name).strip()


# ===================================================
# SNAPSHOT (DEBUG)
# ===================================================
def get_snapshot(symbol):

    if symbol in SNAPSHOT_CACHE:
        print(f"[SNAPSHOT CACHE HIT] {symbol} -> {SNAPSHOT_CACHE[symbol]}", flush=True)
        return SNAPSHOT_CACHE[symbol]

    try:
        # 1. TRY PREV
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"

        resp = requests.get(
            url,
            params={"apiKey": POLYGON_API_KEY, "adjusted": "true"},
            timeout=5
        ).json()

        print(f"[SNAPSHOT PREV] {symbol} -> {resp}", flush=True)

        results = resp.get("results") or []
        if results:
            price = results[0].get("c")
            print(f"[SNAPSHOT PREV PRICE] {symbol} price={price}", flush=True)
            SNAPSHOT_CACHE[symbol] = {"price": price}
            print(f"[SNAPSHOT CACHE SAVE] {symbol} -> {price}", flush=True)
            return SNAPSHOT_CACHE[symbol]

        # 2. FALLBACK: last 1-day range
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-01-01/2025-12-31"

        resp = requests.get(
            url,
            params={
                "apiKey": POLYGON_API_KEY,
                "sort": "desc",
                "limit": 1
            },
            timeout=5
        ).json()

        print(f"[SNAPSHOT RANGE] {symbol} -> {resp}", flush=True)

        results = resp.get("results") or []
        if results:
            price = results[0].get("c")
            print(f"[SNAPSHOT RANGE PRICE] {symbol} price={price}", flush=True)
            SNAPSHOT_CACHE[symbol] = {"price": price}
            print(f"[SNAPSHOT CACHE SAVE] {symbol} -> {price}", flush=True)
            return SNAPSHOT_CACHE[symbol]

        print(f"[SNAPSHOT FAILED] {symbol} no price found", flush=True)
        return {"price": None}

    except Exception as e:
        print(f"[SNAPSHOT ERROR] {symbol} {repr(e)}", flush=True)
        return {"price": None}
        
# ===================================================
# POLYGON ACTIVE CHECK
# ===================================================
def ticker_is_active(symbol):
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"

        resp = requests.get(
            url,
            params={"apiKey": POLYGON_API_KEY},
            timeout=5
        ).json()

        result = resp.get("results")

        if not result:
            status = resp.get("status")
            message = resp.get("message", "")

            if status == "NOT_FOUND" or "Ticker not found" in message:
                print(f"[TICKER ACTIVE NOT FOUND] {symbol} -> inactive", flush=True)
                return False

            print(f"[TICKER ACTIVE UNKNOWN] {symbol} -> {resp}", flush=True)
            return True

        active = result.get("active", True)

        print(f"[TICKER ACTIVE] {symbol} active={active}", flush=True)

        return active

    except Exception as e:
        print(f"[TICKER ACTIVE ERROR] {symbol} {repr(e)}", flush=True)
        return True

# ===================================================
# STOCK EXCHANGE
# ===================================================
def get_exchange(symbol):
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"

        resp = requests.get(
            url,
            params={"apiKey": POLYGON_API_KEY},
            timeout=5
        ).json()

        result = resp.get("results", {})

        exchange_code = (
            result.get("primary_exchange")
            or result.get("market")
            or "Unknown"
        )

        exchange_map = {
            "XNAS": "NASDAQ",
            "XNYS": "NYSE",
            "ARCX": "NYSE Arca",
            "BATS": "Cboe",
            "OTCM": "OTC Markets"
        }

        exchange = exchange_map.get(exchange_code, exchange_code)

        print(f"[EXCHANGE] {symbol} -> {exchange}", flush=True)

        return exchange

    except Exception as e:
        print(f"[EXCHANGE ERROR] {symbol} {repr(e)}", flush=True)
        return "Unknown"

# ===================================================
# FINNHUB INDUSTRY
# ===================================================
def get_finnhub_industry(symbol: str):
    try:
        url = "https://finnhub.io/api/v1/stock/profile2"
        resp = requests.get(
            url,
            params={"symbol": symbol, "token": FINNHUB_API_KEY},
            timeout=5
        ).json()

        return (
            resp.get("finnhubIndustry") or "Unknown",
            resp.get("sector") or "Unknown"
        )

    except Exception:
        return "Unknown", "Unknown"


# ===================================================
# LLM INDUSTRY MAP (OPTIONAL)
# ===================================================
def map_industry_llm(raw_industry: str, sector: str):

    cache_key = f"{raw_industry}|{sector}".lower()

    if cache_key in INDUSTRY_LLM_CACHE:
        print(
            f"[INDUSTRY CACHE HIT] {cache_key} -> {INDUSTRY_LLM_CACHE[cache_key]}",
            flush=True
        )
        return INDUSTRY_LLM_CACHE[cache_key]

    print(
        f"[INDUSTRY LLM INPUT] industry='{raw_industry}' sector='{sector}'",
        flush=True
    )

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
Map this company into EXACTLY ONE of these industries:

{list(INDUSTRY_PEERS.keys())}

Raw Industry:
{raw_industry}

Sector:
{sector}

Return ONLY JSON:

{{
    "industry":"...",
    "confidence":0.0
}}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_response = resp.choices[0].message.content.strip()

        print(
            f"[INDUSTRY LLM RAW] {raw_response}",
            flush=True
        )

        cleaned = (
            raw_response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        parsed = json.loads(cleaned)

        print(
            f"[INDUSTRY LLM PARSED] {parsed}",
            flush=True
        )

        INDUSTRY_LLM_CACHE[cache_key] = parsed

        print(
            f"[INDUSTRY CACHE SAVE] {cache_key} -> {parsed}",
            flush=True
        )

        return parsed
    except Exception as e:

        print(
            f"[INDUSTRY LLM ERROR] {repr(e)}",
            flush=True
        )

        return {
            "industry": None,
            "confidence": 0.0
        }


# ===================================================
# INDUSTRY RESOLVER
# ===================================================
def resolve_industry(raw_industry, sector, llm_result):
    llm_industry = llm_result.get("industry")
    confidence = float(llm_result.get("confidence", 0))

    print(
        f"[INDUSTRY RESOLVE] "
        f"raw='{raw_industry}' "
        f"sector='{sector}' "
        f"llm='{llm_industry}' "
        f"confidence={confidence}",
        flush=True
    )

    if llm_industry in INDUSTRY_PEERS and confidence >= 0.6:
        print(
            f"[INDUSTRY ACCEPTED] {llm_industry}",
            flush=True
        )
        return llm_industry, confidence

    raw = f"{raw_industry} {sector}".lower()

    for k in INDUSTRY_PEERS:
        if k.lower() in raw:
            print(
                f"[INDUSTRY FALLBACK MATCH] {k}",
                flush=True
            )
            return k, 0.7

    print(
        "[INDUSTRY FAILED] Returning Unknown",
        flush=True
    )

    return "Unknown", 0.0


# ===================================================
# EPS
# ===================================================

EPS_CACHE = {}
FINNHUB_PRICE_CACHE = {}
SNAPSHOT_CACHE = {}

TICKER_CACHE = {}
INDUSTRY_LLM_CACHE = {}

def get_eps(symbol):

    if symbol in EPS_CACHE:
        print(f"[EPS CACHE HIT] {symbol} -> {EPS_CACHE[symbol]}", flush=True)
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

        print(f"[EPS CACHE SAVE] {symbol} -> {eps}", flush=True)

        return eps

    except Exception:
        return None

# ===================================================
# FINNHUB PRICE
# ===================================================
def get_finnhub_price(symbol):
    if symbol in FINNHUB_PRICE_CACHE:
        print(
            f"[FINNHUB PRICE CACHE HIT] {symbol} -> {FINNHUB_PRICE_CACHE[symbol]}",
            flush=True
        )
        return FINNHUB_PRICE_CACHE[symbol]

    try:
        url = "https://finnhub.io/api/v1/quote"

        resp = requests.get(
            url,
            params={
                "symbol": symbol,
                "token": FINNHUB_API_KEY
            },
            timeout=5
        ).json()

        price = resp.get("c") or resp.get("pc")

        FINNHUB_PRICE_CACHE[symbol] = price

        print(
            f"[FINNHUB PRICE CACHE SAVE] {symbol} -> {price}",
            flush=True
        )

        return price

    except Exception as e:
        print(f"[FINNHUB PRICE ERROR] {symbol} {repr(e)}", flush=True)
        return None

# ===================================================
# TICKER RESOLVER (FIXED CORE LOGIC)
# ===================================================
def llm_resolve_ticker(name: str):

    cache_key = clean_name(name).lower()

    if cache_key in TICKER_CACHE:
        print(
            f"[TICKER CACHE HIT] {cache_key} -> {TICKER_CACHE[cache_key]}",
            flush=True
        )
        return TICKER_CACHE[cache_key]

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        print(f"[LLM ENTER] {name}", flush=True)

        prompt = f"""
Return ONLY valid US stock ticker.

Company: {name}

Examples:
Nvidia -> NVDA
Intel -> INTC
Apple -> AAPL
Tesla -> TSLA
IBM -> IBM
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = resp.choices[0].message.content.strip().upper()

        print(f"[LLM RAW] {raw}", flush=True)

        # Handles answers like: IBM -> IBM
        if "->" in raw:
            raw = raw.split("->")[-1].strip()

        # Remove markdown/code formatting if any
        raw = raw.replace("`", "").strip()

        if re.fullmatch(r"[A-Z]{1,5}", raw):
            TICKER_CACHE[cache_key] = raw
            print(f"[TICKER CACHE SAVE] {cache_key} -> {raw}", flush=True)
            return raw

        return None

    except Exception as e:
        print(f"[LLM ERROR] {repr(e)}", flush=True)
        return None


def resolve_ticker(name: str):

    name_clean = clean_name(name)
    print(f"[RESOLVE] raw={name} clean={name_clean}", flush=True)

    # 1. LLM attempt (optional)
    llm_ticker = llm_resolve_ticker(name_clean)
    print(f"[LLM RESULT] {llm_ticker}", flush=True)

    # IMPORTANT FIX:
    # DO NOT reject symbol just because price is missing
    if llm_ticker:
        print(f"[LLM ACCEPTED WITHOUT PRICE CHECK] {llm_ticker}", flush=True)
        return llm_ticker

    # 2. Polygon fallback
    print("[FALLBACK] Polygon search", flush=True)

    url = "https://api.polygon.io/v3/reference/tickers"
    resp = requests.get(url, params={
        "search": name_clean,
        "active": "true",
        "limit": 10,
        "apiKey": POLYGON_API_KEY
    }, timeout=5).json()

    results = resp.get("results", [])
    print(f"[POLYGON COUNT] {len(results)}", flush=True)

    if not results:
        raise HTTPException(400, f"Ticker resolution failed for '{name}'")

    # return FIRST valid ticker (do NOT require price validation)
    for r in results:
        symbol = r.get("ticker")
        if symbol:
            print(f"[POLYGON PICK] {symbol}", flush=True)
            return symbol

    raise HTTPException(400, f"Ticker resolution failed for '{name}'")


# ===================================================
# MAIN API
# ===================================================
@app.get("/debug-files")
def debug_files():
    return {
        "base_dir": BASE_DIR,
        "files": os.listdir(BASE_DIR),
        "templates_exists": os.path.exists(os.path.join(BASE_DIR, "templates", "index.html")),
        "static_exists": os.path.exists(os.path.join(BASE_DIR, "static", "style.css")),
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/lookup")
def lookup(name: str):

    # ===================================================
    # Resolve ticker
    # ===================================================
    try:
        ticker = resolve_ticker(name)
    except HTTPException:
        return {
            "input": name,
            "ticker": None,
            "price": None,
            "eps": None,
            "pe": None,
            "industry_raw": "Unknown",
            "industry_sector": "Unknown",
            "industry_used": "Unknown",
            "industry_llm_confidence": 0.0,
            "peers": [],
            "valuation_peers": [],
            "excluded_peers": [],
            "peer_median_pe": None,
            "assessment": {
                "rating": "Unknown",
                "explanation": (
                    "Company could not be found on a U.S. stock exchange. "
                    "It may be a private company, listed only on a foreign exchange, "
                    "or the company name may be misspelled."
                )
            }
        }

    ticker_active = ticker_is_active(ticker)
    exchange = get_exchange(ticker)
    
    # ===================================================
    # Price snapshot
    # ===================================================
    price = get_finnhub_price(ticker)

    print(
        f"[MAIN PRICE FINNHUB] {ticker} -> {price}",
        flush=True
    )

    eps = get_eps(ticker)
    pe = compute_pe(price, eps)
    # ===================================================
    # Industry resolution
    # ===================================================
    fin_industry, fin_sector = get_finnhub_industry(ticker)

    llm_result = map_industry_llm(fin_industry, fin_sector)
    industry_used, confidence = resolve_industry(fin_industry, fin_sector, llm_result)

    peers = INDUSTRY_PEERS.get(industry_used, [])

    # ===================================================
    # Peer valuation
    # ===================================================
    valuation_peers = []
    excluded_peers = []
    peer_pes = []

    for p in peers:
        peer_price = get_finnhub_price(p)
        peer_eps = get_eps(p)
        v = compute_pe(peer_price, peer_eps)

        if v is not None and v > 0:
            valuation_peers.append(p)
            peer_pes.append(v)
        else:
            excluded_peers.append(p)

    peer_median = median(peer_pes)

    is_bankrupt = ticker.upper().endswith("Q")

    if is_bankrupt:

        rating = "Not Meaningful"

        explanation = (
            "This ticker appears to be in bankruptcy or bankruptcy-related trading status. "
            "The P/E ratio is not meaningful for this stock."
        )

        return {
            "input": name,
            "ticker": ticker,
            "exchange": exchange,
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
    
    if not ticker_active:

        rating = "Inactive"

        explanation = (
            "This ticker appears to be inactive or delisted. "
            "The company may have been acquired, gone private, "
            "or no longer trades under this ticker."
        )

        return {
            "input": name,
            "ticker": ticker,
            "exchange": exchange,
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
    # ===================================================
    # Rating logic (FIXED: negative EPS handling)
    # ===================================================
    if pe is None:
        rating = "Unknown"
        explanation = "Insufficient data"

    elif eps is not None and eps < 0:
        rating = "Not Meaningful"
        explanation = (
            "EPS is negative, which means the company is not currently profitable. "
            "The P/E ratio is not a meaningful valuation metric in this case."
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

        explanation = f"PE {pe:.2f} vs peer median {peer_median:.2f}"

    # ===================================================
    # Response
    # ===================================================
    return {
        "input": name,
        "ticker": ticker,
        "exchange": exchange,
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
