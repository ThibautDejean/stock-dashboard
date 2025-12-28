import os, sys
import subprocess
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
import json
import altair as alt
import html as htmllib
import ast
import pandas as pd
from playwright.async_api import async_playwright
import asyncio
from concurrent.futures import ThreadPoolExecutor


@st.cache_resource(show_spinner=False)
def ensure_playwright_installed():
    proc = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"playwright install failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return True

ensure_playwright_installed()

st.set_page_config(page_title="Dashboard ISIN", layout="wide")

OPENFIGI_API_KEY = st.secrets.get("OPENFIGI_API_KEY", "")

PERIODS = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]
client = "Delphine & Guillaume"

@dataclass
class MappedInstrument:
    isin: str
    name: Optional[str] = None
    ticker: Optional[str] = None
    exch_code: Optional[str] = None
    currency: Optional[str] = None


def _openfigi_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        h["X-OPENFIGI-APIKEY"] = OPENFIGI_API_KEY
    return h


@st.cache_data(ttl=6 * 60 * 60)
def map_isins_openfigi(isins: Tuple[str, ...]) -> List[MappedInstrument]:
    if not isins:
        return []

    url = "https://api.openfigi.com/v3/mapping"
    payload = [{"idType": "ID_ISIN", "idValue": i} for i in isins]

    r = requests.post(url, headers=_openfigi_headers(), json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    out: List[MappedInstrument] = []
    for isin, item in zip(isins, data):
        mi = MappedInstrument(isin=isin)
        rows = item.get("data") or []
        if rows:
            best = rows[0]
            mi.name = best.get("name")
            mi.ticker = best.get("ticker")
            mi.exch_code = best.get("exchCode")
            mi.currency = best.get("currency")
        out.append(mi)
    return out


def pick_yahoo_symbol(mi: MappedInstrument, client: str) -> Optional[str]:
    with open('isin_mapping.json') as f : 
        mapper = json.load(f)
    if mi.isin in mapper[client].keys(): 
        return mapper[client][mi.isin]
    else : 
        return mi.ticker

import pandas as pd
from typing import Union, Optional, Dict

def get_close_series(
    hist: Union[pd.DataFrame, Dict[str, pd.DataFrame], None],
    yahoo_symbol: str,
    timeframe: str = "1D",
) -> pd.Series:
    if hist is None:
        return pd.Series(dtype="float64")

    if isinstance(hist, dict):
        tf = timeframe.upper()
        if tf in ["1D", "5D"]:
            df = hist.get("new")
        elif tf in ["1M"] : 
            df = hist.get("mid_up")
        elif tf in ["6M", "1Y", "YTD"] : 
            df = hist.get("mid")
        else:
            df = hist.get("old")

        if df is None:
            return pd.Series(dtype="float64")
        hist = df

    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return pd.Series(dtype="float64")

    if isinstance(hist.columns, pd.MultiIndex):
        if ("Close", yahoo_symbol) in hist.columns:
            return pd.to_numeric(hist[("Close", yahoo_symbol)], errors="coerce")

        if (yahoo_symbol, "Close") in hist.columns:
            return pd.to_numeric(hist[(yahoo_symbol, "Close")], errors="coerce")

        close_cols = [c for c in hist.columns if str(c[0]).lower() == "close" or str(c[1]).lower() == "close"]
        if close_cols:
            return pd.to_numeric(hist[close_cols[0]], errors="coerce")

        return pd.Series(dtype="float64")

    if "Close" in hist.columns:
        return pd.to_numeric(hist["Close"], errors="coerce")

    return pd.Series(dtype="float64")


_executor = ThreadPoolExecutor(max_workers=1)

def run_async_in_thread(coro):
    def _job():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _executor.submit(_job).result()

async def fetch_history_df(ticker: str) -> pd.DataFrame:
    if ticker == "FR001400G321" : 
        structure_url = "https://quantalys.com/Structure/2087537"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(structure_url, wait_until="networkidle")

        div_html = await page.inner_html("#chartHistoriqueY5")
        await browser.close()

    decoded = htmllib.unescape(div_html)

    idx_start = decoded.find("dataProvider")
    idx_end = decoded.find("decimalSeparator")
    if idx_start == -1 or idx_end == -1 or idx_end <= idx_start:
        raise RuntimeError("dataProvider/decimalSeparator introuvable dans le div.")

    extracted_string = decoded[idx_start + len("dataProvider") + 2 : idx_end - 2]
    x = ast.literal_eval(extracted_string)

    rows = [{"Date": pd.to_datetime(i["x"]), "Close": float(i["y_0"])} for i in x]
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df.columns = pd.MultiIndex.from_product([["Close"], [ticker]], names=["Price", "Ticker"])
    return df

def fetch_history_df_sync(ticker: str) -> pd.DataFrame:
    return run_async_in_thread(fetch_history_df(ticker))

def get_df_cached(ticker: str) -> pd.DataFrame:
    return fetch_history_df_sync(ticker)

@st.cache_data(ttl=60 * 60)
def load_history(yahoo_symbol: str, years: int = 60, period: str = "mid") -> pd.DataFrame:
    if yahoo_symbol in ["FR001400G321"] : 
        df = get_df_cached(yahoo_symbol)
        return df
    
    if not isinstance(yahoo_symbol, str) or not yahoo_symbol.strip():
        return pd.DataFrame()
    
    if period == "old" :
        start = (dt.date.today() - relativedelta(years=60)).isoformat()
        df = yf.download(yahoo_symbol.strip(), start=start, progress=False, auto_adjust=True, interval="1d")
        if df is None or df.empty:
            return pd.DataFrame()

        # df.index = pd.to_datetime(df.index)
        df.index = pd.to_datetime(df.index).normalize() + pd.Timedelta(hours=8)
    
    if period == "mid" :
        start = (dt.date.today() - relativedelta(days=700)).isoformat()
        df = yf.download(yahoo_symbol.strip(), start=start, progress=False, auto_adjust=True, interval="60m")
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
    
    if period == "mid_up" :
        start = (dt.date.today() - relativedelta(days=55)).isoformat()
        df = yf.download(yahoo_symbol.strip(), start=start, progress=False, auto_adjust=True, interval="15m")
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
    if period == "new" :
        start = (dt.date.today() - relativedelta(days=15)).isoformat()
        df = yf.download(yahoo_symbol.strip(), start=start, progress=False, auto_adjust=True, interval="2m")
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
    
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns=str.title)

    return df

def get_first_timepoint(close, label):
    idx_last = close.index[-1]
    if label == "1D":
        target = idx_last - relativedelta(days=1)
    if label == "5D":
       target = idx_last - relativedelta(days=5)
    elif label == "1M":
        target = idx_last - relativedelta(months=1)
    elif label == "6M":
        target = idx_last - relativedelta(months=6)
    elif label == "1Y":
        target = idx_last - relativedelta(years=1)
    elif label == "5Y":
        target = idx_last - relativedelta(years=5)
    elif label == "YTD":
        target = dt.datetime(idx_last.year, 1, 1)
    elif label == "MAX":
        target = idx_last - relativedelta(years=60)
    return target

@st.cache_data(ttl=60 * 60)
def load_info(yahoo_symbol: str) -> Dict:
    t = yf.Ticker(yahoo_symbol)
    return getattr(t, "info", {}) or {}

def pct(a: float) -> float:
    return float(a) * 100.0

def perf_from_close(hist, yahoo, label: str) -> Optional[float]:
    close = get_close_series(hist, yahoo, label)

    if close is None or close.dropna().shape[0] < 2:
        return None
        
    close = close.dropna()
    last = close.iloc[-1]
    idx_last = close.index[-1]

    if label == "1D":
        target = idx_last - relativedelta(days=1)
    elif label == "5D":
        target = idx_last - relativedelta(days=5)
    elif label == "1M":
        target = idx_last - relativedelta(months=1)
    elif label == "6M":
        target = idx_last - relativedelta(months=6)
    elif label == "1Y":
        target = idx_last - relativedelta(years=1)
    elif label == "5Y":
        target = idx_last - relativedelta(years=5)
    elif label == "YTD":
        target = dt.datetime(idx_last.year, 1, 1)
    elif label == "MAX":
        first = close.iloc[0]
        return pct(last / first - 1.0)
    else:
        return None

    try:
        pos = close.index.get_indexer([target], method="ffill")[0]
    except Exception:
        return None

    if pos == -1:
        return None

    ref = close.iloc[pos]
    return pct(last / ref - 1.0)


def daily_change_pct(hist, yahoo, label) -> Optional[float]:
    close = get_close_series(hist, yahoo, label)

    if close is None or close.dropna().shape[0] < 2:
        return None
    c = close.dropna()
    today_values = c[c.index.date == c.index[-1].date()]
    if len(today_values) >= 2:
        first = today_values.iloc[0]
        last = today_values.iloc[-1]
        return pct(last / first - 1.0)
    if len(c) >= 2:
        return pct(c.iloc[-1] / c.iloc[-2] - 1.0)

    return None

def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}%".replace(",", " ")

def fmt_int(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(x):,}".replace(",", " ")


st.title("Suivi des indices boursiers")

CLIENT_CHOICES = ["Delphine & Guillaume", "Delphine", "Guillaume"]
client = st.radio("Compte", CLIENT_CHOICES, horizontal=True)

with st.sidebar:
    st.header("Ajout")
    raw = st.text_area("Ajoutez un code ISIN (1 par ligne)", height=220, placeholder="Ex:\nIE00B4L5Y983\nFR0014002B31\n...")
    years = 60

with open('isin_mapping.json') as f : 
    global_mapper = json.load(f)

client_mapper = global_mapper[client]
base_isins = list(client_mapper.keys())
additional_isins = list([x.strip().upper() for x in raw.splitlines() if x.strip()])
isins = sorted(set(base_isins + additional_isins))

if not isins:
    st.info("Ajoutez un code ISIN.")
    st.stop()

try:
    mapped = map_isins_openfigi(tuple(isins))
except Exception as e:
    st.error(f"Erreur OpenFIGI : {e}")
    st.stop()

rows = []
charts: Dict[str, pd.DataFrame] = {}
isins_codes: Dict[str, str] = {}
names_to_isin = {}

for k, mi in enumerate(mapped):
    yahoo = pick_yahoo_symbol(mi, client)

    if not yahoo:
        rows.append({"ISIN": mi.isin, "Nom": mi.name or "—", "Ticker": "—", "Cap.": "—", "Var. jour": "—", **{p: "—" for p in PERIODS}})
        continue
    if yahoo == mi.isin : 
        old_hist = mid_hist = mid_up_hist = new_hist = load_history(yahoo)
    else : 
        old_hist = load_history(yahoo, period = "old")
        mid_hist = load_history(yahoo, period = "mid")
        mid_up_hist = load_history(yahoo, period = "mid_up")
        new_hist = load_history(yahoo, period = "new")

    hist = {
        "old": old_hist,
        "mid": mid_hist,
        "mid_up" : mid_up_hist,
        "new": new_hist,
        }
    if hist["new"].empty or "Close" not in hist["new"].columns:
        rows.append({"ISIN": mi.isin, "Nom": mi.name or "—", "Ticker": yahoo, "Cap.": "—", "Var. jour": "—", **{p: "—" for p in PERIODS}})
        continue

    try : 
        if yahoo == mi.isin : 
            info = {}
        else : 
            info = load_info(yahoo)
    except : 
        info = {}

    perf = {p: fmt_pct(perf_from_close(hist, yahoo, p)) for p in PERIODS}
    varj = fmt_pct(daily_change_pct(hist, yahoo, "1D"))

    rows.append({
        "ISIN": mi.isin,
        "Nom": mi.name or info.get("shortName") or info.get("longName") or mi.isin,
        "Ticker": yahoo,
        "Var. jour": varj,
        **perf,
    })

    charts[mi.isin] = hist
    isins_codes[mi.isin] = mi.name
    names_to_isin[yahoo] = mi.isin

df = pd.DataFrame(rows)

PERIOD_CHOICES = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]
period_chart = st.radio("Timeframe", PERIOD_CHOICES, horizontal=True, key=mi.isin, index=3)

st.subheader("Performances")
st.dataframe(df, width='stretch', hide_index=True)

st.subheader("Détail")
name_options = sorted(set(isins_codes.values()))
selected_names = st.multiselect(
    "Choisir un ou plusieurs indices",
    options=name_options,
    default=name_options[:1],
)

name_to_isins = {}
for isin, nm in isins_codes.items():
    name_to_isins.setdefault(nm, []).append(isin)

selected_isins = []
for nm in selected_names:
    selected_isins.extend(name_to_isins.get(nm, []))
selected_isins = list(dict.fromkeys(selected_isins))

if selected_isins :#in isins_codes.values() :
    # selected = names_to_isin[selected_name]
    if period_chart in ["1D", "5D"] : 
        key = "new"
    elif period_chart == "1M":
        key = "mid_up"
    elif period_chart in ["6M", "YTD", "1Y"] :
        key = "mid"
    else : 
        key = "old"

    rebased = st.checkbox("Afficher en base 100", value=True)

    long_frames = []

    for isin in selected_isins:
        if isin not in charts:
            continue

        s = charts[isin][key].get("Close")
        if s is None or s.empty:
            continue

        old_timepoint = get_first_timepoint(s, period_chart)
        plot_s = s[s.index > old_timepoint]

        # while plot_s.shape[0] <= 1 and period_chart == "1D":
        #     old_timepoint -= relativedelta(days=1)
        #     plot_s = s[s.index > old_timepoint]

        plot_s = plot_s.dropna()
        if plot_s.shape[0] < 2:
            continue

        df_one = plot_s.reset_index()
        date_col = df_one.columns[0]
        val_col = df_one.columns[1]
        df_one = df_one.rename(columns={val_col: "Value"})
        df_one[date_col] = pd.to_datetime(df_one[date_col])

        if rebased:
            base = float(df_one["Value"].iloc[0])
            if base != 0:
                df_one["Value"] = df_one["Value"] / base * 100.0

        df_one["Name"] = isins_codes.get(isin, isin)
        df_one = df_one.rename(columns={date_col: "Date"})

        long_frames.append(df_one[["Date", "Name", "Value"]])

    if not long_frames:
        st.warning("Pas assez de données pour afficher un graphique.")

    else:
        df_long = pd.concat(long_frames, ignore_index=True)
        y_title = "Base 100" if rebased else "Value"

        values_list = np.asarray(df_long["Value"]).astype(float)
        y_min, y_max = min(values_list), max(values_list)
        padding = (y_max - y_min) * 0.05
        y_domain = [y_min - padding, y_max + padding]

        chart = (
            alt.Chart(df_long)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(domain=[float(y_domain[0]), float(y_domain[1])])),
                color=alt.Color("Name:N", title="Instrument"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Name:N", title="Instrument"),
                    alt.Tooltip("Value:Q", title=y_title, format=",.2f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(chart, width='stretch')
else:
    st.info("Sélectionnez au moins un indice.")
    