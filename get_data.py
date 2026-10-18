"""
QF600 group project - data download

Downloads:
  1. Top 100 S&P 500 stocks by market cap
  2. Daily adjusted prices (total return, dividends reinvested) and volume, 2016-2025
  3. Daily Fama-French 5 factors, momentum, and short-term reversal (Ken French library)

Outputs go to ./data/

Requirements:
  pip install yfinance pandas lxml requests
"""

import io
import os
import time
import zipfile

import pandas as pd
import requests
import yfinance as yf


START = "2016-01-01"
END = "2026-01-01"          # yfinance end date is exclusive, so this includes 2025-12-31
N_STOCKS = 100
OUT_DIR = "data"

# Drop the second share class when a company has two in the index.
# GOOG and GOOGL are the same company, so they would always be the "best" pair.
DROP_TICKERS = ["GOOG"]

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
    "rev": "F-F_ST_Reversal_Factor_daily_CSV.zip",
}


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]

    tickers = []
    sectors = {}
    for i in range(len(table)):
        symbol = table.loc[i, "Symbol"]
        symbol = symbol.replace(".", "-")      # BRK.B -> BRK-B for Yahoo
        tickers.append(symbol)
        sectors[symbol] = table.loc[i, "GICS Sector"]
    return tickers, sectors


def get_market_caps(tickers):
    cache_path = os.path.join(OUT_DIR, "market_caps_cache.csv")

    # reuse caps from an earlier run so we don't start from zero
    caps = {}
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        for i in range(len(cached)):
            caps[cached.loc[i, "ticker"]] = cached.loc[i, "market_cap"]
        print("  loaded", len(caps), "market caps from cache")

    for i in range(len(tickers)):
        t = tickers[i]
        if t in caps:
            continue

        try:
            cap = yf.Ticker(t).fast_info["market_cap"]
            caps[t] = cap
            print("  ", i + 1, "/", len(tickers), t, cap)
        except Exception as e:
            print("  ", i + 1, "/", len(tickers), t, "failed -", e)

        # save every 25 tickers in case the run gets interrupted
        if (i + 1) % 25 == 0:
            save_caps(caps, cache_path)
            time.sleep(1)

    save_caps(caps, cache_path)
    return caps


def save_caps(caps, path):
    rows = []
    for t in caps:
        rows.append({"ticker": t, "market_cap": caps[t]})
    pd.DataFrame(rows).to_csv(path, index=False)


def pick_top_stocks(caps, n):
    ranked = []
    for t in caps:
        if t in DROP_TICKERS:
            continue
        if caps[t] is None:
            continue
        ranked.append((t, caps[t]))

    # sort by market cap, largest first
    ranked.sort(key=lambda pair: pair[1], reverse=True)

    top = []
    for i in range(min(n, len(ranked))):
        top.append(ranked[i])
    return top


def download_prices(tickers):
    # auto_adjust=False keeps the raw OHLC plus a separate "Adj Close".
    # Adj Close is adjusted for splits AND dividends, which is the
    # total return index the paper uses.
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=False,
        progress=True,
        group_by="column",
    )

    # long format: one row per (date, ticker) with full OHLCV
    rows = []
    for t in tickers:
        for date in raw.index:
            close = raw["Close"][t][date]
            if pd.isna(close):
                continue
            rows.append({
                "date": date,
                "ticker": t,
                "open": raw["Open"][t][date],
                "high": raw["High"][t][date],
                "low": raw["Low"][t][date],
                "close": close,
                "adj_close": raw["Adj Close"][t][date],
                "volume": raw["Volume"][t][date],
            })
    ohlcv = pd.DataFrame(rows)

    prices = raw["Adj Close"]
    volume = raw["Volume"]
    return ohlcv, prices, volume


def read_french_zip(filename):
    url = FRENCH_BASE + filename
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = zf.namelist()[0]
    text = zf.read(inner_name).decode("latin-1")
    lines = text.splitlines()

    # Data rows start with an 8-digit date (YYYYMMDD).
    # The header is the line just above the first data row.
    header = None
    rows = []
    for i in range(len(lines)):
        parts = lines[i].split(",")
        first = parts[0].strip()
        if len(first) == 8 and first.isdigit():
            if header is None:
                header_parts = lines[i - 1].split(",")
                header = []
                for h in header_parts[1:]:
                    header.append(h.strip())
            values = []
            for v in parts[1:]:
                values.append(float(v))
            rows.append([first] + values)
        elif header is not None and len(rows) > 0 and first == "":
            # blank line after the data block, stop here
            break

    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")

    # French data is in percent, convert to decimals
    df = df / 100.0
    return df


def download_factors():
    frames = []
    for key in FRENCH_FILES:
        print("  downloading", FRENCH_FILES[key])
        df = read_french_zip(FRENCH_FILES[key])
        frames.append(df)

    factors = frames[0]
    for df in frames[1:]:
        factors = factors.join(df, how="left")

    # tidy up column names
    rename = {}
    for col in factors.columns:
        if col.lower().startswith("mom"):
            rename[col] = "MOM"
        if col.lower().startswith("st_rev") or col.lower().startswith("st rev"):
            rename[col] = "REV"
    factors = factors.rename(columns=rename)

    factors = factors.loc[START:END]
    return factors


def quality_report(prices, volume):
    rows = []
    for t in prices.columns:
        p = prices[t]
        v = volume[t]
        first_valid = p.first_valid_index()
        n_missing = int(p.isna().sum())
        n_zero_vol = int((v == 0).sum())
        rows.append({
            "ticker": t,
            "first_date": first_valid,
            "missing_days": n_missing,
            "zero_volume_days": n_zero_vol,
            "full_history": first_valid is not None and str(first_valid)[:10] <= "2016-01-05",
        })
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("1. Getting S&P 500 list")
    tickers, sectors = get_sp500_tickers()
    print("  found", len(tickers), "tickers")

    print("2. Getting market caps")
    caps = get_market_caps(tickers)
    top = pick_top_stocks(caps, N_STOCKS)

    universe_rows = []
    for rank in range(len(top)):
        t = top[rank][0]
        universe_rows.append({
            "rank": rank + 1,
            "ticker": t,
            "market_cap": top[rank][1],
            "sector": sectors.get(t, ""),
        })
    universe = pd.DataFrame(universe_rows)
    universe.to_csv(os.path.join(OUT_DIR, "universe.csv"), index=False)
    print("  saved universe.csv")

    top_tickers = []
    for row in universe_rows:
        top_tickers.append(row["ticker"])

    print("3. Downloading daily prices and volume")
    ohlcv, prices, volume = download_prices(top_tickers)
    ohlcv.to_csv(os.path.join(OUT_DIR, "ohlcv_raw.csv"), index=False)
    print("  saved ohlcv_raw.csv:", ohlcv.shape)
    prices.to_csv(os.path.join(OUT_DIR, "prices_adj.csv"))
    volume.to_csv(os.path.join(OUT_DIR, "volume.csv"))
    print("  saved prices_adj.csv and volume.csv:", prices.shape)

    report = quality_report(prices, volume)
    report.to_csv(os.path.join(OUT_DIR, "data_quality.csv"), index=False)
    n_short = int((report["full_history"] == False).sum())
    print("  stocks without full 2016 history:", n_short)

    print("4. Downloading Fama-French factors")
    factors = download_factors()
    factors.to_csv(os.path.join(OUT_DIR, "factors_daily.csv"))
    print("  saved factors_daily.csv:", factors.shape)
    print("  columns:", list(factors.columns))

    print("Done.")


if __name__ == "__main__":
    main()