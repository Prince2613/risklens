import yfinance as yf
import pandas as pd


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _strip_tz(df):
    """Keep wall-clock dates so US and India bars can align."""
    if getattr(df.index, 'tz', None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def fetch_stock(ticker, period='6mo', interval='1d'):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

        if df is None or df.empty:
            return None

        df = _flatten_columns(df)

        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                return None

        df = df.dropna(subset=required)

        if len(df) < 2:
            return None

        df = df.sort_index()
        df = _strip_tz(df)
        return df

    except Exception as e:
        print(f"[fetch_stock] {ticker} error: {e}")
        return None


def fetch_multiple(tickers, period='1y'):
    data = {}
    for t in tickers:
        df = fetch_stock(t, period, interval='1d')
        if df is not None and len(df) > 30:
            close = df['Close'].copy()
            close.index = pd.to_datetime(close.index).normalize()
            data[t] = close
    if not data:
        return pd.DataFrame()
    # Keep days where every ticker traded (holidays differ across markets).
    return pd.DataFrame(data).dropna(how='any')
