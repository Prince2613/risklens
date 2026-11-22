import numpy as np
from data_fetcher import fetch_stock

SCENARIOS = {
    "COVID-19 Crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 Rate-Hike Selloff (Jan–Oct 2022)": ("2022-01-03", "2022-10-14"),
    "2008 Financial Crisis (Oct 2007–Mar 2009)": ("2007-10-09", "2009-03-09"),
}


def compute_stress_test(tickers, weights):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    # Fetch full history once per ticker
    histories = {}
    for t in tickers:
        df = fetch_stock(t, period="max")
        if df is not None:
            histories[t] = df["Close"]

    results = {}
    for name, (start, end) in SCENARIOS.items():
        rets = {}
        excluded = []

        for t in tickers:
            if t not in histories:
                excluded.append(t)
                continue
            s = histories[t]
            window = s[(s.index >= start) & (s.index <= end)]
            if len(window) < 5:
                excluded.append(t)
                continue
            rets[t] = float(window.iloc[-1] / window.iloc[0] - 1)

        if not rets:
            results[name] = {
                "portfolio_return": None,
                "excluded_tickers": excluded,
                "note": "No historical data available for this period.",
            }
            continue

        included_idx = [i for i, t in enumerate(tickers) if t in rets]
        w_sub = weights[included_idx]
        w_sub = w_sub / w_sub.sum()

        port_ret = sum(
            w_sub[j] * rets[tickers[included_idx[j]]]
            for j in range(len(included_idx))
        )

        results[name] = {
            "portfolio_return": round(port_ret * 100, 2),
            "excluded_tickers": excluded,
            "note": (
                f"{', '.join(excluded)} excluded (no data for this period)."
                if excluded else None
            ),
        }

    return results