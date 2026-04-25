"""
run_signals.py — CoinGecko + Deribit public API
Coğrafi kısıtlama yok, auth gerektirmez.
"""

import json, time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

COINGECKO = "https://api.coingecko.com/api/v3"
DERIBIT   = "https://www.deribit.com/api/v2/public"

def cg_get(endpoint, params={}):
    r = requests.get(f"{COINGECKO}{endpoint}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def deribit_get(method, params={}):
    r = requests.get(f"{DERIBIT}/{method}", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})

# ── BTC Macro Regime ──────────────────────────────────────────
def get_btc_regime():
    try:
        data  = cg_get("/simple/price", {"ids": "bitcoin", "vs_currencies": "usd",
                                          "include_24hr_change": "true"})
        spot  = float(data["bitcoin"]["usd"])
        chg   = float(data["bitcoin"].get("usd_24h_change", 0))
        regime = "BULL_GAMMA" if chg > 0 else "BEAR_GAMMA"
        return {"regime": regime, "spot": round(spot, 0),
                "chg_24h": round(chg, 2), "alt_multiplier": 1.0 if chg > 0 else 0.3}
    except Exception as e:
        print(f"[WARN] BTC regime: {e}")
        return {"regime": "NEUTRAL", "spot": 0, "chg_24h": 0, "alt_multiplier": 0.5}

# ── Altcoin Universe ──────────────────────────────────────────
def get_universe(top_n=20):
    try:
        coins = cg_get("/coins/markets", {
            "vs_currency":    "usd",
            "order":          "volume_desc",
            "per_page":       80,
            "page":           1,
            "sparkline":      "false",
            "price_change_percentage": "24h"
        })
        alts = []
        exclude = {"bitcoin", "ethereum", "tether", "usd-coin", "binance-usd",
                   "dai", "true-usd", "usdd", "frax", "staked-ether"}
        for c in coins:
            if c["id"] in exclude:
                continue
            mcap = c.get("market_cap") or 0
            vol  = c.get("total_volume") or 0
            if not (50_000_000 <= mcap <= 10_000_000_000):
                continue
            if vol < 5_000_000:
                continue
            alts.append({
                "id":          c["id"],
                "symbol":      c["symbol"].upper(),
                "price":       c.get("current_price", 0),
                "mcap":        mcap,
                "volume_usd":  vol,
                "price_chg_24h": c.get("price_change_percentage_24h") or 0
            })
        return alts[:top_n]
    except Exception as e:
        print(f"[WARN] Universe: {e}")
        return []

# ── Fiyat geçmişi ─────────────────────────────────────────────
def get_price_history(coin_id):
    try:
        data = cg_get(f"/coins/{coin_id}/market_chart",
                      {"vs_currency": "usd", "days": 60, "interval": "daily"})
        prices = [p[1] for p in data.get("prices", [])]
        return pd.Series(prices)
    except Exception:
        return pd.Series()

# ── HyperTrend Skorlayıcı ─────────────────────────────────────
def score(close):
    if len(close) < 25:
        return {"score": 0.0, "trend": 0.0, "momentum": 0.0, "carry": 0.0}

    # Trend: 20-gün high kırılımı
    hi = close.shift(1).rolling(20).max()
    lo = close.shift(1).rolling(20).min()
    t  = pd.Series(0.0, index=close.index)
    t[close > hi] =  1.0
    t[close < lo] = -1.0

    # Momentum: 20-gün risk-adj return
    ret = close.pct_change(20)
    vol = close.pct_change().rolling(20).std() * np.sqrt(365)
    m   = (ret / (vol + 1e-9)).fillna(0)

    def zs(s):
        mu, std = s.mean(), s.std()
        if std == 0: return 0.0
        return float(np.clip((s.iloc[-1] - mu) / std, -3, 3))

    t_z = zs(t)
    m_z = zs(m)
    combined = float(np.clip(0.6*t_z + 0.4*m_z, -3, 3)) / 3

    return {"score": round(combined, 3), "trend": round(t_z, 2),
            "momentum": round(m_z, 2), "carry": 0.0}

def label(s):
    if s >  0.55: return "STRONG_LONG"
    if s >  0.20: return "LONG"
    if s < -0.55: return "STRONG_SHORT"
    if s < -0.20: return "SHORT"
    return "FLAT"

def pos_size(close, score_val, mult):
    if close.empty: return 0.0
    vol = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(365)
    if not vol or np.isnan(vol): return 0.0
    raw = (0.15 / vol) * abs(score_val) * mult
    return round(float(np.clip(raw, 0, 0.20)) * 100, 1)

# ── Main ──────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Altcoin Alpha Engine başlıyor...")

    print("→ BTC regime...")
    regime = get_btc_regime()
    print(f"  {regime['regime']} | ${regime['spot']:,.0f} | 24h: {regime['chg_24h']:+.1f}%")

    print("→ Universe çekiliyor...")
    universe = get_universe(20)
    print(f"  {len(universe)} coin bulundu")

    signals = []
    for i, coin in enumerate(universe):
        sym = coin["symbol"]
        try:
            close  = get_price_history(coin["id"])
            sc     = score(close)
            lbl    = label(sc["score"])
            psize  = pos_size(close, sc["score"], regime["alt_multiplier"])

            sig = {
                "symbol":        sym,
                "price":         coin["price"],
                "volume_usd":    coin["volume_usd"],
                "signal":        lbl,
                "score":         sc["score"],
                "trend_z":       sc["trend"],
                "momentum_z":    sc["momentum"],
                "carry_z":       0.0,
                "funding":       0.0,
                "funding_z":     0.0,
                "spike":         False,
                "spike_note":    "",
                "pos_size_pct":  psize,
                "regime_mult":   regime["alt_multiplier"],
                "price_chg_24h": round(coin["price_chg_24h"], 2)
            }
            signals.append(sig)

            bar = ("+" * max(0, int(sc["score"]*10))) if sc["score"] >= 0 \
                  else ("-" * max(0, int(-sc["score"]*10)))
            print(f"  {sym:8s} {lbl:14s} {sc['score']:+.2f} {bar}")

            # CoinGecko rate limit: 30 req/dk → her istekte bekle
            if i < len(universe) - 1:
                time.sleep(2.5)

        except Exception as e:
            print(f"  {sym:8s} [HATA] {e}")

    signals.sort(key=lambda x: abs(x["score"]), reverse=True)
    longs  = [s for s in signals if s["signal"] in ("LONG","STRONG_LONG")]
    shorts = [s for s in signals if s["signal"] in ("SHORT","STRONG_SHORT")]

    output = {
        "generated_at":     now.isoformat(),
        "generated_at_str": now.strftime("%d %b %Y, %H:%M UTC"),
        "btc_regime":       regime,
        "portfolio_summary": {
            "total_long_pct":  round(min(sum(s["pos_size_pct"] for s in longs), 80), 1),
            "total_short_pct": round(min(sum(s["pos_size_pct"] for s in shorts), 40), 1),
            "long_count":      len(longs),
            "short_count":     len(shorts),
            "flat_count":      len(signals) - len(longs) - len(shorts)
        },
        "signals": signals
    }

    out = DOCS / "signals.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (DOCS / "last_run.txt").write_text(now.isoformat())

    print(f"\n  Long: {len(longs)} | Short: {len(shorts)} | Flat: {len(signals)-len(longs)-len(shorts)}")
    print(f"  Çıktı: {out}")

if __name__ == "__main__":
    main()
