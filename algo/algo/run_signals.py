"""
algo/run_signals.py
Deribit PUBLIC API'den veri çeker (auth gerektirmez)
Sinyalleri docs/signals.json'a yazar → GitHub Pages'te gösterilir

Çalıştır: python algo/run_signals.py
"""

import json
import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# ── Dizin yapısı ──────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
DOCS   = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
(ROOT / "algo").mkdir(exist_ok=True)

# ── Deribit Public API ─────────────────────────────────────────
DERIBIT = "https://www.deribit.com/api/v2/public"

def deribit_get(method: str, params: dict = {}) -> dict:
    r = requests.get(f"{DERIBIT}/{method}", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})

# ── Binance Public API (funding + fiyat) ──────────────────────
BINANCE = "https://fapi.binance.com/fapi/v1"

def binance_get(endpoint: str, params: dict = {}) -> list | dict:
    r = requests.get(f"{BINANCE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# ══════════════════════════════════════════════════════════════
# 1. BTC MACRO REGİME (Deribit'ten GEX proxy)
# ══════════════════════════════════════════════════════════════

def get_btc_macro_regime() -> dict:
    """
    Deribit public API'den BTC options verisi çekip
    basit GEX proxy hesaplar.
    Tam G-DIVE bağlantısı olmadan çalışır.
    """
    try:
        # BTC spot
        ticker = deribit_get("get_index_price", {"index_name": "btc_usd"})
        spot   = float(ticker.get("index_price", 0))

        # BTC option book summary — açık pozisyon bilgisi
        options = deribit_get("get_book_summary_by_currency",
                              {"currency": "BTC", "kind": "option"})

        if not options or spot == 0:
            return {"regime": "NEUTRAL", "spot": spot, "net_gex_proxy": 0}

        # GEX proxy: her strike için gamma × OI × direction
        net_gex = 0.0
        for opt in options:
            try:
                strike     = float(opt.get("strike", 0))
                oi         = float(opt.get("open_interest", 0))
                option_type = "call" if "C" in opt.get("instrument_name", "") else "put"
                mid        = float(opt.get("mid_price", 0)) * spot

                moneyness  = (spot - strike) / spot
                # Basit gamma proxy: OI × moneyness proximity
                gamma_proxy = oi * np.exp(-0.5 * (moneyness / 0.05) ** 2)

                if option_type == "call":
                    net_gex += gamma_proxy
                else:
                    net_gex -= gamma_proxy
            except (ValueError, TypeError):
                continue

        regime = "BULL_GAMMA" if net_gex > 0 else "BEAR_GAMMA"
        flip_est = spot * (1 - 0.02 if net_gex > 0 else 1 + 0.02)

        return {
            "regime":        regime,
            "spot":          round(spot, 0),
            "net_gex_proxy": round(net_gex, 0),
            "flip_estimate": round(flip_est, 0),
            "alt_multiplier": 1.0 if regime == "BULL_GAMMA" else 0.0
        }

    except Exception as e:
        print(f"[WARN] BTC regime hatası: {e}")
        return {"regime": "NEUTRAL", "spot": 0, "net_gex_proxy": 0, "alt_multiplier": 0.6}


# ══════════════════════════════════════════════════════════════
# 2. ALTCOIN UNIVERSE (Binance top perpetuals)
# ══════════════════════════════════════════════════════════════

def get_altcoin_universe(top_n: int = 20) -> list[dict]:
    """
    Binance perp market'tan en likit altcoin universe'ini al.
    $50M-$5B market cap filtresi burada hacim proxy olarak uygulanır.
    """
    try:
        tickers = binance_get("ticker/24hr")
        alts = []

        for t in tickers:
            symbol = t.get("symbol", "")
            if not symbol.endswith("USDT") or "BTC" in symbol or "ETH" in symbol:
                continue

            volume_usd = float(t.get("quoteVolume", 0))
            price      = float(t.get("lastPrice", 0))
            price_chg  = float(t.get("priceChangePercent", 0))
            count      = int(t.get("count", 0))

            # Hacim filtresi: $5M-$500M arası (size decay proxy)
            if not (5_000_000 <= volume_usd <= 500_000_000):
                continue

            alts.append({
                "symbol":     symbol.replace("USDT", ""),
                "price":      round(price, 6),
                "volume_usd": round(volume_usd, 0),
                "price_chg_24h": round(price_chg, 2),
                "trade_count": count
            })

        # Hacme göre sırala, top N al
        alts.sort(key=lambda x: x["volume_usd"], reverse=True)
        return alts[:top_n]

    except Exception as e:
        print(f"[WARN] Universe hatası: {e}")
        return []


# ══════════════════════════════════════════════════════════════
# 3. HYPERTREND SINYAL HESAPLAYICI
# ══════════════════════════════════════════════════════════════

def get_klines(symbol: str, interval: str = "4h", limit: int = 100) -> pd.DataFrame:
    """Binance'ten OHLCV verisi çek."""
    try:
        data = binance_get("klines", {
            "symbol":   symbol + "USDT",
            "interval": interval,
            "limit":    limit
        })
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
        df["close"]  = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["quote_vol"])
        df.set_index("time", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_funding_rate(symbol: str) -> float:
    """Anlık funding rate al."""
    try:
        data = binance_get("premiumIndex", {"symbol": symbol + "USDT"})
        return float(data.get("lastFundingRate", 0))
    except Exception:
        return 0.0


def compute_hypertrend_score(df: pd.DataFrame, funding: float) -> dict:
    """
    Trend (50%) + Momentum (30%) + Carry (20%)
    """
    if len(df) < 60:
        return {"score": 0.0, "trend": 0.0, "momentum": 0.0, "carry": 0.0}

    close = df["close"]

    # Trend: 20-bar high kırılımı
    hi = close.shift(1).rolling(20).max()
    lo = close.shift(1).rolling(20).min()
    trend_raw = pd.Series(0.0, index=close.index)
    trend_raw[close > hi] =  1.0
    trend_raw[close < lo] = -1.0

    # Momentum: 60-bar risk-adjusted return
    ret = close.pct_change(60)
    vol = close.pct_change().rolling(60).std() * np.sqrt(365 * 6)  # 4h → annual
    mom_raw = (ret / (vol + 1e-9)).fillna(0)

    # Carry: funding rate (negatif → long carry)
    annual_funding = funding * 3 * 365
    carry_raw = pd.Series(-annual_funding, index=close.index)

    def last_zscore(s: pd.Series) -> float:
        mu  = s.rolling(60).mean().iloc[-1]
        std = s.rolling(60).std().iloc[-1]
        val = s.iloc[-1]
        if std == 0 or np.isnan(std):
            return 0.0
        return float(np.clip((val - mu) / std, -3, 3))

    t_z = last_zscore(trend_raw)
    m_z = last_zscore(mom_raw)
    c_z = last_zscore(carry_raw)

    combined = float(np.clip(0.5*t_z + 0.3*m_z + 0.2*c_z, -3, 3)) / 3

    return {
        "score":    round(combined, 3),
        "trend":    round(t_z, 2),
        "momentum": round(m_z, 2),
        "carry":    round(c_z, 2)
    }


def get_signal_label(score: float) -> str:
    if score >  0.60: return "STRONG_LONG"
    if score >  0.25: return "LONG"
    if score < -0.60: return "STRONG_SHORT"
    if score < -0.25: return "SHORT"
    return "FLAT"


def compute_position_size(df: pd.DataFrame, score: float, regime_mult: float) -> float:
    """Vol-weighted position sizing."""
    if df.empty:
        return 0.0
    vol = df["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(365 * 6)
    if vol == 0 or np.isnan(vol):
        return 0.0
    target_vol = 0.15
    raw = (target_vol / vol) * abs(score) * regime_mult
    return round(float(np.clip(raw, 0, 0.20)) * 100, 1)


# ══════════════════════════════════════════════════════════════
# 4. FUNDING MANİPÜLASYON KONTROLÜ
# ══════════════════════════════════════════════════════════════

def check_funding_spike(symbol: str) -> dict:
    """Son 48 funding periyodunun z-score'unu hesapla."""
    try:
        data = binance_get("fundingRate", {
            "symbol": symbol + "USDT",
            "limit":  50
        })
        rates = [float(d["fundingRate"]) for d in data]
        if len(rates) < 10:
            return {"spike": False, "z": 0.0}
        arr   = np.array(rates)
        mu    = arr[:-1].mean()
        std   = arr[:-1].std()
        last  = arr[-1]
        z     = float((last - mu) / (std + 1e-9))

        if abs(z) > 3.0:
            direction = "pos" if z > 0 else "neg"
            return {
                "spike":     True,
                "z":         round(z, 2),
                "direction": direction,
                "action":    f"KONTRARIAN_{'LONG' if direction == 'pos' else 'SHORT'}",
                "note":      f"Funding spike {z:+.1f}σ → flush sonrası fırsat"
            }
        return {"spike": False, "z": round(z, 2)}

    except Exception:
        return {"spike": False, "z": 0.0}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Altcoin Alpha Engine başlıyor...")

    # 1. BTC macro regime
    print("→ BTC regime analiz ediliyor...")
    regime = get_btc_macro_regime()
    print(f"  Regime: {regime['regime']} | Spot: ${regime['spot']:,.0f} | "
          f"GEX proxy: {regime['net_gex_proxy']:+.0f}")

    # 2. Altcoin universe
    print("→ Altcoin universe çekiliyor...")
    universe = get_altcoin_universe(top_n=20)
    print(f"  {len(universe)} altcoin filtreye geçti")

    # 3. Her coin için sinyal hesapla
    signals = []
    for coin in universe:
        sym = coin["symbol"]
        try:
            df      = get_klines(sym, interval="4h", limit=100)
            funding = get_funding_rate(sym)
            scores  = compute_hypertrend_score(df, funding)
            spike   = check_funding_spike(sym)

            score = scores["score"]

            # Funding spike varsa kontrarian boost
            if spike["spike"]:
                if spike["direction"] == "pos":
                    score = max(score, 0.35)
                else:
                    score = min(score, -0.35)

            pos_size = compute_position_size(df, score, regime["alt_multiplier"])
            label    = get_signal_label(score)

            sig = {
                "symbol":      sym,
                "price":       coin["price"],
                "volume_usd":  coin["volume_usd"],
                "signal":      label,
                "score":       round(score, 3),
                "trend_z":     scores["trend"],
                "momentum_z":  scores["momentum"],
                "carry_z":     scores["carry"],
                "funding":     round(funding * 100, 4),
                "funding_z":   spike.get("z", 0.0),
                "spike":       spike.get("spike", False),
                "spike_note":  spike.get("note", ""),
                "pos_size_pct": pos_size,
                "regime_mult": regime["alt_multiplier"],
                "price_chg_24h": coin["price_chg_24h"]
            }
            signals.append(sig)

            bar = "+" * max(0, int(score * 10)) if score >= 0 else "-" * max(0, int(-score * 10))
            print(f"  {sym:8s} {label:14s} score={score:+.2f} {bar}")
            time.sleep(0.15)  # rate limit

        except Exception as e:
            print(f"  {sym:8s} [HATA] {e}")
            continue

    # Skora göre sırala
    signals.sort(key=lambda x: abs(x["score"]), reverse=True)

    # 4. Portföy özeti
    longs  = [s for s in signals if s["signal"] in ("LONG", "STRONG_LONG")]
    shorts = [s for s in signals if s["signal"] in ("SHORT", "STRONG_SHORT")]
    total_long_pct  = min(sum(s["pos_size_pct"] for s in longs), 80.0)
    total_short_pct = min(sum(s["pos_size_pct"] for s in shorts), 40.0)

    # 5. Output JSON
    output = {
        "generated_at":      now.isoformat(),
        "generated_at_str":  now.strftime("%d %b %Y, %H:%M UTC"),
        "btc_regime": regime,
        "portfolio_summary": {
            "total_long_pct":  round(total_long_pct, 1),
            "total_short_pct": round(total_short_pct, 1),
            "long_count":      len(longs),
            "short_count":     len(shorts),
            "flat_count":      len(signals) - len(longs) - len(shorts)
        },
        "signals": signals
    }

    out_path = DOCS / "signals.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    (DOCS / "last_run.txt").write_text(now.isoformat())

    print(f"\n  ÖZET:")
    print(f"  Long:  {len(longs)} coin (%{round(total_long_pct,1)} portföy)")
    print(f"  Short: {len(shorts)} coin (%{round(total_short_pct,1)} portföy)")
    print(f"  Flat:  {len(signals)-len(longs)-len(shorts)} coin")
    print(f"\n  Çıktı: {out_path}")
    print("  Tamamlandı.")


if __name__ == "__main__":
    main()
