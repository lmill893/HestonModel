"""
Monte Carlo / Heston Options Pricer — Flask Backend
Deploy to Railway: https://railway.app
"""

import os
import math
import traceback
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)

# Allow all origins — required for file:// HTML and Railway
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(response):
    """Belt-and-braces CORS — ensures headers survive any middleware."""
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def safe_float(value, default=0.0):
    """Convert to float, returning default on None / NaN / errors.

    Critical fix: float(NaN or 0) == NaN because NaN is truthy in Python.
    The old 'row.get(...) or 0' pattern silently passed NaN through to JSON,
    which then broke the frontend with serialisation errors.
    """
    try:
        v = float(value)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """Convert to int, returning default on None / NaN / errors.

    int(float('nan')) raises ValueError — must sanitise before casting.
    """
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "message": "Monte Carlo / Heston Options API is running"})


# ── Stock price ───────────────────────────────────────────────────────────────

@app.route("/api/stock", methods=["GET", "OPTIONS"])
def get_stock():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        stock = yf.Ticker(ticker)

        # Try fast_info first (no extra HTTP call), fall back to full info dict
        price = None
        try:
            fi    = stock.fast_info
            price = safe_float(getattr(fi, "last_price", None)) or \
                    safe_float(getattr(fi, "previous_close", None))
        except Exception:
            pass

        if not price:
            try:
                info  = stock.info
                price = safe_float(info.get("currentPrice"))       or \
                        safe_float(info.get("regularMarketPrice")) or \
                        safe_float(info.get("previousClose"))
            except Exception:
                pass

        if not price:
            return jsonify({"error": f"Could not fetch price for '{ticker}'. Check the ticker is valid."}), 404

        # Name / currency — best effort, never let this crash the endpoint
        name     = ticker
        currency = "USD"
        try:
            info     = stock.info   # yfinance caches this after the first call
            name     = info.get("longName") or info.get("shortName") or ticker
            currency = info.get("currency", "USD") or "USD"
        except Exception:
            pass

        return jsonify({
            "ticker":   ticker,
            "price":    round(price, 2),
            "name":     name,
            "currency": currency,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error fetching {ticker}: {str(e)}"}), 500


# ── Option chain ──────────────────────────────────────────────────────────────

@app.route("/api/options", methods=["GET", "OPTIONS"])
def get_options():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    ticker   = request.args.get("ticker", "").upper().strip()
    expiry   = request.args.get("expiry", None)
    opt_type = request.args.get("type", "calls").lower()

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        stock    = yf.Ticker(ticker)
        expiries = list(stock.options)

        if not expiries:
            return jsonify({"error": f"No options data available for '{ticker}'."}), 404

        selected = expiry if (expiry and expiry in expiries) else expiries[0]
        chain    = stock.option_chain(selected)
        df       = chain.calls if opt_type == "calls" else chain.puts

        rows = []
        for _, row in df.iterrows():
            try:
                strike = safe_float(row.get("strike"))
                if strike <= 0:
                    continue    # skip any row without a valid strike

                rows.append({
                    "strike":       round(strike, 2),
                    "lastPrice":    round(safe_float(row.get("lastPrice")),          4),
                    "bid":          round(safe_float(row.get("bid")),                4),
                    "ask":          round(safe_float(row.get("ask")),                4),
                    "impliedVol":   round(safe_float(row.get("impliedVolatility")),  4),
                    "volume":       safe_int(row.get("volume")),
                    "openInterest": safe_int(row.get("openInterest")),
                    "inTheMoney":   bool(row.get("inTheMoney", False)),
                })
            except Exception:
                continue    # silently skip malformed rows

        return jsonify({
            "ticker":   ticker,
            "expiry":   selected,
            "type":     opt_type,
            "expiries": expiries[:12],
            "options":  rows,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error fetching options for {ticker}: {str(e)}"}), 500


# ── Historical volatility ─────────────────────────────────────────────────────

@app.route("/api/volatility", methods=["GET", "OPTIONS"])
def get_volatility():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="1y")

        if hist.empty:
            return jsonify({"error": f"No price history found for '{ticker}'."}), 404

        closes      = hist["Close"].dropna()
        log_returns = np.log(closes / closes.shift(1)).dropna()

        if len(log_returns) < 10:
            return jsonify({"error": f"Insufficient price history for '{ticker}' (need at least 10 days)."}), 404

        daily_vol  = float(log_returns.std())
        annual_vol = round(daily_vol * np.sqrt(252), 4)

        return jsonify({
            "ticker":       ticker,
            "annualVol":    annual_vol,
            "dailyVol":     round(daily_vol, 6),
            "thetaHint":    round(annual_vol ** 2, 6),  # v0 = sigma^2 for Heston init
            "observations": len(log_returns),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error computing volatility for {ticker}: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Monte Carlo / Heston Options API  ->  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
