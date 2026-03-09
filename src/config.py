# EPForecast Methodology — ML Configuration Constants
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app

"""EPForecast ML configuration constants."""

# ── Model defaults ────────────────────────────────────────────────────
QUANTILE_TARGET = 0.55  # Quantile loss target (None = MAE/absolute_error)

# ── Horizon groups ────────────────────────────────────────────────────
# Day-ahead (DA): D+1 predictions, origin ~10:00 UTC
#   DA1: hours 0-11 of D+1 (14-25h ahead)
#   DA2: hours 12-23 of D+1 (26-37h ahead)
#
# Strategic (S): D+2 to D+7 predictions, origin ~15:00 UTC
#   S1: D+2 (33-56h ahead)
#   S2: D+3 (57-80h ahead)
#   S3: D+4 (81-104h ahead)
#   S4: D+5 (105-128h ahead)
#   S5: D+6-D+7 (129-177h ahead)

# ── Feature engineering ───────────────────────────────────────────────
MIN_HISTORY_HOURS = 504  # 3 weeks of history required for multi-week lags

# ── Price sanity check ────────────────────────────────────────────────
MAX_PRICE_EUR_MWH = 500  # Abort if max price exceeds this threshold
