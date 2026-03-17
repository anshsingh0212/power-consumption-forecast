"""
forecast.py
Power consumption forecasting pipeline using multiple time-series models.

Models:
  1. Baseline (same-day-last-week)
  2. Linear Regression with calendar features
  3. Random Forest Regressor
  4. XGBoost Regressor with lag features
  5. SARIMA (seasonal decomposition + evaluation)

Forecasts daily energy consumption per feeder (aggregated from smart meters).
Author: Ansh Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings, os, json
from datetime import datetime

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost               import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_feeder_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic daily feeder-level aggregated energy consumption.
    Simulates one year of data for 3 feeders (residential, commercial, mixed).
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    records = []
    feeder_params = {
        "F-01_Residential": {"base": 850,  "seasonal_amp": 120, "trend": 0.3,  "noise": 40},
        "F-02_Commercial":  {"base": 2400, "seasonal_amp": 200, "trend": 0.5,  "noise": 90},
        "F-03_Industrial":  {"base": 8500, "seasonal_amp": 500, "trend": 1.2,  "noise": 250},
    }

    for feeder, p in feeder_params.items():
        for i, d in enumerate(dates):
            # Long-term trend
            trend = p["trend"] * i
            # Seasonal: summer peak (India: April–June hot, Jan cool)
            seasonal = p["seasonal_amp"] * np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
            # Weekly pattern
            weekly = -p["base"] * 0.12 if d.weekday() >= 5 else 0
            if feeder == "F-01_Residential":
                weekly = p["base"] * 0.08 if d.weekday() >= 5 else 0
            # Random noise
            noise = np.random.normal(0, p["noise"])
            # Occasional outage/event dip
            event_dip = -p["base"] * 0.25 if np.random.rand() < 0.02 else 0

            energy = max(0, p["base"] + trend + seasonal + weekly + noise + event_dip)

            records.append({
                "date"         : d,
                "feeder_id"    : feeder,
                "energy_mwh"   : round(energy / 1000, 4),   # Convert kWh → MWh
                "temp_c"       : round(15 + 15 * np.sin(2 * np.pi * (d.dayofyear - 30) / 365)
                                       + np.random.normal(0, 2), 1),
                "humidity_pct" : round(60 + 20 * np.sin(2 * np.pi * d.dayofyear / 365)
                                       + np.random.normal(0, 5), 1),
                "is_holiday"   : int(np.random.rand() < 0.025),
            })

    df = pd.DataFrame(records)
    df.to_csv("data/feeder_daily_energy.csv", index=False)
    print(f"Generated {len(df):,} records | {df['feeder_id'].nunique()} feeders | {n_days} days")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["feeder_id", "date"]).reset_index(drop=True)
    grp = df.groupby("feeder_id")["energy_mwh"]

    # Calendar
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df["date"].dt.quarter

    # Lag features
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = grp.transform(lambda x: x.shift(lag))

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"roll_mean_{window}"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"roll_std_{window}"]  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

    # Cyclical encoding (avoids ordinality issues with sin/cos)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "day_of_week", "month", "day_of_year", "week_of_year",
    "is_weekend", "quarter", "is_holiday",
    "temp_c", "humidity_pct",
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7",  "roll_std_30",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]


def evaluate_model(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

    print(f"  {name:<30} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE": round(mape, 3)}


def train_and_evaluate(df: pd.DataFrame, feeder: str) -> pd.DataFrame:
    fdata = df[df["feeder_id"] == feeder].copy().sort_values("date")
    fdata = create_features(fdata)

    # 80/20 chronological split
    split_idx = int(len(fdata) * 0.8)
    train = fdata.iloc[:split_idx]
    test  = fdata.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train["energy_mwh"]
    X_test,  y_test  = test[FEATURE_COLS],  test["energy_mwh"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = []

    # 1. Baseline: same day last week
    baseline_pred = test["lag_7"].values
    results.append(evaluate_model("Baseline (Same-Day-Last-Week)", y_test.values, baseline_pred))

    # 2. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    results.append(evaluate_model("Linear Regression", y_test, lr.predict(X_test_s)))

    # 3. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    results.append(evaluate_model("Ridge Regression", y_test, ridge.predict(X_test_s)))

    # 4. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", y_test, rf.predict(X_test)))

    # 5. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    results.append(evaluate_model("Gradient Boosting", y_test, gb.predict(X_test)))

    # 6. XGBoost
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    xgb_pred = xgb.predict(X_test)
    results.append(evaluate_model("XGBoost", y_test, xgb_pred))

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"outputs/model_comparison_{feeder.replace(' ','_')}.csv", index=False)

    return test, y_test, xgb_pred, xgb, results_df, train, fdata


# ══════════════════════════════════════════════════════════════════════════════
#  4. SEASONAL DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def decompose_and_plot(df: pd.DataFrame, feeder: str):
    fdata = df[df["feeder_id"] == feeder].set_index("date")["energy_mwh"].asfreq("D")
    fdata = fdata.fillna(method="ffill")

    decomp = seasonal_decompose(fdata, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f"Seasonal Decomposition — {feeder}", fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes,
                                [fdata, decomp.trend, decomp.seasonal, decomp.resid],
                                ["Observed", "Trend", "Seasonal (Weekly)", "Residual"]):
        ax.plot(data.index, data.values, linewidth=0.9,
                color="steelblue" if title != "Residual" else "salmon")
        ax.set_title(title)
        ax.set_ylabel("MWh")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=8)

    plt.tight_layout()
    fname = f"outputs/decomposition_{feeder.split('_')[0]}.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. FORECAST VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast(test_df, y_test, xgb_pred, feeder, results_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Power Consumption Forecast — {feeder}", fontsize=14, fontweight="bold")

    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.plot(test_df["date"].values, y_test.values, label="Actual",    color="steelblue", lw=2)
    ax.plot(test_df["date"].values, xgb_pred,      label="XGBoost",   color="darkorange", lw=2, linestyle="--")
    ax.set_title("Actual vs XGBoost Forecast (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (MWh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=8)

    # 2. Residuals
    ax2 = axes[0, 1]
    residuals = y_test.values - xgb_pred
    ax2.scatter(xgb_pred, residuals, alpha=0.4, color="steelblue", s=15)
    ax2.axhline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_title("Residual Plot")
    ax2.set_xlabel("Predicted (MWh)")
    ax2.set_ylabel("Residual")

    # 3. Model comparison bar chart
    ax3 = axes[1, 0]
    results_df = results_df.sort_values("MAE")
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_df))]
    ax3.barh(results_df["model"], results_df["MAE"], color=colors, edgecolor="black")
    ax3.set_title("Model Comparison — MAE (lower is better)")
    ax3.set_xlabel("MAE (MWh)")
    ax3.invert_yaxis()

    # 4. MAPE comparison
    ax4 = axes[1, 1]
    ax4.barh(results_df["model"], results_df["MAPE"], color=colors, edgecolor="black")
    ax4.set_title("Model Comparison — MAPE % (lower is better)")
    ax4.set_xlabel("MAPE (%)")
    ax4.invert_yaxis()

    plt.tight_layout()
    fname = f"outputs/forecast_{feeder.split('_')[0]}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_combined_dashboard(df: pd.DataFrame):
    """Overlay all 3 feeders on one chart for executive overview."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("All Feeders — Annual Energy Consumption Overview", fontsize=13, fontweight="bold")

    colors = {"F-01_Residential": "#3498db", "F-02_Commercial": "#e67e22", "F-03_Industrial": "#2ecc71"}

    for feeder, grp in df.groupby("feeder_id"):
        grp = grp.sort_values("date")
        # Weekly rolling to smooth
        smoothed = grp["energy_mwh"].rolling(7, center=True).mean()
        axes[0].plot(grp["date"], smoothed, label=feeder, color=colors[feeder], lw=2)

    axes[0].set_title("Weekly Smoothed Energy (All Feeders)")
    axes[0].set_ylabel("Energy (MWh/day)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, fontsize=8)

    # Monthly totals stacked bar
    df["month_year"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month_year", "feeder_id"])["energy_mwh"].sum().unstack(fill_value=0)
    monthly.plot(kind="bar", ax=axes[1], color=list(colors.values()), edgecolor="black", linewidth=0.5)
    axes[1].set_title("Monthly Energy by Feeder (MWh)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Energy (MWh)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/all_feeders_overview.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/all_feeders_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Power Consumption Forecasting Pipeline")
    print("=" * 65)

    os.makedirs("data", exist_ok=True)
    df = generate_feeder_data(n_days=365)

    all_metrics = {}
    for feeder in df["feeder_id"].unique():
        print(f"\n{'─'*55}")
        print(f"  Feeder: {feeder}")
        print(f"{'─'*55}")

        decompose_and_plot(df, feeder)
        test_df, y_test, xgb_pred, xgb_model, results_df, train, full = train_and_evaluate(df, feeder)
        plot_forecast(test_df, y_test, xgb_pred, feeder, results_df)
        all_metrics[feeder] = results_df.to_dict("records")

    plot_combined_dashboard(df)

    with open("outputs/all_model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✓ Forecasting pipeline complete.")
    print("  Outputs saved to outputs/")


if __name__ == "__main__":
    main()
