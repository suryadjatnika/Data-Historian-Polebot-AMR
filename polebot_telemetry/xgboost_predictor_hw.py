#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime

from influxdb_client import InfluxDBClient
import xgboost as xgb

# KONFIGURASI
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_hw"

DATA_RANGE = "start: -15m"
OUTPUT_DIR  = os.path.expanduser("~/xgboost_results_demo")
FORECAST_N  = 60     # prediksi 60 detik ke depan
LOOK_BACK   = 10     # fitur lag (10 timestep sebelumnya)

# XGBoost parameter
XGB_PARAMS = {
    'n_estimators'    : 500,
    'max_depth'       : 6,
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'random_state'    : 42,
    'n_jobs'          : -1,
}

# Variabel target
# XGBoost menggunakan fitur multi-variabel
# Target utama: P_total (beban motor)
# Fitur input : v_linear, accel, omega, SOC
TARGET_VARIABLES = [
    {
        'field'  : 'joint_P_total',
        'label'  : 'Total Motor Power (P_total)',
        'unit'   : 'Watt',
        'color'  : '#c084fc',
    },
    {
        'field'  : 'batt_soc_percent',
        'label'  : 'Battery State of Charge (SOC)',
        'unit'   : '%',
        'color'  : '#22c55e',
    },
    {
        'field'  : 'odom_v_linear',
        'label'  : 'Linear Velocity',
        'unit'   : 'm/s',
        'color'  : '#fb923c',
    },
]

# Semua field yang diambil dari InfluxDB
ALL_FIELDS = [
    'joint_P_total',
    'batt_soc_percent',
    'odom_v_linear',
    'odom_accel',
    'odom_omega',
    'joint_load_ratio',
    'batt_power_draw',
]

# FUNGSI UTAMA
def fetch_all_data() -> pd.DataFrame:
    """
    Ambil semua field sekaligus dari InfluxDB,
    lalu pivot menjadi DataFrame wide format.
    Setiap kolom = satu field sensor.
    """
    print(f"\n  Mengambil data dari InfluxDB ({len(ALL_FIELDS)} field)...")

    client = InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    fields_filter = ' or '.join([f'r._field == "{f}"' for f in ALL_FIELDS])

    query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => r.source == "hardware_pzem")
  |> filter(fn: (r) => {fields_filter})
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns: ["_time"])
'''

    try:
        df = client.query_api().query_data_frame(query)
        client.close()

        # Handle list of DataFrames (pivot query kadang return multiple tables)
        if isinstance(df, list):
            if len(df) == 0:
                print("Data kosong!")
                return None
            df = pd.concat(df, ignore_index=True)

        if df.empty:
            print("Data kosong!")
            return None

        # Pastikan semua kolom numerik
        for col in ALL_FIELDS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Bersihkan kolom metadata InfluxDB
        cols_keep = ['_time'] + [c for c in ALL_FIELDS if c in df.columns]
        df = df[cols_keep].copy()
        df['_time'] = pd.to_datetime(df['_time'], utc=True)
        df = df.set_index('_time')
        df.index = df.index.tz_convert('Asia/Jakarta').tz_localize(None)
        df = df.resample('1s').mean().dropna()

        if 'odom_accel' in df.columns and 'odom_v_linear' in df.columns:
            df['is_dynamic'] = (
                (df['odom_accel'].abs() >= 0.10) |
                (df['odom_v_linear'].abs() >= 0.18)
            )
            n_dyn = df['is_dynamic'].sum()
            print(f"  Kondisi dinamis: {n_dyn} titik")
            print(f"  Kondisi statis : {len(df) - n_dyn} titik")

        print(f"  ✅ {len(df)} titik data, {len(df.columns)} field")
        print(f"     Field: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"Gagal: {e}")
        return None


def create_lag_features(df: pd.DataFrame,
                        target: str,
                        look_back: int) -> pd.DataFrame:
    """
    Buat fitur lag dari semua kolom.
    Ini yang membedakan XGBoost dari ARIMA/LSTM:
    XGBoost bisa menggunakan BANYAK FITUR sekaligus,
    bukan hanya satu variabel time-series.

    Fitur yang dibuat:
      - Lag 1–look_back untuk target variable
      - Lag 1–3 untuk fitur pendukung lainnya
      - Fitur turunan: rolling mean, rolling std
    """
    df_feat = df.copy()

    # Lag fitur untuk target
    for lag in range(1, look_back + 1):
        df_feat[f'{target}_lag{lag}'] = df_feat[target].shift(lag)

    # Lag fitur untuk variabel pendukung (lag 1, 2, 3)
    support_cols = [c for c in df.columns if c != target]
    for col in support_cols:
        for lag in range(1, 4):
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # Fitur rolling (mean dan std dari 5 detik terakhir)
    df_feat[f'{target}_roll_mean5'] = df_feat[target].rolling(5).mean()
    df_feat[f'{target}_roll_std5']  = df_feat[target].rolling(5).std()

    # Hapus baris dengan NaN (akibat shifting)
    df_feat = df_feat.dropna()

    return df_feat


def train_xgboost(df: pd.DataFrame, config: dict) -> dict:
    target = config['field']
    label  = config['label']
    unit   = config['unit']

    print(f"\n  Membuat fitur lag (look_back={LOOK_BACK})...")
    if 'is_dynamic' in df.columns:
        df['is_dynamic'] = df['is_dynamic'].astype(int)
    
    df_feat = create_lag_features(df, target, LOOK_BACK)

    # Filter kondisi dinamis hanya untuk target SOC baterai
    if target == 'batt_soc_percent' and 'is_dynamic' in df_feat.columns:
        before = len(df_feat)
        df_feat = df_feat[df_feat['is_dynamic'] == True]
        print(f"  Filter dinamis (XGBoost SOC): {before} → {len(df_feat)} titik")

    # Gunakan semua kolom kecuali target itu sendiri sebagai fitur
    X_cols = [c for c in df_feat.columns if c != target]
    X = df_feat[X_cols]
    y = df_feat[target]

    print(f"  Jumlah fitur: {len(X_cols)}")
    print(f"  Jumlah data : {len(X)}")

    # Split 80% train, 20% test (time-ordered, tidak shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Data train : {len(X_train)} titik")
    print(f"  Data test  : {len(X_test)} titik")

    # Latih XGBoost
    print(f"\n  Melatih model XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluasi
    pred_test = model.predict(X_test)

    mae   = np.mean(np.abs(y_test.values - pred_test))
    rmse  = np.sqrt(np.mean((y_test.values - pred_test) ** 2))
    
    # sMAPE: exclude titik di mana KEDUANYA mendekati nol
    _both_near_zero = (np.abs(y_test.values) < 0.05) & (np.abs(pred_test) < 0.05)
    _mask_smape = ~_both_near_zero
    if _mask_smape.sum() > 10:
        smape = np.mean(
            2 * np.abs(y_test.values[_mask_smape] - pred_test[_mask_smape]) /
            (np.abs(y_test.values[_mask_smape]) + np.abs(pred_test[_mask_smape]) + 1e-10)
        ) * 100
    else:
        smape = 0.0

    print(f"\n  Hasil Evaluasi ")
    print(f"  MAE   : {mae:.4f} {unit}")
    print(f"  RMSE  : {rmse:.4f} {unit}")
    print(f"  sMAPE : {smape:.2f}%")

    # Feature Importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_cols
    ).sort_values(ascending=False).head(10)

    print(f"\n  Top 5 Fitur Terpenting:")
    for feat, imp in importance.head(5).items():
        print(f"    {feat:<35} {imp:.4f}")

    # Forecast iteratif
    print(f"\n  Memprediksi {FORECAST_N} detik ke depan")

    # Gunakan baris terakhir sebagai seed forecast
    last_row     = df_feat.iloc[-1:].copy()
    forecast_vals = []
    df_rolling   = df.copy()

    for step in range(FORECAST_N):
        # Buat fitur dari state terbaru
        df_temp   = create_lag_features(df_rolling, target, LOOK_BACK)
        if df_temp.empty:
            break
        X_fore    = df_temp[X_cols].iloc[-1:]

        # Prediksi satu langkah ke depan
        next_val = model.predict(X_fore)[0]
        # Daya motor tidak bisa negatif secara fisik — clip ke 0
        if target == 'joint_P_total':
            next_val = max(0.0, next_val)
        forecast_vals.append(next_val)

        # Update rolling dataframe
        new_row   = df_rolling.iloc[-1:].copy()
        new_row.index = new_row.index + pd.Timedelta(seconds=1)
        new_row[target] = next_val
        df_rolling = pd.concat([df_rolling, new_row])

    # Buat Series forecast
    last_time     = df.index[-1]
    future_idx    = pd.date_range(
        start=last_time + pd.Timedelta(seconds=1),
        periods=len(forecast_vals), freq='1s')
    forecast_series = pd.Series(forecast_vals, index=future_idx)

    print(f"  Nilai prediksi akhir: {forecast_vals[-1]:.4f} {unit}")

    return {
        'model'      : model,
        'importance' : importance,
        'df'         : df,
        'df_feat'    : df_feat,
        'split_idx'  : split_idx,
        'X_cols'     : X_cols,
        'y_test'     : y_test,
        'pred_test'  : pred_test,
        'forecast'   : forecast_series,
        'metrics'    : {
            'mae': mae, 'rmse': rmse, 'smape': smape
        }
    }

def plot_results(result: dict, config: dict, output_path: str):
    df         = result['df']
    importance = result['importance']
    split_idx  = result['split_idx']
    y_test     = result['y_test']
    pred_test  = result['pred_test']
    forecast   = result['forecast']
    metrics    = result['metrics']
    target     = config['field']
    label      = config['label']
    unit       = config['unit']

    # Warna standar akademis
    COLOR_IMP      = '#2166AC'   # biru tua  feature importance bar
    COLOR_ACTUAL   = '#1A9641'   # hijau tua data aktual
    COLOR_PRED     = '#D7191C'   # merah     prediksi XGBoost
    COLOR_TRAIN    = '#999999'   # abu       training line
    COLOR_FORECAST = '#F46D43'   # oranye    forecast
    COLOR_SCATTER  = '#2166AC'   # biru      scatter aktual
    COLOR_DIAG     = '#D7191C'   # merah     garis diagonal

    # Setup style jurnal
    plt.rcParams.update({
        'font.family'      : 'serif',
        'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
        'font.size'        : 10,
        'axes.titlesize'   : 11,
        'axes.labelsize'   : 10,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
        'legend.fontsize'  : 9,
        'figure.dpi'       : 300,
        'savefig.dpi'      : 300,
        'axes.linewidth'   : 0.8,
        'grid.linewidth'   : 0.5,
        'grid.alpha'       : 0.4,
        'grid.color'       : '#CCCCCC',
        'axes.grid'        : True,
        'axes.facecolor'   : 'white',
        'figure.facecolor' : 'white',
        'axes.edgecolor'   : '#333333',
    })

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    # Panel 1: Feature Importance (horizontal bar)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(
        range(len(importance)),
        importance.values,
        color=COLOR_IMP, alpha=0.80,
        edgecolor='white', linewidth=0.4
    )
    ax1.set_yticks(range(len(importance)))
    ax1.set_yticklabels(
        [f[:28] for f in importance.index], fontsize=8)
    ax1.set_title('Top 10 Feature Importance',
                  fontsize=10, fontweight='bold', pad=6)
    ax1.set_xlabel('Importance Score', fontsize=9)
    ax1.invert_yaxis()

    # Label nilai di ujung bar
    for i, (bar, val) in enumerate(zip(bars, importance.values)):
        ax1.text(val + max(importance.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}',
                 va='center', ha='left', fontsize=7.5,
                 color='#333333')

    # Panel 2: Scatter prediksi vs aktual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_test.values, pred_test,
                color=COLOR_SCATTER, s=6, alpha=0.45,
                linewidths=0, label='Predicted vs. Actual')

    # Garis diagonal sempurna
    lims = [min(y_test.min(), pred_test.min()),
            max(y_test.max(), pred_test.max())]
    ax2.plot(lims, lims, color=COLOR_DIAG,
             linewidth=1.8, linestyle='--',
             label='Perfect prediction', alpha=0.8)

    ax2.set_title(
        f'Predicted vs. Actual (Test Set)\n'
        f'MAE = {metrics["mae"]:.3f}  |  '
        f'RMSE = {metrics["rmse"]:.3f}  |  '
        f'sMAPE = {metrics["smape"]:.1f}%',
        fontsize=9.5, fontweight='bold', pad=6)
    ax2.set_xlabel(f'Actual ({unit})', fontsize=9)
    ax2.set_ylabel(f'Predicted ({unit})', fontsize=9)
    ax2.legend(frameon=True, framealpha=0.9,
               edgecolor='#CCCCCC', fontsize=8)

    # Panel 3: Data historis + test + forecast
    ax3 = fig.add_subplot(gs[1, :])

    train_series = df[target].iloc[:split_idx]
    test_series  = df[target].iloc[split_idx:]

    ax3.plot(train_series.index, train_series.values,
             color=COLOR_TRAIN, linewidth=1.0, alpha=0.9,
             label='Training data')
    ax3.plot(test_series.index, test_series.values,
             color=COLOR_ACTUAL, linewidth=1.6,
             label='Actual (test)')
    ax3.plot(forecast.index, forecast.values,
             color=COLOR_FORECAST, linewidth=1.8,
             label=f'Forecast (+{FORECAST_N} s)')

    # Garis pemisah vertikal
    ax3.axvline(x=test_series.index[0], color='#444444',
                linewidth=0.8, linestyle=':', alpha=0.7)
    ax3.axvline(x=forecast.index[0], color=COLOR_FORECAST,
                linewidth=0.8, linestyle=':', alpha=0.7)

    ax3.set_title(
        f'XGBoost — Historical Data and Forecast  |  '
        f'MAE = {metrics["mae"]:.3f} {unit}  |  '
        f'sMAPE = {metrics["smape"]:.1f}%',
        fontsize=10, fontweight='bold', pad=6)
    ax3.set_ylabel(unit, fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.legend(frameon=True, framealpha=0.9,
               edgecolor='#CCCCCC', loc='upper left', ncol=2)

    # Judul utama
    fig.suptitle(
        f'XGBoost Predictive Analytics — {label}\n',
        fontsize=12, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

    # Reset rcParams ke default agar tidak mempengaruhi plot lain
    plt.rcdefaults()
    print(f"  Plot tersimpan: {output_path}")


def save_summary(results: list, output_dir: str):
    """Simpan ringkasan hasil."""
    txt_path = os.path.join(output_dir, 'xgboost_hw_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HASIL XGBoost — Sistem Data Historian Polebot AMR\n")
        f.write(f"Dijalankan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameter : n_estimators={XGB_PARAMS['n_estimators']}, "
                f"max_depth={XGB_PARAMS['max_depth']}, "
                f"lr={XGB_PARAMS['learning_rate']}\n")
        f.write(f"Look back : {LOOK_BACK} detik\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"Variabel : {r['label']}\n")
            f.write(f"Field    : {r['field']}\n")
            f.write(f"Data pts : {r['n_data']}\n")
            f.write(f"MAE      : {r['mae']:.4f}\n")
            f.write(f"RMSE     : {r['rmse']:.4f}\n")
            f.write(f"sMAPE    : {r['smape']:.2f}%\n")
            f.write(f"Top feat : {r['top_feature']}\n")
            f.write("-" * 40 + "\n\n")

    print(f"\n  Ringkasan: {txt_path}")

    json_path = os.path.join(output_dir, 'xgboost_hw_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ENTRY POINT
def main():
    print("=" * 60)
    print("  XGBoost Predictor — Data Historian Polebot AMR")
    print("=" * 60)
    print(f"  Look back  : {LOOK_BACK} detik")
    print(f"  Forecast   : {FORECAST_N} detik ke depan")
    print(f"  Estimators : {XGB_PARAMS['n_estimators']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")

    # Ambil semua data sekaligus (lebih efisien)
    df = fetch_all_data()
    if df is None or len(df) < 50:
        print("Data tidak cukup.")
        return

    summary_results = []

    for config in TARGET_VARIABLES:
        field = config['field']
        label = config['label']

        if field not in df.columns:
            print(f"\n Field '{field}' tidak ditemukan, dilewati.")
            continue

        print(f"\n{'═'*60}")
        print(f"  Memproses: {label}")
        print(f"{'═'*60}")

        result = train_xgboost(df, config)
        if result is None:
            continue

        # Plot
        plot_path = os.path.join(OUTPUT_DIR, f'xgboost_hw_{field}.png')
        plot_results(result, config, plot_path)

        # Simpan forecast CSV
        csv_path = os.path.join(OUTPUT_DIR, f'xgboost_hw_forecast_{field}.csv')
        forecast_df = pd.DataFrame({
            'timestamp'     : result['forecast'].index,
            'forecast_value': result['forecast'].values,
            'field'         : field,
            'unit'          : config['unit'],
            'model'         : 'XGBoost',
        })
        forecast_df.to_csv(csv_path, index=False)

        metrics = result['metrics']
        summary_results.append({
            'field'      : field,
            'label'      : label,
            'n_data'     : len(df),
            'mae'        : metrics['mae'],
            'rmse'       : metrics['rmse'],
            'smape'      : metrics['smape'],
            'top_feature': result['importance'].index[0],
        })

    if summary_results:
        save_summary(summary_results, OUTPUT_DIR)

        print("  SELESAI - Ringkasan Hasil XGBoost")
        print(f"{'Variabel':<30} {'sMAPE':>8} {'RMSE':>8} {'Top Feature'}")
        for r in summary_results:
            print(f"{r['label'][:30]:<30} "
                  f"{r['smape']:>7.1f}%  "
                  f"{r['rmse']:>8.4f}  "
                  f"{r['top_feature'][:25]}")
        print(f"\nSemua file tersimpan di: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()