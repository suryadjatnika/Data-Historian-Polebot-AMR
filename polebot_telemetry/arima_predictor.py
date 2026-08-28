#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend untuk server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from influxdb_client import InfluxDBClient
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import json
from datetime import datetime

# KONFIGURASI
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_data"

# Rentang data yang diambil
DATA_RANGE      = "start: 2026-05-31T18:29:24Z, stop: 2026-05-31T21:31:55Z"

# Folder output hasil ARIMA
OUTPUT_DIR      = os.path.expanduser("~/polebot_arima_results")

# Variabel yang akan dianalisis ARIMA
TARGET_VARIABLES = [
    {
        'field'      : 'joint_P_total',
        'label'      : 'Total Motor Power (P_total)',
        'unit'       : 'Watt',
        'color'      : '#c084fc',
        'resample'   : '1s',    # resample ke 1 detik
        'forecast_n' : 60,      # prediksi 60 detik ke depan
    },
    {
        'field'      : 'batt_soc_percent',
        'label'      : 'Battery State of Charge (SOC)',
        'unit'       : '%',
        'color'      : '#22c55e',
        'resample'   : '1s',
        'forecast_n' : 60,
    },
    {
        'field'      : 'odom_v_linear',
        'label'      : 'Linear Velocity',
        'unit'       : 'm/s',
        'color'      : '#fb923c',
        'resample'   : '1s',
        'forecast_n' : 60,
    },
]

# FUNGSI UTAMA
def fetch_data(field: str, resample: str) -> pd.Series:
    """
    Ambil data dari InfluxDB dan kembalikan sebagai pandas Series.
    """
    print(f"\n  Mengambil data '{field}' dari InfluxDB...")

    client = InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => r._field == "{field}")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> sort(columns: ["_time"])
'''

    try:
        df = client.query_api().query_data_frame(query)
        client.close()

        # Handle list of DataFrames (InfluxDB kadang kembalikan multiple tables)
        if isinstance(df, list):
            if len(df) == 0:
                print(f"  ⚠️  Data '{field}' kosong!")
                return None
            df = pd.concat(df, ignore_index=True)

        if df.empty:
            print(f"  ⚠️  Data '{field}' kosong!")
            return None

        # Bersihkan kolom yang tidak diperlukan
        df = df[['_time', '_value']].copy()
        # Fix dtype str: pastikan _value selalu numerik
        df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
        df = df.dropna(subset=['_value'])
        df['_time'] = pd.to_datetime(df['_time'], utc=True)
        df = df.set_index('_time')
        df.index = df.index.tz_convert('Asia/Jakarta').tz_localize(None)  # WIB
        # Resample ke interval yang konsisten
        series = df['_value'].resample(resample).mean().dropna()
        if any(k in field for k in ['P_total', 'v_linear', 'omega', 'accel']):
            before = len(series)
            series = series[series.abs() > 0.01]
            print(f'  Filter nol: {before} -> {len(series)} titik')

        # Filter kondisi statis untuk SOC baterai
        # ARIMA hanya dilatih saat robot idle atau gerak konstan
        if 'soc' in field or 'batt' in field:
            # Ambil data akselerasi dan kecepatan sebagai filter
            fields_dyn = ' or '.join([
                'r._field == "odom_accel"',
                'r._field == "odom_v_linear"'
            ])
            query_dyn = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => {fields_dyn})
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns: ["_time"])
'''
            client2 = InfluxDBClient(
                url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            df_dyn = client2.query_api().query_data_frame(query_dyn)
            client2.close()

            # Handle list of DataFrames
            if isinstance(df_dyn, list):
                df_dyn = pd.concat(df_dyn, ignore_index=True) if df_dyn else pd.DataFrame()

            if not df_dyn.empty:
                df_dyn['_time'] = pd.to_datetime(df_dyn['_time'], utc=True)
                df_dyn = df_dyn.set_index('_time')
                df_dyn.index = df_dyn.index.tz_convert('Asia/Jakarta').tz_localize(None)                
                # Pilih HANYA kolom yang dibutuhkan sebelum resample
                # (kolom metadata InfluxDB bertipe str akan crash di .mean())
                _cols_needed = [c for c in ['odom_accel', 'odom_v_linear']
                                if c in df_dyn.columns]
                if _cols_needed:
                    df_dyn = df_dyn[_cols_needed]
                    for _c in _cols_needed:
                        df_dyn[_c] = pd.to_numeric(df_dyn[_c], errors='coerce')
                df_dyn = df_dyn.resample(resample).mean(numeric_only=True)

                # Kondisi statis: akselerasi kecil DAN kecepatan tidak tinggi
                mask_static = (
                    df_dyn['odom_accel'].abs() < 0.15
                ) & (
                    df_dyn['odom_v_linear'].abs() < 0.6
                )
                # Align index dengan series SOC
                mask_static = mask_static.reindex(series.index).fillna(False)
                before = len(series)
                series = series[mask_static]
                print(f'  Filter statis (ARIMA): {before} → {len(series)} titik')

        print(f"  ✅ {len(series)} titik data ({series.index[0].strftime('%H:%M:%S')} "
              f"→ {series.index[-1].strftime('%H:%M:%S')} WIB)")
        return series

    except Exception as e:
        print(f"  ❌ Gagal ambil data: {e}")
        return None


def adf_test(series: pd.Series, label: str) -> dict:
    """
    Augmented Dickey-Fuller Test untuk cek stasioneritas.
    Menentukan parameter d untuk ARIMA.
    """
    print(f"\n  ADF Test untuk '{label}'...")

    result = adfuller(series.dropna(), autolag='AIC')

    adf_stat  = result[0]
    p_value   = result[1]
    n_lags    = result[2]
    n_obs     = result[3]
    crit_vals = result[4]

    is_stationary = p_value < 0.05

    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Stasioner     : {'Ya ✅ (d=0)' if is_stationary else 'Tidak ⚠️ (perlu differencing, d≥1)'}")

    return {
        'adf_stat'      : adf_stat,
        'p_value'       : p_value,
        'is_stationary' : is_stationary,
        'd_param'       : 0 if is_stationary else 1,
        'critical_vals' : crit_vals,
    }


def determine_params(series: pd.Series, d: int) -> tuple:
    """
    Tentukan parameter p dan q dari plot ACF/PACF.
    Untuk simplifikasi, gunakan auto-selection dengan AIC.
    """
    print(f"\n  Menentukan parameter p dan q (auto-selection)...")

    best_aic = np.inf
    best_p   = 1
    best_q   = 1

    # Grid search p dan q (0-3) × (0-3)
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(series, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_p   = p
                    best_q   = q
            except Exception:
                continue

    print(f"  Parameter terpilih: p={best_p}, d={d}, q={best_q} (AIC={best_aic:.2f})")
    return best_p, best_q

def train_arima(series: pd.Series, p: int, d: int, q: int):
    """
    Latih model ARIMA dengan parameter yang sudah ditentukan.
    """
    print(f"\n  Melatih model ARIMA({p},{d},{q})...")

    # Split data: 80% train, 20% test
    split_idx  = int(len(series) * 0.8)
    train_data = series[:split_idx]
    test_data  = series[split_idx:]

    print(f"  Data train : {len(train_data)} titik")
    print(f"  Data test  : {len(test_data)} titik")

    # Latih model
    model  = ARIMA(train_data, order=(p, d, q))
    fitted = model.fit()

    # Prediksi pada data test (in-sample)
    test_pred = pd.Series(
        fitted.forecast(steps=len(test_data)).values,
        index=test_data.index
    )

    # Hitung metrik evaluasi
    mae  = np.mean(np.abs(test_data.values - test_pred.values))
    rmse = np.sqrt(np.mean((test_data.values - test_pred.values) ** 2))
    mape = mape = np.mean(
        2 * np.abs(test_data.values - test_pred.values) /
        (np.abs(test_data.values) + np.abs(test_pred.values) + 1e-10)
    ) * 100
    # sMAPE — format sama dengan XGBoost agar comparison_plot bisa baca
    smape = mape

    print(f"\n  ── Hasil Evaluasi ──────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  sMAPE: {mape:.2f}%")
    print(f"  AIC  : {fitted.aic:.2f}")
    print(f"  ────────────────────────────────────────")

    return fitted, train_data, test_data, test_pred, {
        'mae' : mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'aic': fitted.aic
    }

def forecast_future(fitted_model, series: pd.Series,
                    n_steps: int, unit: str):
    """
    Prediksi n_steps ke depan dari akhir data.
    """
    print(f"\n  Memprediksi {n_steps} detik ke depan...")

    # Retrain dengan seluruh data untuk forecast masa depan
    p, d, q = fitted_model.model.order
    full_model  = ARIMA(series, order=(p, d, q))
    full_fitted = full_model.fit()

    forecast_result = full_fitted.get_forecast(steps=n_steps)
    forecast_mean   = forecast_result.predicted_mean
    forecast_ci     = forecast_result.conf_int(alpha=0.05)

    # Buat index waktu untuk forecast
    last_time   = series.index[-1]
    freq        = series.index.freq or pd.tseries.frequencies.to_offset('1s')
    future_idx  = pd.date_range(
        start=last_time + freq, periods=n_steps, freq=freq)

    forecast_mean.index = future_idx
    forecast_ci.index   = future_idx

    print(f"  Nilai prediksi terakhir: "
          f"{forecast_mean.iloc[-1]:.4f} {unit}")

    return forecast_mean, forecast_ci

def plot_results(series: pd.Series, train: pd.Series,
                 test: pd.Series, test_pred: pd.Series,
                 forecast: pd.Series, forecast_ci: pd.DataFrame,
                 adf_result: dict, metrics: dict,
                 config: dict, output_path: str):

    field = config['field']
    label = config['label']
    unit  = config['unit']

    # ── Warna standar akademis ───────────────────────────────────
    COLOR_HIST     = '#2166AC'   # biru tua  — data historis
    COLOR_TRAIN    = '#999999'   # abu       — training line
    COLOR_ACTUAL   = '#1A9641'   # hijau tua — data aktual test
    COLOR_PRED     = '#D7191C'   # merah     — prediksi ARIMA
    COLOR_FORECAST = '#F46D43'   # oranye    — forecast
    COLOR_CI       = '#FDAE61'   # kuning    — confidence interval

    # ── Setup style jurnal ───────────────────────────────────────
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
    gs  = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)

    p = adf_result.get('p', 1)
    d = adf_result.get('d_param', 0)
    q = adf_result.get('q', 1)

    # ── Panel 1: Data historis lengkap ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series.index, series.values,
             color=COLOR_HIST, linewidth=0.7, alpha=0.85,
             label='Historical data')
    ax1.set_title(
        f'{label} — Historical Data ({len(series)} points)',
        fontsize=11, fontweight='bold', pad=6)
    ax1.set_ylabel(unit, fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC')

    # ── Panel 2: ACF ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(series.dropna(), ax=ax2, lags=40,
             color=COLOR_HIST, alpha=0.05, zero=False,
             title='')
    ax2.set_title('ACF — Autocorrelation Function',
                  fontsize=10, fontweight='bold', pad=4)
    ax2.set_xlabel('Lag', fontsize=9)
    ax2.set_ylabel('Autocorrelation', fontsize=9)

    # ── Panel 3: PACF ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(series.dropna(), ax=ax3, lags=40,
              color=COLOR_HIST, alpha=0.05, zero=False,
              method='ywm', title='')
    ax3.set_title('PACF — Partial Autocorrelation Function',
                  fontsize=10, fontweight='bold', pad=4)
    ax3.set_xlabel('Lag', fontsize=9)
    ax3.set_ylabel('Partial Autocorrelation', fontsize=9)

    # ── Panel 4: Fitting + Forecast ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])

    ax4.plot(train.index, train.values,
             color=COLOR_TRAIN, linewidth=0.6, alpha=0.7,
             label='Training data')
    ax4.plot(test.index, test.values,
             color=COLOR_ACTUAL, linewidth=1.0,
             label='Actual (test)')
    ax4.plot(test.index, test_pred.values,
             color=COLOR_PRED, linewidth=1.0,
             linestyle='--', label='ARIMA prediction (test)')
    ax4.plot(forecast.index, forecast.values,
             color=COLOR_FORECAST, linewidth=1.2,
             linestyle='-', label=f'Forecast (+{len(forecast)} s)')
    ax4.fill_between(
        forecast.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color=COLOR_CI, alpha=0.25,
        label='95% Confidence Interval')

    # Garis pemisah vertikal
    ax4.axvline(x=test.index[0], color='#444444',
                linewidth=0.8, linestyle=':', alpha=0.7)
    ax4.axvline(x=forecast.index[0], color=COLOR_FORECAST,
                linewidth=0.8, linestyle=':', alpha=0.7)

    ax4.set_title(
        f'ARIMA({p},{d},{q}) — Model Fitting and Forecast  |  '
        f'MAE = {metrics["mae"]:.3f}  |  '
        f'RMSE = {metrics["rmse"]:.3f}  |  '
        f'sMAPE = {metrics["smape"]:.1f}%',
        fontsize=10, fontweight='bold', pad=6)
    ax4.set_ylabel(unit, fontsize=10)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
               loc='upper left', ncol=2)

    # ── Judul utama ───────────────────────────────────────────────
    fig.suptitle(
        f'ARIMA Predictive Analytics — {label}\n',
        fontsize=12, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

    # Reset rcParams ke default agar tidak mempengaruhi plot lain
    plt.rcdefaults()
    print(f"  Plot tersimpan: {output_path}")


def save_summary(results: list, output_dir: str):
    """Simpan ringkasan semua hasil ke file JSON dan TXT."""

    # JSON
    json_path = os.path.join(output_dir, 'arima_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # TXT — ringkasan human-readable
    txt_path = os.path.join(output_dir, 'arima_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HASIL ARIMA — Sistem Data Historian Polebot AMR\n")
        f.write(f"Dijalankan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"Variabel : {r['label']}\n")
            f.write(f"Field    : {r['field']}\n")
            f.write(f"Model    : ARIMA({r['p']},{r['d']},{r['q']})\n")
            f.write(f"Data pts : {r['n_data']}\n")
            f.write(f"Stasioner: {'Ya' if r['is_stationary'] else 'Tidak'} "
                    f"(p-value={r['adf_pvalue']:.4f})\n")
            f.write(f"MAE      : {r['mae']:.4f}\n")
            f.write(f"RMSE     : {r['rmse']:.4f}\n")
            f.write(f"sMAPE    : {r['smape']:.2f}%\n")
            f.write(f"AIC      : {r['aic']:.2f}\n")
            f.write("-" * 40 + "\n\n")

    print(f"\n  Ringkasan tersimpan di: {txt_path}")

# ENTRY POINT
def main():
    print("=" * 60)
    print("  ARIMA Predictor — Data Historian Polebot AMR")
    print("=" * 60)

    # Buat folder output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput akan disimpan di: {OUTPUT_DIR}")

    summary_results = []

    for config in TARGET_VARIABLES:
        field    = config['field']
        label    = config['label']
        unit     = config['unit']
        resample = config['resample']
        n_fc     = config['forecast_n']

        print(f"\n{'═'*60}")
        print(f"  Memproses: {label}")
        print(f"{'═'*60}")

        # 1. Ambil data
        series = fetch_data(field, resample)
        if series is None or len(series) < 50:
            print(f"  ⚠️  Data tidak cukup untuk {field}, dilewati.")
            continue

        # 2. ADF Test
        adf_result = adf_test(series, label)
        d = adf_result['d_param']
        if field == 'batt_soc_percent' and d == 0:
            print("  ⚠️  Override d=0→1 untuk SOC (non-stasioner fisik, ADF mungkin keliru)")
            d = 1
            adf_result['d_param']     = 1
            adf_result['is_stationary'] = False

        # 3. Tentukan p dan q 
        p, q = determine_params(series, d)

        # 4. Latih ARIMA
        fitted, train, test, test_pred, metrics = \
            train_arima(series, p, d, q)

        # Simpan p, q ke adf_result untuk referensi plot
        adf_result['p'] = p
        adf_result['q'] = q

        # 5. Forecast
        forecast, forecast_ci = forecast_future(fitted, series, n_fc, unit)

        # 6. Plot
        plot_path = os.path.join(OUTPUT_DIR, f'arima_{field}.png')
        plot_results(
            series, train, test, test_pred,
            forecast, forecast_ci,
            adf_result, metrics, config, plot_path)

        # 7. Simpan forecast values
        forecast_csv = os.path.join(OUTPUT_DIR, f'forecast_{field}.csv')
        forecast_df  = pd.DataFrame({
            'timestamp'       : forecast.index,
            'forecast_value'  : forecast.values,
            'ci_lower'        : forecast_ci.iloc[:, 0].values,
            'ci_upper'        : forecast_ci.iloc[:, 1].values,
            'field'           : field,
            'unit'            : unit,
        })
        forecast_df.to_csv(forecast_csv, index=False)
        print(f"  Forecast CSV: {forecast_csv}")

        # 8. Kumpulkan hasil
        summary_results.append({
            'field'         : field,
            'label'         : label,
            'n_data'        : len(series),
            'p'             : p,
            'd'             : d,
            'q'             : q,
            'is_stationary' : adf_result['is_stationary'],
            'adf_pvalue'    : adf_result['p_value'],
            'mae'           : metrics['mae'],
            'rmse'          : metrics['rmse'],
            'mape'          : metrics['mape'],
            'smape'         : metrics.get('smape', metrics['mape']),
            'aic'           : metrics['aic'],
            'forecast_last' : float(forecast.iloc[-1]),
            'forecast_unit' : unit,
        })

    # Simpan ringkasan
    if summary_results:
        save_summary(summary_results, OUTPUT_DIR)

        print(f"\n{'═'*60}")
        print("  SELESAI — Ringkasan Hasil")
        print(f"{'═'*60}")
        print(f"{'Variabel':<30} {'ARIMA':^12} {'sMAPE':^8} {'RMSE':^8}")
        print("-" * 60)
        for r in summary_results:
            order = f"({r['p']},{r['d']},{r['q']})"
            print(f"{r['label'][:30]:<30} {order:^12} "
                  f"{r['smape']:>6.1f}%  {r['rmse']:>8.4f}")
        print(f"\nSemua file tersimpan di: {OUTPUT_DIR}")
    else:
        print("\n⚠️  Tidak ada variabel yang berhasil diproses.")


if __name__ == '__main__':
    main()