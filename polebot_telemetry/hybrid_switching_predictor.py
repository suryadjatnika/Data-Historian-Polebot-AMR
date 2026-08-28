#!/usr/bin/env python3
"""
hybrid_switching_predictor.py
══════════════════════════════════════════════════════════════════════════
Condition-Based Temporal Switching — Polebot AMR
Politeknik Manufaktur Bandung · 2025/2026 · Surya Dharma Jatnika

MEKANISME:
  Di setiap timestep t, sistem menentukan kondisi operasional:
    STATIS  → |a(t)| < 0.15 m/s²  DAN  |v(t)| < 0.6 m/s
    DINAMIS → selain itu

  Kemudian memilih model yang sesuai:
    STATIS  → prediksi dari model ARIMA  (tren linear SOC)
    DINAMIS → prediksi dari model XGBoost (multi-sensor non-linear)

  Hasilnya digabungkan menjadi satu deret prediksi hybrid kontinu.

OUTPUT (folder ~/polebot_hybrid_results/):
  1. hybrid_switching_chart.png  — 3-panel chart dengan switching visual
  2. hybrid_switching_metrics.json — MAE/RMSE/sMAPE per variabel
══════════════════════════════════════════════════════════════════════════
"""
import warnings
warnings.filterwarnings('ignore')

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

from influxdb_client import InfluxDBClient
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════
# KONFIGURASI  ← ubah DATA_RANGE setelah simulasi C7 selesai
# ══════════════════════════════════════════════════════════════════════════
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_data"

# ▼▼▼  ISI INI SETELAH SIMULASI C7 SELESAI  ▼▼▼
# Format: "start: YYYY-MM-DDTHH:MM:SSZ, stop: YYYY-MM-DDTHH:MM:SSZ"
# Contoh: simulasi mulai 02:15:00 WIB → UTC 19:15:00 hari sebelumnya
DATA_RANGE_C7   = "start: 2026-06-10T17:01:00Z, stop: 2026-06-10T17:31:03Z"
# ▲▲▲ ──────────────────────────────────────────────────────────────────

OUTPUT_DIR      = os.path.expanduser("~/polebot_hybrid_results")

# ── Parameter filter kondisi (sama persis dengan classifier di predictor lain)
ACCEL_THRESHOLD = 0.15   # m/s²  — batas maksimum akselerasi untuk statis
SPEED_THRESHOLD = 0.60   # m/s   — batas maksimum kecepatan untuk statis

# ── Parameter model
LOOK_BACK       = 10     # XGBoost: jumlah lag timestep
TRAIN_RATIO     = 0.80   # rasio data pelatihan (80%)
XGB_PARAMS = {
    'n_estimators'    : 500,
    'max_depth'       : 6,
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'random_state'    : 42,
    'n_jobs'          : -1,
}

# ── Semua field yang diambil dari InfluxDB
ALL_FIELDS = [
    'joint_P_total',
    'batt_soc_percent',
    'odom_v_linear',
    'odom_accel',
    'odom_omega',
    'joint_load_ratio',
    'batt_power_draw',
]

# ── 3 variabel target yang ditampilkan di chart
TARGETS = [
    {'field': 'batt_soc_percent', 'label': 'State of Charge Baterai (SOC)', 'unit': '%'},
    {'field': 'joint_P_total',    'label': 'Daya Motor Total (P_total)',      'unit': 'Watt'},
    {'field': 'odom_v_linear',    'label': 'Kecepatan Linear Robot',          'unit': 'm/s'},
]

# ── Warna chart
COLOR_STATIC_LINE  = '#003087'   # biru navy tua  — data aktual segmen STATIS
COLOR_DYNAMIC_LINE = '#CC0000'   # merah          — data aktual segmen DINAMIS
COLOR_HYBRID_PRED  = '#00A651'   # hijau          — prediksi HYBRID
COLOR_STATIC_BG    = '#D6E8FF'   # biru muda      — background region statis
COLOR_DYNAMIC_BG   = '#FFD6D6'   # merah muda     — background region dinamis
LW_DATA            = 1.2         # linewidth data aktual (standar)
LW_PRED            = 1.8         # linewidth prediksi
LW_PRED_DASH       = (6, 2)      # dashes pattern prediksi


# ══════════════════════════════════════════════════════════════════════════
# 1. AMBIL DATA DARI INFLUXDB
# ══════════════════════════════════════════════════════════════════════════
def fetch_all_data() -> pd.DataFrame:
    """Ambil semua field dari C7 dan kembalikan sebagai DataFrame wide."""
    print("\n" + "═"*60)
    print("  MENGAMBIL DATA C7 DARI INFLUXDB")
    print("═"*60)
    print(f"  Range: {DATA_RANGE_C7}")

    client = InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    fields_filter = ' or '.join([f'r._field == "{f}"' for f in ALL_FIELDS])
    query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE_C7})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => {fields_filter})
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns: ["_time"])
'''

    df = client.query_api().query_data_frame(query)
    client.close()

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()

    if df.empty:
        raise ValueError("❌ Data C7 kosong! Periksa DATA_RANGE_C7 dan pastikan "
                         "simulasi C7 sudah dijalankan.")

    # Bersihkan kolom InfluxDB
    drop_cols = [c for c in df.columns
                 if c.startswith('_') and c not in ['_time']]
    df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore',
            inplace=True)

    # Atur index waktu
    df['_time'] = pd.to_datetime(df['_time'], utc=True)
    df.index = df['_time'].dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    df.drop(columns=['_time'], inplace=True, errors='ignore')

    # Konversi numerik dan interpolasi gap kecil
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.select_dtypes(include=['number'])
    df = df.interpolate(method='time', limit=5).ffill().bfill()

    # Pastikan odom_accel ada
    if 'odom_accel' not in df.columns:
        df['odom_accel'] = df['odom_v_linear'].diff().fillna(0.0)

    print(f"  ✅ Data berhasil diambil: {len(df)} rekaman")
    print(f"  Periode: {df.index[0].strftime('%H:%M:%S')} — "
          f"{df.index[-1].strftime('%H:%M:%S')}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. KLASIFIKASI KONDISI PER TIMESTEP
# ══════════════════════════════════════════════════════════════════════════
def classify_conditions(df: pd.DataFrame) -> pd.Series:
    """
    Kembalikan Series boolean: True = STATIS, False = DINAMIS
    STATIS  = |odom_accel| < ACCEL_THRESHOLD  DAN  |odom_v_linear| < SPEED_THRESHOLD
    DINAMIS = sebaliknya
    """
    accel = df['odom_accel'].abs()
    speed = df['odom_v_linear'].abs()

    is_static = (accel < ACCEL_THRESHOLD) & (speed < SPEED_THRESHOLD)

    n_static  = is_static.sum()
    n_dynamic = (~is_static).sum()
    pct_static = n_static / len(is_static) * 100

    print(f"\n  Klasifikasi kondisi:")
    print(f"    STATIS  : {n_static:5d} titik ({pct_static:.1f}%)")
    print(f"    DINAMIS : {n_dynamic:5d} titik ({100-pct_static:.1f}%)")
    return is_static


def count_transitions(is_static: pd.Series) -> int:
    """Hitung berapa kali kondisi berganti STATIS↔DINAMIS."""
    changes = (is_static != is_static.shift()).sum()
    print(f"    Jumlah transisi STATIS↔DINAMIS : {changes}")
    return changes


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL ARIMA — FIT DAN PREDIKSI
# ══════════════════════════════════════════════════════════════════════════
def fit_arima_model(series: pd.Series, label: str):
    """
    Fit ARIMA pada seluruh series.
    Kembalikan fittedvalues + forecast dictionary.
    """
    print(f"\n  [ARIMA] Fitting model untuk: {label}")

    # Uji ADF
    adf_stat, pvalue, *_ = adfuller(series.dropna(), autolag='AIC')
    d = 0 if pvalue < 0.05 else 1

    # Override fisik untuk SOC
    if 'soc' in label.lower() or 'batt' in label.lower():
        d = 1

    # Grid search AIC — p,q ∈ {0,1,2,3}
    best_aic  = np.inf
    best_pq   = (1, 1)
    for p in range(4):
        for q in range(4):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_pq  = (p, q)
            except Exception:
                continue

    p, q = best_pq
    print(f"    Parameter terpilih: ARIMA({p},{d},{q}), AIC={best_aic:.2f}")

    model = ARIMA(series, order=(p, d, q)).fit()

    # Fitted values (in-sample)
    fitted = model.fittedvalues
    fitted = fitted.reindex(series.index).bfill().ffill()

    return {
        'model'    : model,
        'fitted'   : fitted,
        'order'    : (p, d, q),
        'aic'      : best_aic,
        'adf_pval' : pvalue,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. MODEL XGBOOST — FIT DAN PREDIKSI
# ══════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, target_field: str,
                   look_back: int = LOOK_BACK) -> tuple:
    """
    Bangun matriks fitur dan vektor target untuk XGBoost.
    Fitur: lag 1–10 target + lag 1–3 semua sensor lain + rolling mean/std.
    """
    feat_df = pd.DataFrame(index=df.index)
    target  = df[target_field].copy()

    # Self-lag target
    for lag in range(1, look_back + 1):
        feat_df[f'{target_field}_lag{lag}'] = target.shift(lag)

    # Fitur sensor lain (lag 1–3)
    for col in ALL_FIELDS:
        if col == target_field or col not in df.columns:
            continue
        for lag in range(1, 4):
            feat_df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Rolling statistik target (window 5)
    feat_df[f'{target_field}_roll_mean5'] = \
        target.rolling(5, min_periods=1).mean().shift(1)
    feat_df[f'{target_field}_roll_std5']  = \
        target.rolling(5, min_periods=1).std().shift(1).fillna(0)

    # Kondisi dinamis sebagai fitur
    accel = df['odom_accel'].abs()
    speed = df['odom_v_linear'].abs()
    feat_df['is_dynamic'] = (
        (accel >= ACCEL_THRESHOLD) | (speed >= SPEED_THRESHOLD)
    ).astype(float)

    # Hapus baris dengan NaN
    valid_mask = feat_df.notna().all(axis=1) & target.notna()
    X = feat_df[valid_mask].values
    y = target[valid_mask].values
    idx = feat_df.index[valid_mask]

    return X, y, idx


def fit_xgboost_model(df: pd.DataFrame, target_field: str, label: str):
    """Fit XGBoost, kembalikan prediksi pada seluruh valid index."""
    print(f"\n  [XGBoost] Fitting model untuk: {label}")

    X, y, idx = build_features(df, target_field)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    preds = model.predict(X)
    pred_series = pd.Series(preds, index=idx)

    print(f"    ✅ XGBoost selesai — {len(pred_series)} prediksi")
    return {
        'model'      : model,
        'predictions': pred_series,
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. STITCH — GABUNGKAN PREDIKSI BERDASARKAN KONDISI
# ══════════════════════════════════════════════════════════════════════════
def stitch_predictions(arima_res: dict, xgb_res: dict,
                       is_static: pd.Series,
                       target_series: pd.Series) -> pd.Series:
    """
    Di setiap timestep yang tersedia:
      STATIS  → gunakan prediksi ARIMA
      DINAMIS → gunakan prediksi XGBoost
    Kembalikan Series hybrid yang sudah di-stitch.
    """
    arima_pred = arima_res['fitted']
    xgb_pred   = xgb_res['predictions']

    # Tentukan index bersama (hanya timestep yang punya keduanya)
    common_idx = arima_pred.index.intersection(xgb_pred.index)
    common_idx = common_idx.intersection(is_static.index)

    hybrid = pd.Series(index=common_idx, dtype=float)
    cond   = is_static.reindex(common_idx)

    hybrid[cond]  = arima_pred.reindex(common_idx)[cond]
    hybrid[~cond] = xgb_pred.reindex(common_idx)[~cond]

    return hybrid


# ══════════════════════════════════════════════════════════════════════════
# 6. HITUNG METRIK
# ══════════════════════════════════════════════════════════════════════════
def smape(actual, predicted) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)

def compute_metrics(actual: pd.Series, hybrid: pd.Series,
                    arima_pred: pd.Series, xgb_pred: pd.Series,
                    is_static: pd.Series) -> dict:
    """Hitung MAE/RMSE/sMAPE untuk hybrid, ARIMA-only, XGBoost-only."""
    idx = actual.index.intersection(hybrid.index)
    y   = actual.reindex(idx).values
    h   = hybrid.reindex(idx).values
    a   = arima_pred.reindex(idx).values
    x   = xgb_pred.reindex(idx).values

    mae_h  = float(np.mean(np.abs(y - h)))
    rmse_h = float(np.sqrt(np.mean((y - h)**2)))
    smap_h = smape(y, h)

    mae_a  = float(np.mean(np.abs(y - a)))
    mae_x  = float(np.mean(np.abs(y - x)))

    # Per kondisi
    st = is_static.reindex(idx).values.astype(bool)
    if st.sum() > 0:
        mae_st = float(np.mean(np.abs(y[st] - h[st])))
    else:
        mae_st = None
    if (~st).sum() > 0:
        mae_dy = float(np.mean(np.abs(y[~st] - h[~st])))
    else:
        mae_dy = None

    return {
        'MAE_hybrid'        : round(mae_h,  4),
        'RMSE_hybrid'       : round(rmse_h, 4),
        'sMAPE_hybrid'      : round(smap_h, 2),
        'MAE_ARIMA_only'    : round(mae_a,  4),
        'MAE_XGBoost_only'  : round(mae_x,  4),
        'MAE_static_seg'    : round(mae_st, 4) if mae_st else None,
        'MAE_dynamic_seg'   : round(mae_dy, 4) if mae_dy else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# 7. VISUALISASI
# ══════════════════════════════════════════════════════════════════════════
def draw_condition_bands(ax, is_static: pd.Series,
                         ymin: float, ymax: float):
    """Gambar background shading sesuai kondisi (statis=biru, dinamis=merah)."""
    condition = is_static.copy()
    changes   = condition[condition != condition.shift()].index.tolist()
    changes   = [condition.index[0]] + changes + [condition.index[-1]]

    for i in range(len(changes) - 1):
        t0 = changes[i]
        t1 = changes[i + 1]
        c  = condition[t0]
        color = COLOR_STATIC_BG if c else COLOR_DYNAMIC_BG
        ax.axvspan(t0, t1, alpha=0.25, color=color, linewidth=0, zorder=0)


def plot_switching_chart(df: pd.DataFrame,
                         hybrid_dict: dict,
                         arima_dict: dict,
                         xgb_dict: dict,
                         is_static: pd.Series,
                         all_metrics: dict):
    """
    Buat 3-panel chart switching:
      Baris 1 : SOC baterai
      Baris 2 : Daya motor total
      Baris 3 : Kecepatan linear
    """
    plt.rcParams.update({
        'font.family'     : 'serif',
        'font.serif'      : ['Times New Roman', 'DejaVu Serif'],
        'font.size'       : 10,
        'axes.titlesize'  : 11,
        'axes.labelsize'  : 10,
        'axes.linewidth'  : 1.2,
        'figure.dpi'      : 150,
        'savefig.dpi'     : 300,
        'axes.facecolor'  : 'white',
        'figure.facecolor': 'white',
    })

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.52)

    panel_info = [
        ('batt_soc_percent', 'State of Charge Baterai (SOC)',   '%'),
        ('joint_P_total',    'Daya Motor Total (P_total)',       'Watt'),
        ('odom_v_linear',    'Kecepatan Linear Robot',           'm/s'),
    ]

    time_fmt = mdates.DateFormatter('%H:%M:%S')

    for row, (field, title, unit) in enumerate(panel_info):
        ax = fig.add_subplot(gs[row])
        ax.set_facecolor('white')

        series  = df[field].dropna() if field in df.columns else pd.Series()
        hybrid  = hybrid_dict.get(field, pd.Series())
        arima_p = arima_dict.get(field, pd.Series())
        xgb_p   = xgb_dict.get(field, pd.Series())

        if series.empty:
            ax.text(0.5, 0.5, f'Data {field} tidak tersedia',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        ymin = series.min() - abs(series.min()) * 0.06
        ymax = series.max() + abs(series.max()) * 0.06

        # ── Background shading per kondisi ──
        is_st_aligned = is_static.reindex(series.index).fillna(True)
        draw_condition_bands(ax, is_st_aligned, ymin, ymax)

        idx    = series.index
        is_arr = is_st_aligned.values

        # ── Plot prediksi hybrid DULU (di belakang, zorder rendah) ──
        if not hybrid.empty:
            h_aligned = hybrid.reindex(series.index).ffill().bfill()
            ax.plot(h_aligned.index, h_aligned.values,
                    color=COLOR_HYBRID_PRED,
                    linewidth=LW_PRED,
                    linestyle='--',
                    dashes=LW_PRED_DASH,
                    alpha=0.85,
                    zorder=2)

        # ── Plot data aktual berwarna per segmen (di atas prediksi) ──
        # Setiap segmen +1 titik agar sambung ke segmen berikutnya
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and is_arr[j] == is_arr[i]:
                j += 1
            end_ext = min(j + 1, len(idx))
            seg_idx = idx[i:end_ext]
            seg_val = series.iloc[i:end_ext]
            color   = COLOR_STATIC_LINE if is_arr[i] else COLOR_DYNAMIC_LINE
            ax.plot(seg_idx, seg_val,
                    color=color,
                    linewidth=LW_DATA,
                    solid_capstyle='butt',
                    solid_joinstyle='round',
                    zorder=3)
            i = j

        # ── Garis vertikal tipis di setiap transisi ──
        prev = None
        for pos, (t, s) in enumerate(is_st_aligned.items()):
            if prev is not None and s != prev:
                ax.axvline(x=t, color='#888888',
                           linewidth=0.5, linestyle=':', alpha=0.35, zorder=1)
            prev = s

        # ── Metrik di sudut kanan atas ──
        m = all_metrics.get(field, {})
        if m:
            metric_txt = (f"MAE Hybrid = {m.get('MAE_hybrid','—'):.4f} {unit}\n"
                          f"sMAPE = {m.get('sMAPE_hybrid','—'):.2f}%")
            ax.text(0.99, 0.96, metric_txt,
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8.5, color='#222222',
                    bbox=dict(facecolor='white', alpha=0.85,
                              edgecolor='#AAAAAA', boxstyle='round,pad=0.3'),
                    zorder=5)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_xlim(series.index[0], series.index[-1])
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_formatter(time_fmt)
        ax.tick_params(colors='#333333')
        ax.grid(True, color='#CCCCCC', alpha=0.45, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#AAAAAA')

    # ── Legenda global ──
    legend_patches = [
        mpatches.Patch(color=COLOR_STATIC_LINE,  label='Data Aktual — Kondisi STATIS (ARIMA aktif)'),
        mpatches.Patch(color=COLOR_DYNAMIC_LINE, label='Data Aktual — Kondisi DINAMIS (XGBoost aktif)'),
        mpatches.Patch(color=COLOR_HYBRID_PRED,  label='Prediksi Hybrid — ARIMA/XGBoost bergantian'),
        mpatches.Patch(color=COLOR_STATIC_BG,    alpha=0.5, label='Zona STATIS (background)'),
        mpatches.Patch(color=COLOR_DYNAMIC_BG,   alpha=0.5, label='Zona DINAMIS (background)'),
    ]
    fig.legend(handles=legend_patches,
               loc='lower center',
               ncol=3,
               fontsize=9,
               facecolor='white',
               edgecolor='#AAAAAA',
               framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        'Condition-Based Temporal Switching — ARIMA ↔ XGBoost\n'
        'Polebot AMR · Skenario C7 (Mixed Switching Demo)\n'
        'Politeknik Manufaktur Bandung · 2025/2026',
        fontsize=12, fontweight='bold', y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'hybrid_switching_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  ✅ Chart tersimpan: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════
# 8. SIMPAN HASIL
# ══════════════════════════════════════════════════════════════════════════
def save_metrics(all_metrics: dict, n_transitions: int, df: pd.DataFrame,
                 is_static: pd.Series):
    """Simpan ringkasan hasil ke JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_static  = int(is_static.sum())
    n_dynamic = int((~is_static).sum())

    hasil = {
        'dijalankan'            : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'skenario'              : 'C7 — Mixed Switching Demo',
        'metode'                : 'Condition-Based Temporal Switching',
        'total_rekaman'         : len(df),
        'titik_statis'          : n_static,
        'titik_dinamis'         : n_dynamic,
        'persen_statis'         : round(n_static/len(df)*100, 1),
        'jumlah_transisi'       : n_transitions,
        'threshold_accel'       : ACCEL_THRESHOLD,
        'threshold_speed'       : SPEED_THRESHOLD,
        'metrik_per_variabel'   : all_metrics,
    }

    json_path = os.path.join(OUTPUT_DIR, 'hybrid_switching_metrics.json')
    with open(json_path, 'w') as f:
        def _convert(o):
            if isinstance(o, (int, float)): return o
            try: return int(o)
            except: return str(o)
        json.dump(hasil, f, indent=2, ensure_ascii=False, default=_convert)

    print(f"  ✅ Metrik tersimpan: {json_path}")
    return hasil


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*60)
    print("  HYBRID SWITCHING PREDICTOR — POLEBOT AMR")
    print("  Condition-Based Temporal Switching")
    print("  Skenario C7 — Mixed Switching Demo")
    print("═"*60)

    # 1. Ambil data
    df = fetch_all_data()

    # 2. Klasifikasi kondisi
    print("\n" + "─"*60)
    print("  KLASIFIKASI KONDISI OPERASIONAL")
    print("─"*60)
    is_static    = classify_conditions(df)
    n_trans      = count_transitions(is_static)

    # 3. Fit model untuk setiap variabel
    print("\n" + "─"*60)
    print("  FITTING MODEL")
    print("─"*60)
    arima_results = {}
    xgb_results   = {}

    for t in TARGETS:
        field  = t['field']
        label  = t['label']
        series = df[field].dropna()

        arima_results[field] = fit_arima_model(series, label)
        xgb_results[field]   = fit_xgboost_model(df, field, label)

    # 4. Stitch prediksi
    print("\n" + "─"*60)
    print("  CONDITION-BASED TEMPORAL SWITCHING")
    print("─"*60)
    hybrid_preds = {}
    for t in TARGETS:
        field = t['field']
        hybrid = stitch_predictions(
            arima_results[field],
            xgb_results[field],
            is_static,
            df[field]
        )
        hybrid_preds[field] = hybrid
        n_arima_used  = int(is_static.reindex(hybrid.index).sum())
        n_xgb_used    = int((~is_static.reindex(hybrid.index)).sum())
        print(f"  {t['label'][:35]:<35}"
              f"  ARIMA: {n_arima_used:5d} titik  |"
              f"  XGBoost: {n_xgb_used:5d} titik")

    # 5. Hitung metrik
    print("\n" + "─"*60)
    print("  METRIK EVALUASI")
    print("─"*60)
    all_metrics = {}
    for t in TARGETS:
        field  = t['field']
        unit   = t['unit']
        m = compute_metrics(
            actual   = df[field],
            hybrid   = hybrid_preds[field],
            arima_pred = arima_results[field]['fitted'],
            xgb_pred   = xgb_results[field]['predictions'],
            is_static  = is_static
        )
        all_metrics[field] = m
        print(f"\n  {t['label']}")
        print(f"    MAE  Hybrid   : {m['MAE_hybrid']:.4f} {unit}")
        print(f"    RMSE Hybrid   : {m['RMSE_hybrid']:.4f} {unit}")
        print(f"    sMAPE Hybrid  : {m['sMAPE_hybrid']:.2f}%")
        print(f"    MAE ARIMA-only: {m['MAE_ARIMA_only']:.4f} {unit}")
        print(f"    MAE XGB-only  : {m['MAE_XGBoost_only']:.4f} {unit}")

    # 6. Visualisasi
    print("\n" + "─"*60)
    print("  MEMBUAT CHART SWITCHING")
    print("─"*60)

    arima_fitted = {f: arima_results[f]['fitted'] for f in arima_results}
    xgb_preds    = {f: xgb_results[f]['predictions'] for f in xgb_results}

    plot_switching_chart(
        df            = df,
        hybrid_dict   = hybrid_preds,
        arima_dict    = arima_fitted,
        xgb_dict      = xgb_preds,
        is_static     = is_static,
        all_metrics   = all_metrics,
    )

    # 7. Simpan metrik
    hasil = save_metrics(all_metrics, n_trans, df, is_static)

    # 8. Ringkasan terminal
    print("\n" + "═"*60)
    print("  SELESAI — RINGKASAN")
    print("═"*60)
    print(f"  Total rekaman C7  : {len(df)}")
    print(f"  Titik STATIS      : {hasil['titik_statis']} "
          f"({hasil['persen_statis']:.1f}%)")
    print(f"  Titik DINAMIS     : {hasil['titik_dinamis']} "
          f"({100-hasil['persen_statis']:.1f}%)")
    print(f"  Transisi S↔D      : {n_trans} kali")
    print(f"\n  File output:")
    print(f"    {OUTPUT_DIR}/hybrid_switching_chart.png")
    print(f"    {OUTPUT_DIR}/hybrid_switching_metrics.json")
    print("═"*60)


if __name__ == '__main__':
    main()