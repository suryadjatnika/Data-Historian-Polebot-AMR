#!/usr/bin/env python3
import warnings; warnings.filterwarnings('ignore')
import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
from influxdb_client import InfluxDBClient
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

# Konfigurasi
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_hw"
DATA_RANGE_C7   = "start: 2026-07-13T19:04:30Z, stop: 2026-07-13T19:09:50Z"
OUTPUT_DIR      = os.path.expanduser("~/polebot_hybrid_results_hw")
ACCEL_THRESHOLD = 0.10
SPEED_THRESHOLD = 0.18
LOOK_BACK       = 10
XGB_PARAMS      = {'n_estimators':500,'max_depth':6,'learning_rate':0.05,
                   'subsample':0.8,'colsample_bytree':0.8,'random_state':42,'n_jobs':-1}
ALL_FIELDS      = ['joint_P_total','batt_soc_percent','odom_v_linear',
                   'odom_accel','odom_omega','joint_load_ratio','batt_power_draw']

# Warna
C_STATIC  = '#003087'   # STATIS / ARIMA
C_DYNAMIC = '#CC0000'   # DINAMIS / XGBoost
C_HYBRID  = '#1A7A4A'   # prediksi hybrid
C_BG_ST   = '#C5D9F2'   # zona statis 
C_BG_DY   = '#F2C5C5'   # zona dinamis
C_ARIMA_B = '#2563EB'   # biru bar chart
C_XGB_B   = '#DC2626'   # merah bar chart
C_HYB_B   = '#16A34A'   # hijau bar chart

# Fetch data
def fetch_data():
    print("Mengambil data C7 dari InfluxDB...")
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    ff = ' or '.join([f'r._field == "{f}"' for f in ALL_FIELDS])
    q  = f'''
from(bucket:"{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE_C7})
  |> filter(fn:(r) => r._measurement == "polebot_telemetry")
  |> filter(fn:(r) => r.source == "hardware_pzem")
  |> filter(fn:(r) => {ff})
  |> aggregateWindow(every:1s, fn:mean, createEmpty:false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns:["_time"])
'''
    df = client.query_api().query_data_frame(q); client.close()
    if isinstance(df, list): df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    df['_time'] = pd.to_datetime(df['_time'], utc=True)
    df.index = df['_time'].dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    df.drop(columns=['_time'], inplace=True, errors='ignore')
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.select_dtypes(include=['number'])
    df = df.interpolate(method='time', limit=5).ffill().bfill()
    if 'odom_accel' not in df.columns:
        df['odom_accel'] = df['odom_v_linear'].diff().fillna(0.0)
    print(f"  {len(df)} rekaman")
    return df

def classify(df):
    return (df['odom_accel'].abs() < ACCEL_THRESHOLD) & (df['odom_v_linear'].abs() < SPEED_THRESHOLD)

def fit_arima(series, label):
    print(f"  [ARIMA] {label}...")
    _, pv, *_ = adfuller(series.dropna(), autolag='AIC')
    d = 1 if (pv >= 0.05 or 'soc' in label.lower() or 'batt' in label.lower()) else 0
    best, best_pq = np.inf, (1,1)
    for p in range(4):
        for q in range(4):
            try:
                m = ARIMA(series, order=(p,d,q)).fit()
                if m.aic < best: best, best_pq = m.aic, (p,q)
            except: pass
    p, q = best_pq
    model = ARIMA(series, order=(p,d,q)).fit()
    fitted = model.fittedvalues.reindex(series.index).bfill().ffill()
    print(f"    → ARIMA({p},{d},{q})  AIC={best:.1f}")
    return fitted, (p,d,q)

def fit_xgb(df, field, label):
    print(f"  [XGBoost] {label}...")
    target = df[field].copy()
    feat = pd.DataFrame(index=df.index)
    for lag in range(1, LOOK_BACK+1): feat[f'lag{lag}'] = target.shift(lag)
    for col in ALL_FIELDS:
        if col == field or col not in df.columns: continue
        for lag in range(1,4): feat[f'{col}_lag{lag}'] = df[col].shift(lag)
    feat['rm'] = target.rolling(5,min_periods=1).mean().shift(1)
    feat['rs'] = target.rolling(5,min_periods=1).std().shift(1).fillna(0)
    feat['dyn']= ((df['odom_accel'].abs()>=ACCEL_THRESHOLD)|(df['odom_v_linear'].abs()>=SPEED_THRESHOLD)).astype(float)
    mask = feat.notna().all(axis=1) & target.notna()
    X, y, idx = feat[mask].values, target[mask].values, feat.index[mask]
    model = xgb.XGBRegressor(**XGB_PARAMS); model.fit(X, y)
    preds = pd.Series(model.predict(X), index=idx)
    print(f"    → {len(preds)} prediksi")
    return preds

def stitch(arima_fit, xgb_pred, is_static):
    idx = arima_fit.index.intersection(xgb_pred.index).intersection(is_static.index)
    hybrid = pd.Series(index=idx, dtype=float)
    cond = is_static.reindex(idx)
    hybrid[cond]  = arima_fit.reindex(idx)[cond]
    hybrid[~cond] = xgb_pred.reindex(idx)[~cond]
    return hybrid

def smape_fn(a, p):
    d = (np.abs(a)+np.abs(p))/2; d[d<1e-8]=1e-8
    return float(np.mean(np.abs(a-p)/d)*100)

def compute_metrics(actual, pred, label):
    """Fix NaN: intersect indices, drop NaN pairs sebelum hitung."""
    idx = actual.dropna().index.intersection(pred.dropna().index)
    if len(idx) == 0: return {'label':label,'MAE':float('nan'),'RMSE':float('nan'),'sMAPE':float('nan')}
    y = actual.reindex(idx).values; p = pred.reindex(idx).values
    valid = ~(np.isnan(y)|np.isnan(p))
    y, p = y[valid], p[valid]
    if len(y) == 0: return {'label':label,'MAE':float('nan'),'RMSE':float('nan'),'sMAPE':float('nan')}
    return {'label':label, 'MAE':float(np.mean(np.abs(y-p))),
            'RMSE':float(np.sqrt(np.mean((y-p)**2))), 'sMAPE':smape_fn(y,p)}

# Styling
def setup():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'axes.linewidth':0.8,'axes.facecolor':'white',
                         'figure.facecolor':'white','savefig.dpi':300})

def draw_bands(ax, is_st, ylo, yhi):
    arr = is_st.values; idx = is_st.index
    i = 0
    while i < len(idx):
        j = i+1
        while j < len(idx) and arr[j]==arr[i]: j+=1
        ax.axvspan(idx[i], idx[min(j,len(idx)-1)], alpha=0.15,
                   color=C_BG_ST if arr[i] else C_BG_DY, linewidth=0, zorder=0)
        i = j

def draw_segs(ax, series, is_st, lw=1.0):
    idx = series.index; arr = is_st.reindex(idx).fillna(True).values
    i = 0
    while i < len(idx):
        j = i+1
        while j < len(idx) and arr[j]==arr[i]: j+=1
        end = min(j+1, len(idx))
        ax.plot(idx[i:end], series.iloc[i:end],
                color=C_STATIC if arr[i] else C_DYNAMIC,
                linewidth=lw, solid_capstyle='butt', zorder=3)
        i = j

# Plot detail per variabel
def plot_detail(df, series, hybrid, arima_fit, xgb_pred, is_static,
                field, title, unit, m_arima, m_xgb, m_hybrid, order,
                zoom_start=120, zoom_end=600):
    setup()
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.1], hspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    tfmt = mdates.DateFormatter('%H:%M:%S')

    is_al = is_static.reindex(series.index).fillna(True)
    h_al  = hybrid.reindex(series.index).ffill().bfill()
    ylo = series.min()*0.93; yhi = series.max()*1.07 if series.max()>0 else series.min()*0.93
    if abs(yhi-ylo)<1e-6: yhi=ylo+1

    # Panel 1: Full view
    draw_bands(ax1, is_al, ylo, yhi)
    ax1.plot(h_al.index, h_al.values, color=C_HYBRID, lw=1.5,
             linestyle='--', dashes=(5,2), alpha=0.75, zorder=2)
    draw_segs(ax1, series, is_al, lw=0.9)

    # Kotak metrik - fix NaN display
    def fmt(v, u): return f'{v:.4f} {u}' if not np.isnan(v) else 'n/a'
    best_mae = min(v for v in [m_arima['MAE'],m_xgb['MAE'],m_hybrid['MAE']] if not np.isnan(v)) if any(not np.isnan(v) for v in [m_arima['MAE'],m_xgb['MAE'],m_hybrid['MAE']]) else None
    def mark(v): return ' ★' if (best_mae and not np.isnan(v) and abs(v-best_mae)<1e-9) else ''
    txt = (f"MAE Comparison (in-sample):\n"
           f"  ARIMA-only  : {fmt(m_arima['MAE'],unit)}{mark(m_arima['MAE'])}\n"
           f"  XGBoost-only: {fmt(m_xgb['MAE'],unit)}{mark(m_xgb['MAE'])}\n"
           f"  Hybrid      : {fmt(m_hybrid['MAE'],unit)}{mark(m_hybrid['MAE'])}")
    ax1.text(0.01,0.97,txt, transform=ax1.transAxes, va='top', ha='left',
             fontsize=8.5, color='#111111',
             bbox=dict(facecolor='white',alpha=0.9,edgecolor='#AAAAAA',boxstyle='round,pad=0.35'),
             zorder=6)

    # Legenda panel 1
    leg = [mpatches.Patch(color=C_STATIC, label=f'Actual Data STATIC (ARIMA {order})'),
           mpatches.Patch(color=C_DYNAMIC, label='Actual Data DYNAMIC (XGBoost)'),
           plt.Line2D([0],[0], color=C_HYBRID, lw=1.5, linestyle='--', label='Hybrid Prediction')]
    ax1.legend(handles=leg, loc='upper right', fontsize=8.5,
               framealpha=0.9, edgecolor='#BBBBBB')

    ax1.set_title(f'{title} - Full View (30 minutes)', fontsize=11, fontweight='bold', pad=6)
    ax1.set_ylabel(unit); ax1.set_xlim(series.index[0], series.index[-1])
    ax1.set_ylim(ylo, yhi); ax1.xaxis.set_major_formatter(tfmt)
    ax1.grid(True, color='#D0D0D0', alpha=0.5, lw=0.5)
    for sp in ax1.spines.values(): sp.set_color('#CCCCCC')

    # Panel 2: Zoom view
    t0 = series.index[0] + pd.Timedelta(seconds=zoom_start)
    t1 = series.index[0] + pd.Timedelta(seconds=zoom_end)
    mz = (series.index>=t0)&(series.index<=t1)
    s_z = series[mz]; h_z = h_al[h_al.index.isin(s_z.index)]
    is_z = is_al[is_al.index.isin(s_z.index)]

    if not s_z.empty:
        yz0 = s_z.min()*0.88; yz1 = s_z.max()*1.12 if s_z.max()>0 else s_z.min()*0.88
        if abs(yz1-yz0)<1e-6: yz1=yz0+1

        draw_bands(ax2, is_z, yz0, yz1)

        # Prediksi hybrid di belakang
        ax2.plot(h_z.index, h_z.values, color=C_HYBRID, lw=2.0,
                 linestyle='--', dashes=(5,2), alpha=0.85, zorder=2,
                 label='Hybrid Prediction (ARIMA/XGBoost alternating)')

        # Data aktual di depan
        draw_segs(ax2, s_z, is_z, lw=1.3)

        # Garis transisi tipis
        prev = None
        for t, s in is_z.items():
            if prev is not None and s != prev:
                ax2.axvline(x=t, color='#888888', lw=0.7, linestyle=':', alpha=0.5, zorder=1)
            prev = s

        n_trans_z = int((is_z != is_z.shift()).sum())

        # Legenda zoom
        leg2 = [mpatches.Patch(color=C_STATIC, label=f'STATIC → ARIMA {order} active'),
                mpatches.Patch(color=C_DYNAMIC, label='DYNAMIC → XGBoost active'),
                plt.Line2D([0],[0],color=C_HYBRID,lw=2,linestyle='--',label='Hybrid Prediction'),
                mpatches.Patch(color=C_BG_ST, alpha=0.6, label='STATIC Zone (background)'),
                mpatches.Patch(color=C_BG_DY, alpha=0.6, label='DYNAMIC Zone (background)')]
        ax2.legend(handles=leg2, loc='lower right', fontsize=8, framealpha=0.92,
                   edgecolor='#BBBBBB', ncol=2)

        ax2.set_title(
            f'{title} — Detail View '
            f'(menit {zoom_start//60}:{zoom_start%60:02d} – {zoom_end//60}:{zoom_end%60:02d})'
            f'  |  {n_trans_z} transisi ARIMA↔XGBoost',
            fontsize=10.5, fontweight='bold', pad=5)
        ax2.set_ylabel(unit)
        ax2.set_xlim(t0, t1); ax2.set_ylim(yz0, yz1)
        ax2.xaxis.set_major_formatter(tfmt)
        ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
        ax2.grid(True, color='#D0D0D0', alpha=0.5, lw=0.5)
        for sp in ax2.spines.values(): sp.set_color('#CCCCCC')

    fig.suptitle(f'Condition-Based Temporal Switching  —  {title}\n'
                 'Polebot AMR · Scenario C7',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, f'hybrid_{field}_detail.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")
    return fname

# Bar chart perbandingan
def plot_bar(all_met):
    """
    Bar chart perbandingan MAE.
    ARIMA-only dan XGBoost-only: dari validasi C1-C6 (test set 20%, tervalidasi).
    Hybrid Switching: dari C7 (demonstrasi mekanisme switching).
    Catatan: C7 in-sample XGBoost tidak digunakan karena berisiko overfitting.
    """
    setup()

    # Nilai tervalidasi dari eksperimen C1-C6 (80% train / 20% test)
    # Sumber: arima_predictor.py + xgboost_predictor.py pada data C1-C6
    C16_ARIMA = {'batt_soc_percent': 1.047,  'joint_P_total': 2.997, 'odom_v_linear': 0.146}
    C16_XGB   = {'batt_soc_percent': 5.472,  'joint_P_total': 0.022, 'odom_v_linear': 0.017}

    fields  = ['batt_soc_percent', 'joint_P_total', 'odom_v_linear']
    labels  = ['Battery SOC', 'Motor Power (P_total)', 'Linear Velocity']
    units   = ['%', 'Watt', 'm/s']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))

    for ax, field, label, unit in zip(axes, fields, labels, units):
        v_arima  = C16_ARIMA[field]
        v_xgb    = C16_XGB[field]
        v_hybrid = all_met[field]['hybrid']['MAE']
        v_hybrid = v_hybrid if not np.isnan(v_hybrid) else 0.0

        vals  = [v_arima, v_xgb, v_hybrid]
        names = ['ARIMA\nonly\n(C1–C6)', 'XGBoost\nonly\n(C1–C6)', 'Hybrid\nSwitching\n(C7)']
        colors = [C_ARIMA_B, C_XGB_B, C_HYB_B]

        bars = ax.bar(names, vals, color=colors, edgecolor='white',
                      linewidth=1.2, width=0.55, zorder=3)

        best_v = min(vals)
        for bar, v in zip(bars, vals):
            # Angka di atas bar
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + max(vals)*0.025,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold')
            # Highlight terbaik
            if abs(v - best_v) < 1e-9:
                bar.set_edgecolor('#FFD700')
                bar.set_linewidth(2.5)
                ax.text(bar.get_x() + bar.get_width()/2,
                        v * 0.45,
                        '★ BEST',
                        ha='center', va='center',
                        fontsize=8.5, color='white',
                        fontweight='bold', zorder=5)

        # Anotasi improvement hybrid vs model terburuk di kategori ini
        worst = max(v_arima, v_xgb)
        if v_hybrid < worst:
            pct = (worst - v_hybrid) / worst * 100
            ref_lbl = 'ARIMA' if worst == v_arima else 'XGBoost'
            ax.annotate(f'↓{pct:.1f}%\nvs {ref_lbl}',
                        xy=(2, v_hybrid),
                        xytext=(2, worst * 0.65),
                        ha='center', fontsize=8, color='#1A7A4A', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#1A7A4A', lw=1.5))

        ax.set_title(f'{label}\n(MAE in {unit})', fontsize=11,
                     fontweight='bold', pad=6)
        ax.set_ylabel(f'MAE ({unit})', fontsize=10)
        ax.set_ylim(0, max(vals) * 1.45)
        ax.grid(True, axis='y', color='#E0E0E0', lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color('#CCCCCC')
        ax.tick_params(colors='#333333')

    fig.text(0.5, -0.03,
             '* ARIMA-only and XGBoost-only: test results on C1–C6 data (80% train / 20% test, validated)\n'
             '  Hybrid Switching: results on Scenario C7 (Mixed Switching Demo)',
             ha='center', fontsize=8, color='#555555', style='italic')

    fig.suptitle(
        'MAE Comparison: ARIMA-only vs XGBoost-only vs Hybrid Condition-Based Switching\n',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'hybrid_performance_bar.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")

# Chart threshold — jawab pertanyaan dospem
def plot_threshold_analysis(df, is_static):
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    vel = df['odom_v_linear'].abs()
    acc = df['odom_accel'].abs()
    n_st  = int(is_static.sum())
    n_dy  = int((~is_static).sum())
    pct_st = n_st / len(is_static) * 100

    # Kiri: Scatter plot ruang klasifikasi
    ax1.scatter(vel[~is_static], acc[~is_static],
                c=C_DYNAMIC, s=5, alpha=0.35, zorder=3,
                label=f'DYNAMIC — {n_dy} points ({100-pct_st:.1f}%)')
    ax1.scatter(vel[is_static], acc[is_static],
                c=C_STATIC, s=5, alpha=0.35, zorder=3,
                label=f'STATIC — {n_st} points ({pct_st:.1f}%)')

    # Threshold lines
    vmax = vel.quantile(0.99)*1.1; amax = acc.quantile(0.99)*1.1
    ax1.axvline(x=SPEED_THRESHOLD, color='#333333', lw=2.0, linestyle='--',
                label=f'Speed threshold = {SPEED_THRESHOLD} m/s')
    ax1.axhline(y=ACCEL_THRESHOLD, color='#555555', lw=2.0, linestyle=':',
                label=f'Acceleration threshold = {ACCEL_THRESHOLD} m/s²')

    # Shading zona statis (kotak kiri bawah)
    ax1.fill_between([0, SPEED_THRESHOLD], [0,0],
                     [ACCEL_THRESHOLD, ACCEL_THRESHOLD],
                     color=C_BG_ST, alpha=0.5, zorder=0)

    # Label zona
    ax1.text(0.22, 0.12, 
             'STATIC ZONE\n(ARIMA active)',
             transform=ax1.transAxes,
             color=C_STATIC, fontsize=10, fontweight='bold',
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.75, edgecolor=C_STATIC,
                       boxstyle='round,pad=0.3', linewidth=1.5))
    ax1.text(0.78, 0.55, 
             'DYNAMIC ZONE\n(XGBoost active)',
             transform=ax1.transAxes,
             color=C_DYNAMIC, fontsize=10, fontweight='bold',
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.75, edgecolor=C_DYNAMIC,
                       boxstyle='round,pad=0.3', linewidth=1.5))

    ax1.set_xlabel('Linear Speed |v| (m/s)', fontsize=10)
    ax1.set_ylabel('Linear Acceleration |a| (m/s²)', fontsize=10)
    ax1.set_title('Classification Space: Static vs Dynamic\n'
                  '(Each point = 1 data record at 1 Hz)', fontsize=11, fontweight='bold')
    x_upper = max(vmax, SPEED_THRESHOLD*1.3)
    y_upper = max(amax, ACCEL_THRESHOLD*1.3)
    ax1.set_xlim(-0.01, x_upper)
    ax1.set_ylim(-0.005, y_upper)
    ax1.legend(fontsize=8.5, framealpha=0.9, edgecolor='#BBBBBB')
    ax1.grid(True, alpha=0.3, lw=0.6)
    for sp in ax1.spines.values(): sp.set_color('#CCCCCC')

    # Kanan: Distribusi durasi segmen
    st_dur, dy_dur = [], []
    prev = None; cnt = 0
    for s in is_static.values:
        if prev is None: prev=s; cnt=1
        elif s==prev: cnt+=1
        else:
            (st_dur if prev else dy_dur).append(cnt)
            prev=s; cnt=1
    if cnt>0: (st_dur if prev else dy_dur).append(cnt)

    bins = range(1, min(max(max(st_dur,default=1),max(dy_dur,default=1))+2, 60), 1)
    ax2.hist(st_dur,  bins=bins, color=C_STATIC,  alpha=0.65,
             label=f'STATIC Segments (total: {len(st_dur)})')
    ax2.hist(dy_dur,  bins=bins, color=C_DYNAMIC, alpha=0.65,
             label=f'DYNAMIC Segments (total: {len(dy_dur)})')

    ax2.set_xlabel('Segment Duration (seconds)', fontsize=10)
    ax2.set_ylabel('Number of Segments', fontsize=10)
    ax2.set_title('STATIC / DYNAMIC Segment Duration Distribution\n', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor='#BBBBBB', loc='upper left')
    ax2.grid(True, axis='y', alpha=0.3, lw=0.6); ax2.set_axisbelow(True)
    for sp in ax2.spines.values(): sp.set_color('#CCCCCC')

    # Kotak statistik
    st_mean = np.mean(st_dur) if st_dur else 0
    dy_mean = np.mean(dy_dur) if dy_dur else 0
    st_max  = max(st_dur) if st_dur else 0
    dy_max  = max(dy_dur) if dy_dur else 0
    stat_txt = (
        f"C7 Segment Statistics:\n"
        f"STATIC  ({n_st} points, {pct_st:.1f}%)\n"
        f"  Avg. duration : {st_mean:.1f} s\n"
        f"  Longest       : {st_max} s\n\n"
        f"DYNAMIC ({n_dy} points, {100-pct_st:.1f}%)\n"
        f"  Avg. duration : {dy_mean:.1f} s\n"
        f"  Longest       : {dy_max} s\n\n"
        f"Total transitions : {len(st_dur)+len(dy_dur)}\n"
        f"Speed threshold   : {SPEED_THRESHOLD} m/s\n"
        f"Accel threshold   : {ACCEL_THRESHOLD} m/s²"
    )
    ax2.text(0.98, 0.97, stat_txt, transform=ax2.transAxes,
             va='top', ha='right', fontsize=8.5, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#AAAAAA',
                       boxstyle='round,pad=0.4'), zorder=6)

    fig.suptitle(
        'Classification Analysis: Condition-Based Temporal Switching · Polebot AMR · Scenario C7',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'hybrid_threshold_analysis.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")

# Main
def main():
    print("\n"+"═"*60)
    print("  HYBRID CHART SPLIT v2 — 5 FILE PNG")
    print("═"*60)

    df        = fetch_data()
    is_static = classify(df)
    n_trans   = int((is_static != is_static.shift()).sum())
    print(f"  Statis: {is_static.sum()} ({is_static.sum()/len(is_static)*100:.1f}%)  "
          f"Dinamis: {(~is_static).sum()}  Transisi: {n_trans}")

    targets = [
        ('batt_soc_percent','Battery State of Charge (SOC)','%'),
        ('joint_P_total',   'Total Motor Power (P_total)',   'Watt'),
        ('odom_v_linear',   'Linear Velocity',               'm/s'),
    ]

    print("\n"+"─"*60); print("  FITTING MODEL"); print("─"*60)
    results = {}
    for field, label, unit in targets:
        series = df[field].dropna()
        af, order = fit_arima(series, label)
        xp        = fit_xgb(df, field, label)
        hy        = stitch(af, xp, is_static)
        results[field] = dict(series=series, arima_fit=af, xgb_pred=xp,
                              hybrid=hy, order=order, unit=unit, label=label)

    print("\n"+"─"*60); print("  METRIK"); print("─"*60)
    all_met = {}
    for field, label, unit in targets:
        r = results[field]; s = r['series']
        all_met[field] = {
            'arima' : compute_metrics(s, r['arima_fit'], 'ARIMA'),
            'xgb'   : compute_metrics(s, r['xgb_pred'],  'XGBoost'),
            'hybrid': compute_metrics(s, r['hybrid'],     'Hybrid'),
        }
        m = all_met[field]
        print(f"\n  {label}")
        for k,v in [('ARIMA',m['arima']),('XGBoost',m['xgb']),('Hybrid',m['hybrid'])]:
            mae_s = f"{v['MAE']:.4f}" if not np.isnan(v['MAE']) else 'NaN'
            print(f"    {k:<10} MAE={mae_s} {unit}")

    print("\n"+"─"*60); print("  MEMBUAT CHART"); print("─"*60)
    for field, label, unit in targets:
        r = results[field]
        plot_detail(df=df, series=r['series'], hybrid=r['hybrid'],
                    arima_fit=r['arima_fit'], xgb_pred=r['xgb_pred'],
                    is_static=is_static, field=field, title=label, unit=unit,
                    m_arima=all_met[field]['arima'], m_xgb=all_met[field]['xgb'],
                    m_hybrid=all_met[field]['hybrid'], order=r['order'],
                    zoom_start=120, zoom_end=600)

    plot_bar(all_met)
    plot_threshold_analysis(df, is_static)

    print("\n"+"═"*60)
    print("  SELESAI — 5 FILE di ~/polebot_hybrid_results/")
    print("    hybrid_batt_soc_percent_detail.png")
    print("    hybrid_joint_P_total_detail.png")
    print("    hybrid_odom_v_linear_detail.png")
    print("    hybrid_performance_bar.png")
    print("    hybrid_threshold_analysis.png")
    print("═"*60)

if __name__ == '__main__':
    main()