#!/usr/bin/env python3
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from influxdb_client import InfluxDBClient

# Konfigurasi
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_hw"
DATA_RANGE      = "start: 2026-07-13T19:04:30Z, stop: 2026-07-13T19:09:50Z"
OUTPUT_DIR      = os.path.expanduser("~/polebot_hybrid_results_hw")

ACCEL_THR = 0.10
SPEED_THR = 0.18

# Zoom: potongan waktu (detik relatif) untuk gambar detail
ZOOM_START = 60
ZOOM_END   = 160

# Warna
C_STATIC  = '#003087'
C_DYNAMIC = '#CC0000'
C_BG_ST   = '#C5D9F2'
C_BG_DY   = '#F2C5C5'
C_HYBRID  = '#1A7A4A'


def setup():
    plt.rcParams.update({
        'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'axes.linewidth':0.8,'figure.facecolor':'white','savefig.dpi':300})


def fetch_data():
    print("Mengambil data Skenario 7 dari InfluxDB...")
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    fields = ['odom_v_linear','odom_accel','joint_P_total','batt_soc_percent']
    ff = ' or '.join([f'r._field == "{f}"' for f in fields])
    q = f'''
from(bucket:"{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn:(r) => r._measurement == "polebot_telemetry")
  |> filter(fn:(r) => r.source == "hardware_pzem")
  |> filter(fn:(r) => {ff})
  |> aggregateWindow(every:1s, fn:mean, createEmpty:false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns:["_time"])
'''
    df = client.query_api().query_data_frame(q); client.close()
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    df['_time'] = pd.to_datetime(df['_time'], utc=True)
    df.index = df['_time'].dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    for c in ['odom_v_linear','odom_accel','joint_P_total','batt_soc_percent']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'odom_accel' not in df.columns or df['odom_accel'].isna().all():
        df['odom_accel'] = df['odom_v_linear'].diff().fillna(0.0)
    df = df.dropna(subset=['odom_v_linear','joint_P_total'])
    df['t_rel'] = (df.index - df.index[0]).total_seconds()
    print(f"  {len(df)} rekaman, durasi {df['t_rel'].max():.0f} detik")
    return df


def classify(df):
    return (df['odom_accel'].abs() < ACCEL_THR) & (df['odom_v_linear'].abs() < SPEED_THR)


def draw_bands(ax, t, is_st):
    """Gambar pita latar statis (biru) / dinamis (merah)."""
    arr = np.asarray(is_st); i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and arr[j] == arr[i]: j += 1
        ax.axvspan(t[i], t[min(j, len(t)-1)], alpha=1.0,
                   color=C_BG_ST if arr[i] else C_BG_DY, linewidth=0, zorder=0)
        i = j


def count_transitions(is_st):
    s = pd.Series(is_st)
    return int((s != s.shift()).sum())

# GAMBAR 1 TIMELINE PENUH
def plot_timeline_full(df, is_st):
    setup()
    t = df['t_rel'].values
    v = df['odom_v_linear'].abs().values
    p = df['joint_P_total'].values
    n_st = int(is_st.sum()); n_dy = int((~is_st).sum())
    pct_st = 100*n_st/len(is_st); n_trans = count_transitions(is_st.values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Panel kecepatan
    draw_bands(ax1, t, is_st.values)
    ax1.plot(t, v, color='#333333', lw=1.1, zorder=3)
    ax1.axhline(SPEED_THR, color='#000000', ls='--', lw=1, alpha=0.6,
                label=f'Ambang kecepatan = {SPEED_THR} m/s')
    ax1.set_ylabel('Kecepatan |v| (m/s)')
    ax1.set_title('Timeline Penuh Skenario 7 (Mixed Switching) Condition-Based Temporal Switching',
                  fontweight='bold', pad=8)
    ax1.legend(loc='upper right', fontsize=8.5)
    ax1.grid(alpha=0.3, lw=0.5)

    # Panel daya
    draw_bands(ax2, t, is_st.values)
    ax2.plot(t, p, color='#333333', lw=1.1, zorder=3)
    ax2.set_ylabel('Daya Motor (W)')
    ax2.set_xlabel('Waktu (detik)')
    ax2.grid(alpha=0.3, lw=0.5)

    leg = [mpatches.Patch(color=C_BG_ST, alpha=0.6, label='Zona STATIS → ARIMA'),
           mpatches.Patch(color=C_BG_DY, alpha=0.6, label='Zona DINAMIS → XGBoost')]
    ax2.legend(handles=leg, loc='upper right', fontsize=8.5)

    # Kotak ringkasan
    txt = (f"Statistik Skenario 7:\n"
           f"  STATIS  : {n_st} titik ({pct_st:.1f}%)\n"
           f"  DINAMIS : {n_dy} titik ({100-pct_st:.1f}%)\n"
           f"  Transisi: {n_trans} kali")
    ax1.text(0.01, 0.97, txt, transform=ax1.transAxes, va='top', ha='left',
             fontsize=8.5, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#AAAAAA',
                       boxstyle='round,pad=0.4'), zorder=6)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, 'hybrid_timeline_full.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")


# GAMBAR 2 ZOOM SATU SIKLUS
def plot_cycle_zoom(df, is_st):
    setup()
    t = df['t_rel'].values
    mz = (t >= ZOOM_START) & (t <= ZOOM_END)
    tz = t[mz]
    vz = df['odom_v_linear'].abs().values[mz]
    pz = df['joint_P_total'].values[mz]
    isz = is_st.values[mz]

    if len(tz) == 0:
        print("  ⚠️  Rentang zoom kosong, lewati gambar zoom.")
        return

    n_trans_z = count_transitions(isz)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Panel kecepatan
    draw_bands(ax1, tz, isz)
    ax1.plot(tz, vz, color='#333333', lw=1.4, marker='o', markersize=2.5, zorder=3)
    ax1.axhline(SPEED_THR, color='#000000', ls='--', lw=1, alpha=0.6,
                label=f'Ambang kecepatan = {SPEED_THR} m/s')
    ax1.set_ylabel('Kecepatan |v| (m/s)')
    ax1.set_title(f'Detail Transisi Skenario 7 (detik {ZOOM_START}–{ZOOM_END}) '
                  f'{n_trans_z} transisi ARIMA↔XGBoost',
                  fontweight='bold', pad=8)
    ax1.legend(loc='upper right', fontsize=8.5)
    ax1.grid(alpha=0.3, lw=0.5)

    # Garis transisi vertikal
    prev = None
    for tt, s in zip(tz, isz):
        if prev is not None and s != prev:
            ax1.axvline(tt, color='#888888', ls=':', lw=1, alpha=0.7, zorder=2)
            ax2.axvline(tt, color='#888888', ls=':', lw=1, alpha=0.7, zorder=2)
        prev = s

    # Panel daya
    draw_bands(ax2, tz, isz)
    ax2.plot(tz, pz, color='#333333', lw=1.4, marker='o', markersize=2.5, zorder=3)
    ax2.set_ylabel('Daya Motor (W)')
    ax2.set_xlabel('Waktu (detik)')
    ax2.grid(alpha=0.3, lw=0.5)

    leg = [mpatches.Patch(color=C_BG_ST, alpha=0.6, label='STATIS = ARIMA'),
           mpatches.Patch(color=C_BG_DY, alpha=0.6, label='DINAMIS = XGBoost'),
           plt.Line2D([0],[0], color='#888888', ls=':', lw=1, label='Titik transisi')]
    ax2.legend(handles=leg, loc='upper right', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, 'hybrid_cycle_zoom.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")


# GAMBAR 3 SOC
def plot_soc_separate(df, is_st):
    setup()
    t = df['t_rel'].values
    soc = df['batt_soc_percent'].values
    n_st = int(is_st.sum()); n_dy = int((~is_st).sum())
    pct_st = 100*n_st/len(is_st)

    fig, ax = plt.subplots(figsize=(13, 5))
    draw_bands(ax, t, is_st.values)
    ax.plot(t, soc, color=C_STATIC, lw=1.6, zorder=3)

    ax.set_ylabel('State of Charge (%)')
    ax.set_xlabel('Waktu (detik)')
    ax.set_title('Prediksi State of Charge (SOC) pada Skenario 7',
                 fontweight='bold', pad=8)
    ax.grid(alpha=0.3, lw=0.5)

    # Beri sedikit ruang vertikal agar variasi kecil SOC terlihat
    soc_min, soc_max = soc.min(), soc.max()
    span = max(soc_max - soc_min, 0.5)
    ax.set_ylim(soc_min - span*0.3, soc_max + span*0.3)

    # Kotak keterangan
    delta = soc[0] - soc[-1]
    txt = (f"SOC Skenario 7:\n"
           f"  Awal   : {soc[0]:.2f}%\n"
           f"  Akhir  : {soc[-1]:.2f}%\n"
           f"  Turun  : {delta:.2f}%")
    ax.text(0.01, 0.05, txt, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=8.5, family='monospace',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#AAAAAA',
                      boxstyle='round,pad=0.4'), zorder=6)

    leg = [mpatches.Patch(color=C_BG_ST, alpha=0.6, label='Zona STATIS'),
           mpatches.Patch(color=C_BG_DY, alpha=0.6, label='Zona DINAMIS')]
    ax.legend(handles=leg, loc='upper right', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, 'hybrid_soc_separate.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")


def main():
    print("\n" + "═"*60)
    print("  HYBRID CYCLE ANNOTATION (HARDWARE)")
    print("═"*60)
    df = fetch_data()
    is_st = classify(df)
    n_trans = count_transitions(is_st.values)
    print(f"  Statis: {is_st.sum()} ({100*is_st.mean():.1f}%)  "
          f"Dinamis: {(~is_st).sum()}  Transisi: {n_trans}")

    print("\n  Membuat gambar...")
    plot_timeline_full(df, is_st)
    plot_cycle_zoom(df, is_st)
    plot_soc_separate(df, is_st)

    print("\n" + "═"*60)
    print("  SELESAI - gambar tersimpan di", OUTPUT_DIR)
    print("    hybrid_timeline_full.png")
    print("    hybrid_cycle_zoom.png")
    print("    hybrid_soc_separate.png")
    print("═"*60)


if __name__ == '__main__':
    main()