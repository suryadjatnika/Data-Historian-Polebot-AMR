#!/usr/bin/env python3
"""
hybrid_cycle_annotation.py
═══════════════════════════════════════════════════════════════════════
Menampilkan SATU SIKLUS PENUH Skenario C7 (Blok A–E) secara detail.
Cocok untuk presentasi ke dospem — menunjukkan ARIMA ↔ XGBoost
bergantian dalam satu siklus ~82 detik.

Output: ~/polebot_hybrid_results/hybrid_one_cycle_annotated.png
═══════════════════════════════════════════════════════════════════════
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from influxdb_client import InfluxDBClient

# ── Konfigurasi ──────────────────────────────────────────────────────────────
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_data"

# Ambil data 1 siklus + buffer dari menit ke-5 sampai ke-8 skenario C7
# (menit 5 = 00:01:01 WIB + 5 menit = 00:06:01 WIB = 2026-06-10T17:06:01Z)
# Window: 3 siklus agar terlihat polanya, fokus annotate siklus tengah
DATA_RANGE = "start: 2026-06-10T17:05:00Z, stop: 2026-06-10T17:11:00Z"
OUTPUT_DIR = os.path.expanduser("~/polebot_hybrid_results")

# Threshold klasifikasi (sama dengan predictor)
ACCEL_THR  = 0.15
SPEED_THR  = 0.60

# Warna (konsisten dengan file sebelumnya)
C_STATIC   = '#003087'
C_DYNAMIC  = '#CC0000'
C_HYBRID   = '#1A7A4A'
C_BG_ST    = '#C8DCF5'
C_BG_DY    = '#F5C8C8'

# Warna dan deskripsi setiap blok
BLOKS = [
    {
        'name'    : 'A',
        'label'   : 'Block A',
        'kondisi' : 'STATIC',
        'model'   : 'ARIMA active',
        'deskripsi': 'Creep\n0.15 m/s\nconstant',
        'color'   : '#003087',
        'bg'      : '#D6E8FF',
        'txt'     : 'white',
    },
    {
        'name'    : 'B',
        'label'   : 'Block B',
        'kondisi' : 'DYNAMIC',
        'model'   : 'XGBoost active',
        'deskripsi': 'Sprint 0.65 m/s\n+ Turn 180°\n+ Sprint back',
        'color'   : '#CC0000',
        'bg'      : '#FFD6D6',
        'txt'     : 'white',
    },
    {
        'name'    : 'C',
        'label'   : 'Block C',
        'kondisi' : 'STATIC',
        'model'   : 'ARIMA active',
        'deskripsi': 'Turn 90°\n→ Constant\n0.25 m/s',
        'color'   : '#003087',
        'bg'      : '#D6E8FF',
        'txt'     : 'white',
    },
    {
        'name'    : 'D',
        'label'   : 'Block D',
        'kondisi' : 'DYNAMIC',
        'model'   : 'XGBoost active',
        'deskripsi': 'Burst 0.70 m/s\n+ Turn 180°\n+ Burst back',
        'color'   : '#CC0000',
        'bg'      : '#FFD6D6',
        'txt'     : 'white',
    },
    {
        'name'    : 'E',
        'label'   : 'Block E',
        'kondisi' : 'STATIC',
        'model'   : 'ARIMA active',
        'deskripsi': 'Creep 0.10 m/s\n(slowest phase)\n+ Turn 90°',
        'color'   : '#003087',
        'bg'      : '#D6E8FF',
        'txt'     : 'white',
    },
]

# Durasi tiap blok dalam satu siklus (detik) — sesuai desain scenario_runner.py
BLOK_DURATIONS = {
    'A': 13.5,   # creep 0.15 m/s (12s) + stop (1.5s)
    'B': 16.4,   # sprint(2)+stop(1.5)+turn180(6.4)+stop(1)+sprint(2)+stop(2)+turn90(3.2)+stop(1) → sebagian
    'C': 14.0,   # actual: stop1+konstan(8)+stop1.5 + sisa from B
    'D': 13.5,   # burst(1.5)+stop(2)+turn180(6.4)+stop(0.5)+burst(1.5)+stop(2)
    'E': 24.0,   # turn90(3.2)+stop(1)+creep(15)+stop(1.5)+turn90(3.2)+stop(1)
}

def fetch_cycle_data():
    """Ambil data 6 menit dari InfluxDB untuk menampilkan 2-3 siklus."""
    print("Mengambil data satu siklus dari InfluxDB...")
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    fields = ['joint_P_total', 'batt_soc_percent', 'odom_v_linear',
              'odom_accel', 'joint_load_ratio', 'batt_power_draw']
    ff = ' or '.join([f'r._field == "{f}"' for f in fields])
    q = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn:(r) => r._measurement == "polebot_telemetry")
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
    df.drop(columns=['_time'], inplace=True, errors='ignore')
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.select_dtypes(include=['number'])
    df = df.interpolate(method='time', limit=5).ffill().bfill()
    if 'odom_accel' not in df.columns:
        df['odom_accel'] = df['odom_v_linear'].diff().fillna(0.0)

    print(f"  ✅ {len(df)} rekaman "
          f"({df.index[0].strftime('%H:%M:%S')} – {df.index[-1].strftime('%H:%M:%S')})")
    return df


def find_cycle_start(df):
    """
    Cari titik awal siklus yang representatif:
    - Statis minimal 8 detik berturut-turut (menandakan awal Blok A)
    - Diikuti oleh kondisi dinamis (sprint)
    Kembalikan index dari awal siklus tersebut.
    """
    is_st = (df['odom_accel'].abs() < ACCEL_THR) & (df['odom_v_linear'].abs() < SPEED_THR)

    # Cari blok statis yang cukup panjang (Blok A: creep konstan >= 8 detik)
    WINDOW = 8
    for i in range(len(is_st) - WINDOW - 20):
        # Periksa apakah window statis cukup panjang
        if is_st.iloc[i:i+WINDOW].all():
            # Periksa apakah setelah window ada kondisi dinamis (sprint)
            dyn_after = is_st.iloc[i+WINDOW:i+WINDOW+10]
            if (~dyn_after).any():
                return i
    return 0  # fallback: gunakan awal data


def draw_blok_annotation(ax, t_start, t_end, blok_info, y_min, y_max,
                          y_frac_top=0.98, y_frac_label=0.88, y_frac_desc=0.75):
    """
    Gambar satu blok anotasi:
    - Background shading
    - Header bar di atas dengan nama blok
    - Label kondisi + deskripsi
    - Garis batas kiri
    """
    color   = blok_info['color']
    bg      = blok_info['bg']
    name    = blok_info['label']
    kondisi = blok_info['kondisi']
    model   = blok_info['model']
    deskr   = blok_info['deskripsi']

    # Background shading (transparan)
    ax.axvspan(t_start, t_end, alpha=0.18, color=bg, zorder=0, linewidth=0)

    # Header bar (atas panel)
    h_span = y_max - y_min
    h_bar_top  = y_min + h_span * y_frac_top
    h_bar_bot  = y_min + h_span * (y_frac_top - 0.07)

    ax.fill_betweenx([h_bar_bot, h_bar_top], t_start, t_end,
                     color=color, alpha=0.85, zorder=4)

    # Nama blok di header bar
    t_mid = t_start + (t_end - t_start) / 2
    ax.text(t_mid, (h_bar_top + h_bar_bot) / 2, name,
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', zorder=5)

    # Kondisi + model
    ax.text(t_mid, y_min + h_span * y_frac_label,
            f'{kondisi}\n{model}',
            ha='center', va='top', fontsize=8.5,
            color=color, fontweight='bold', zorder=5,
            bbox=dict(facecolor='white', alpha=0.75,
                      edgecolor=color, boxstyle='round,pad=0.2', linewidth=1.2))

    # Deskripsi fase (italic, di bawah)
    ax.text(t_mid, y_min + h_span * y_frac_desc,
            deskr,
            ha='center', va='top', fontsize=7.5,
            color='#333333', style='italic', zorder=5,
            multialignment='center')

    # Garis batas kiri blok
    ax.axvline(x=t_start, color=color, lw=1.8,
               linestyle='-', alpha=0.7, zorder=3)


def plot_one_cycle(df):
    """Buat figure satu siklus dengan anotasi lengkap."""
    plt.rcParams.update({
        'font.family' : 'serif',
        'font.serif'  : ['Times New Roman', 'DejaVu Serif'],
        'font.size'   : 10,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })

    is_static = ((df['odom_accel'].abs() < ACCEL_THR) &
                 (df['odom_v_linear'].abs() < SPEED_THR))

    # ── Temukan siklus representatif ──────────────────────────
    i_start = find_cycle_start(df)
    # Ambil ~90 detik mulai dari i_start (satu siklus + sedikit buffer)
    i_end = min(i_start + 95, len(df) - 1)
    df_c  = df.iloc[i_start:i_end].copy()
    is_c  = is_static.iloc[i_start:i_end].copy()

    print(f"  Siklus yang ditampilkan: "
          f"{df_c.index[0].strftime('%H:%M:%S')} – "
          f"{df_c.index[-1].strftime('%H:%M:%S')} "
          f"({len(df_c)} detik)")

    t_base = df_c.index[0]  # waktu awal siklus

    # ── Hitung batas blok berdasarkan durasi desain ────────────
    blok_bounds = []
    t_cur = t_base
    for blok in BLOKS:
        dur   = BLOK_DURATIONS[blok['name']]
        t_end = t_cur + pd.Timedelta(seconds=dur)
        blok_bounds.append((t_cur, t_end, blok))
        t_cur = t_end
    t_siklus_end = t_cur

    # ── Setup figure ───────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    panels = [
        ('batt_soc_percent', 'Battery State of Charge (SOC)', '%'),
        ('joint_P_total',    'Total Motor Power (P_total)',     'Watt'),
        ('odom_v_linear',    'Linear Velocity',                 'm/s'),
    ]

    tfmt = mdates.DateFormatter('%H:%M:%S')

    for row, (field, title, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[row])
        ax.set_facecolor('white')

        if field not in df_c.columns:
            ax.text(0.5, 0.5, f'{field} tidak tersedia',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        series = df_c[field].dropna()
        if series.empty:
            continue

        # Y range dengan margin
        y_min = series.min(); y_max = series.max()
        margin = max((y_max - y_min) * 0.25, 0.1)
        y_lo   = y_min - margin * 0.3
        y_hi   = y_max + margin * 1.5  # ruang lebih di atas untuk anotasi

        # ── Gambar anotasi blok (background + header) ─────────
        for t_s, t_e, blok in blok_bounds:
            if t_s <= series.index[-1] and t_e >= series.index[0]:
                t_s_clip = max(t_s, series.index[0])
                t_e_clip = min(t_e, series.index[-1])
                draw_blok_annotation(ax, t_s_clip, t_e_clip, blok,
                                     y_lo, y_hi,
                                     y_frac_top=0.97,
                                     y_frac_label=0.85,
                                     y_frac_desc=0.70)

        # ── Garis penutup siklus ───────────────────────────────
        if t_siklus_end <= series.index[-1]:
            ax.axvline(x=t_siklus_end, color='#333333', lw=2.5,
                       linestyle='--', alpha=0.8, zorder=3)
            ax.text(t_siklus_end, y_lo + (y_hi-y_lo)*0.02,
                    ' ← End of cycle', fontsize=8,
                    color='#333333', va='bottom')

        # ── Plot data aktual berwarna ──────────────────────────
        idx = series.index
        arr = is_c.reindex(idx).fillna(True).values
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and arr[j] == arr[i]: j += 1
            end = min(j + 1, len(idx))
            c = C_STATIC if arr[i] else C_DYNAMIC
            ax.plot(idx[i:end], series.iloc[i:end],
                    color=c, linewidth=2.0, solid_capstyle='butt', zorder=6)
            i = j

        # ── Formatting ────────────────────────────────────────
        ax.set_title(title, fontsize=12, fontweight='bold', pad=5)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_xlim(series.index[0], series.index[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_formatter(tfmt)
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        ax.grid(True, color='#D8D8D8', alpha=0.5, lw=0.6, zorder=1)
        for sp in ax.spines.values(): sp.set_color('#CCCCCC')
        ax.tick_params(colors='#333333')

    # ── Legenda global ─────────────────────────────────────────
    leg_items = [
        mpatches.Patch(color=C_STATIC,  label='Actual Data — STATIC Condition (ARIMA active)'),
        mpatches.Patch(color=C_DYNAMIC, label='Actual Data — DYNAMIC Condition (XGBoost active)'),
        mpatches.Patch(color='#D6E8FF', label='STATIC Zone — Block A, C, E'),
        mpatches.Patch(color='#FFD6D6', label='DYNAMIC Zone — Block B, D'),
    ]
    fig.legend(handles=leg_items, loc='lower center',
               ncol=4, fontsize=9.5, framealpha=0.95,
               edgecolor='#BBBBBB', bbox_to_anchor=(0.5, -0.01))

    # ── Diagram siklus di bawah judul ──────────────────────────
    blok_summary = '  →  '.join([
        f'[{b["name"]}] {b["kondisi"][:3]}'
        for b in BLOKS
    ])

    fig.suptitle(
        'One Full Cycle — Scenario C7: Condition-Based Temporal Switching\n'
        f'CYCLE: {blok_summary}   (~82 seconds total)',
        fontsize=12, fontweight='bold', y=1.02
    )

    # ── Diagram mini siklus di bawah judul ────────────────────
    # (bar horizontal yang menunjukkan urutan blok)
    fig_ax_top = fig.add_axes([0.065, 0.955, 0.915, 0.025])
    fig_ax_top.set_xlim(0, 1); fig_ax_top.set_ylim(0, 1)
    fig_ax_top.axis('off')

    total_dur = sum(BLOK_DURATIONS.values())
    x_cur = 0
    for blok in BLOKS:
        dur_frac = BLOK_DURATIONS[blok['name']] / total_dur
        rect = plt.Rectangle((x_cur, 0), dur_frac, 1,
                              facecolor=blok['color'], alpha=0.85)
        fig_ax_top.add_patch(rect)
        fig_ax_top.text(x_cur + dur_frac/2, 0.5,
                        f"{blok['label']} ({BLOK_DURATIONS[blok['name']]:.0f}s)",
                        ha='center', va='center', fontsize=8.5,
                        fontweight='bold', color='white')
        x_cur += dur_frac

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'hybrid_one_cycle_annotated.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  ✅ Chart tersimpan: {out}")
    return out


if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  CYCLE ANNOTATION — SATU SIKLUS C7")
    print("═"*60)
    df = fetch_cycle_data()
    if df.empty:
        print("❌ Data kosong — periksa DATA_RANGE dan koneksi InfluxDB")
    else:
        plot_one_cycle(df)
    print("═"*60)