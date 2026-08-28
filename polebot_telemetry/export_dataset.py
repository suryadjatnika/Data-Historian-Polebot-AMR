#!/usr/bin/env python3
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
import os
from datetime import datetime

# ── Konfigurasi ────────────────────────────────────────────────
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_data"
OUTPUT_DIR      = os.path.expanduser("~/polebot_dataset")
DATA_RANGE      = "start: 2026-05-31T18:28:00Z, stop: 2026-05-31T21:33:00Z"

# ── Definisi 6 kondisi operasional ────────────────────────────
# Berdasarkan rentang waktu eksekusi overnight (02:07 - 05:09 WIB)
# Setiap kondisi 30 menit = 1800 detik
# Kamu perlu sesuaikan START_TIME dan END_TIME dengan waktu aktual
# dari log run_all_scenarios.sh kamu

CONDITIONS = [
    {
        'id'         : 1,
        'name'       : 'baseline',
        'label'      : 'Kondisi 1 - Baseline (Normal)',
        'velocity'   : '0.3 m/s constant',
        'class'      : 'Static',
        'duration_s' : 1800,
    },
    {
        'id'         : 2,
        'name'       : 'high_load',
        'label'      : 'Kondisi 2 - High Load',
        'velocity'   : '0.7 m/s constant',
        'class'      : 'Dynamic',
        'duration_s' : 1800,
    },
    {
        'id'         : 3,
        'name'       : 'stop_and_go',
        'label'      : 'Kondisi 3 - Stop and Go',
        'velocity'   : '0.5 m/s variable',
        'class'      : 'Dynamic',
        'duration_s' : 1800,
    },
    {
        'id'         : 4,
        'name'       : 'creep',
        'label'      : 'Kondisi 4 - Creep (Sangat Lambat)',
        'velocity'   : '0.1 m/s constant',
        'class'      : 'Static',
        'duration_s' : 1800,
    },
    {
        'id'         : 5,
        'name'       : 'burst',
        'label'      : 'Kondisi 5 - Burst (Akselerasi Agresif)',
        'velocity'   : '0.6 m/s with rapid acceleration',
        'class'      : 'Dynamic',
        'duration_s' : 1800,
    },
    {
        'id'         : 6,
        'name'       : 'mixed',
        'label'      : 'Kondisi 6 - Mixed (Campuran)',
        'velocity'   : 'Variable 0.0–0.6 m/s',
        'class'      : 'Mixed',
        'duration_s' : 1800,
    },
]

# Kolom yang akan diekspor ke CSV dataset
EXPORT_FIELDS = [
    'odom_v_linear',      # kecepatan linear (m/s)
    'odom_omega',         # kecepatan angular (rad/s)
    'odom_accel',         # akselerasi linear (m/s²)
    'joint_P_total',      # total daya motor (W)
    'joint_load_ratio',   # rasio beban motor (0-1)
    'batt_soc_percent',   # State of Charge (%)
    'batt_voltage',       # tegangan baterai (V)
    'batt_current',       # arus baterai (A)
    'batt_power_draw',    # daya listrik sesaat (W)
]


def fetch_all_fields() -> pd.DataFrame:
    """
    Ambil semua field sekaligus dari InfluxDB dengan query pivot.
    Hasilnya DataFrame lebar: setiap kolom = satu sensor.
    """
    print("Mengambil semua data dari InfluxDB...")

    fields_filter = ' or '.join(
        [f'r._field == "{f}"' for f in EXPORT_FIELDS])

    query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => {fields_filter})
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns: ["_time"])
'''

    client = InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    df = client.query_api().query_data_frame(query)
    client.close()

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()

    if df.empty:
        print("⚠️  Data kosong!")
        return None

    # Bersihkan kolom metadata InfluxDB
    keep = ['_time'] + [f for f in EXPORT_FIELDS if f in df.columns]
    df = df[keep].copy()

    # Konversi waktu ke WIB
    df['_time'] = pd.to_datetime(df['_time'], utc=True)
    df['_time'] = df['_time'].dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    df = df.set_index('_time').sort_index()

    # Konversi semua kolom ke numerik
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(how='all')

    print(f"Total data: {len(df)} titik | "
          f"{df.index[0].strftime('%H:%M:%S')} - "
          f"{df.index[-1].strftime('%H:%M:%S')} WIB")
    return df


def split_by_condition(df: pd.DataFrame) -> dict:
    """
    Pisahkan data berdasarkan urutan waktu.
    Setiap kondisi = 1800 detik berurutan.
    Kondisi 1 dimulai dari awal data.
    Offset 20 detik per jeda antar skenario diperhitungkan.
    """
    result = {}
    total_points = len(df)
    points_per_condition = 1800  # 1800 detik @ 1 Hz
    jeda_s = 20                  # harus sama dengan JEDA_ANTAR_SKENARIO di run_all_scenarios.sh

    print(f"\nMemisahkan data ke {len(CONDITIONS)} kondisi...")
    print(f"Total data: {total_points} titik")
    print(f"Offset jeda antar skenario: {jeda_s} detik")

    for i, cond in enumerate(CONDITIONS):
        offset    = i * jeda_s
        start_idx = i * points_per_condition + offset
        end_idx   = start_idx + points_per_condition

        if start_idx >= total_points:
            print(f"  ⚠️  Kondisi {cond['id']} tidak ada datanya")
            continue

        subset = df.iloc[start_idx:min(end_idx, total_points)].copy()

        # Tambah kolom keterangan
        subset['condition_id']    = cond['id']
        subset['condition_name']  = cond['name']
        subset['condition_class'] = cond['class']
        subset['timestep']        = range(len(subset))

        result[cond['id']] = {
            'data'  : subset,
            'config': cond,
        }
        print(f"  Kondisi {cond['id']} ({cond['name']}): "
              f"{len(subset)} titik | "
              f"{subset.index[0].strftime('%H:%M:%S')} - "
              f"{subset.index[-1].strftime('%H:%M:%S')}")

    return result


def save_datasets(conditions_data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []

    for cond_id, item in conditions_data.items():
        df_cond = item['data']
        config  = item['config']

        # Nama kolom yang lebih deskriptif untuk dataset
        rename_map = {
            'odom_v_linear'   : 'linear_velocity_ms',
            'odom_omega'      : 'angular_velocity_rads',
            'odom_accel'      : 'linear_acceleration_ms2',
            'joint_P_total'   : 'motor_power_total_W',
            'joint_load_ratio': 'motor_load_ratio',
            'batt_soc_percent': 'battery_SOC_percent',
            'batt_voltage'    : 'battery_voltage_V',
            'batt_current'    : 'battery_current_A',
            'batt_power_draw' : 'battery_power_draw_W',
        }
        df_export = df_cond.rename(columns=rename_map)

        # Simpan CSV
        filename = f"kondisi_{cond_id}_{config['name']}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_export.to_csv(filepath, index=True, float_format='%.6f')
        print(f"  Tersimpan: {filename} ({len(df_export)} baris)")

        # Statistik untuk summary
        soc_col = 'battery_SOC_percent'
        pwr_col = 'motor_power_total_W'
        vel_col = 'linear_velocity_ms'

        summary_rows.append({
            'condition_id'         : cond_id,
            'condition_name'       : config['name'],
            'class'                : config['class'],
            'velocity_profile'     : config['velocity'],
            'n_samples'            : len(df_export),
            'duration_seconds'     : len(df_export),
            'SOC_initial_pct'      : df_export[soc_col].iloc[0] if soc_col in df_export else '-',
            'SOC_final_pct'        : df_export[soc_col].iloc[-1] if soc_col in df_export else '-',
            'SOC_drop_pct'         : (df_export[soc_col].iloc[0] - df_export[soc_col].iloc[-1])
                                      if soc_col in df_export else '-',
            'motor_power_mean_W'   : df_export[pwr_col].mean() if pwr_col in df_export else '-',
            'motor_power_max_W'    : df_export[pwr_col].max()  if pwr_col in df_export else '-',
            'velocity_mean_ms'     : df_export[vel_col].mean() if vel_col in df_export else '-',
            'velocity_max_ms'      : df_export[vel_col].max()  if vel_col in df_export else '-',
        })

    # Simpan overview CSV
    df_summary = pd.DataFrame(summary_rows)
    overview_path = os.path.join(OUTPUT_DIR, 'dataset_overview.csv')
    df_summary.to_csv(overview_path, index=False, float_format='%.4f')
    print(f"\n  Overview tersimpan: dataset_overview.csv")

    # Simpan summary TXT
    txt_path = os.path.join(OUTPUT_DIR, 'dataset_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("DATASET — Polebot AMR Energy Consumption\n")
        f.write(f"Diekspor: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total kondisi  : {len(summary_rows)}\n")
        f.write(f"Total sampel   : {sum(r['n_samples'] for r in summary_rows)}\n")
        f.write(f"Frekuensi      : 1 Hz (1 sampel/detik setelah resampling)\n")
        f.write(f"Fitur per baris: {len(EXPORT_FIELDS)} variabel sensor\n\n")
        f.write("Deskripsi Kondisi:\n")
        f.write("-" * 65 + "\n")
        for r in summary_rows:
            f.write(f"Kondisi {r['condition_id']} ({r['condition_name'].upper()})\n")
            f.write(f"  Kelas          : {r['class']}\n")
            f.write(f"  Profil kecepatan: {r['velocity_profile']}\n")
            f.write(f"  Jumlah sampel  : {r['n_samples']}\n")
            f.write(f"  SOC awal       : {r['SOC_initial_pct']:.2f}%\n")
            f.write(f"  SOC akhir      : {r['SOC_final_pct']:.2f}%\n")
            f.write(f"  Penurunan SOC  : {r['SOC_drop_pct']:.2f}%\n")
            f.write(f"  Daya motor rata: {r['motor_power_mean_W']:.3f} W\n")
            f.write("-" * 65 + "\n")

    print(f"  Summary tersimpan: dataset_summary.txt")
    return df_summary


def plot_condition_statistics(conditions_data: dict):
    """
    Grafik statistik per kondisi operasional — gaya jurnal akademis.

    Layout: 2 baris × 2 kolom (4 panel)
      (0,0) Battery SOC  — ΔSOC time series per kondisi
      (0,1) Battery SOC  — SOC drop total per kondisi (bar)
      (1,0) Motor Power  — mean per kondisi (bar)
      (1,1) Kecepatan    — mean per kondisi (bar)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        'font.family'      : 'serif',
        'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
        'font.size'        : 10,
        'axes.titlesize'   : 10.5,
        'axes.labelsize'   : 9.5,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
        'legend.fontsize'  : 8,
        'figure.dpi'       : 300,
        'savefig.dpi'      : 300,
        'axes.linewidth'   : 0.8,
        'axes.facecolor'   : 'white',
        'figure.facecolor' : 'white',
        'axes.edgecolor'   : '#AAAAAA',
    })

    COND_COLORS = {
        1: '#2166AC',   # biru tua  — Baseline
        2: '#D7191C',   # merah     — High Load
        3: '#F46D43',   # oranye    — Stop-and-Go
        4: '#1A9641',   # hijau tua — Creep
        5: '#762A83',   # ungu      — Burst
        6: '#B5770D',   # coklat    — Mixed
    }
    COND_LABELS = {
        1: 'C1: Baseline (0.3 m/s)',
        2: 'C2: High Load (0.7 m/s)',
        3: 'C3: Stop-and-Go (0.5 m/s)',
        4: 'C4: Creep (0.1 m/s)',
        5: 'C5: Burst (0.6 m/s)',
        6: 'C6: Mixed (variable)',
    }

    # ── Kumpulkan data per kondisi ───────────────────────────────
    cond_ids      = sorted(conditions_data.keys())
    bar_colors    = [COND_COLORS.get(i, '#333333') for i in cond_ids]
    x_labels      = [f'C{i}' for i in cond_ids]
    full_labels   = [COND_LABELS.get(i, f'C{i}') for i in cond_ids]

    soc_drops    = []   # total SOC drop per kondisi
    soc_deltas   = []   # series ΔSOC untuk time series
    pow_means    = []   # mean motor power per kondisi
    vel_means    = []   # mean |velocity| per kondisi

    for cond_id in cond_ids:
        df_c = conditions_data[cond_id]['data']

        # SOC — coba kedua nama kolom
        soc_col = next((c for c in ['batt_soc_percent', 'battery_SOC_percent']
                        if c in df_c.columns), None)
        if soc_col:
            s = df_c[soc_col].dropna()
            soc_drops.append(abs(float(s.iloc[-1]) - float(s.iloc[0])))
            soc_deltas.append(s.values - s.values[0])
        else:
            soc_drops.append(0.0)
            soc_deltas.append(np.zeros(1800))

        # Motor power
        pow_col = next((c for c in ['joint_P_total', 'motor_power_total_W']
                        if c in df_c.columns), None)
        pow_means.append(float(df_c[pow_col].dropna().mean()) if pow_col else 0.0)

        # Velocity (mean absolute — ada negatif saat robot mundur)
        vel_col = next((c for c in ['odom_v_linear', 'linear_velocity_ms']
                        if c in df_c.columns), None)
        vel_means.append(float(df_c[vel_col].dropna().abs().mean()) if vel_col else 0.0)

    x_pos = np.arange(len(cond_ids))

    # ── Buat figure 2×2 ──────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30)

    # ── Panel (0,0): SOC time series ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (cond_id, delta) in enumerate(zip(cond_ids, soc_deltas)):
        ax1.plot(np.arange(len(delta)), delta,
                 color=bar_colors[i], linewidth=0.9, alpha=0.85,
                 label=full_labels[i])
    ax1.set_title('Battery State of Charge\nTime Series per Condition',
                  fontsize=10.5, fontweight='bold', pad=5)
    ax1.set_xlabel('Time (seconds)', fontsize=9.5)
    ax1.set_ylabel('ΔSOC from condition start (%)', fontsize=9.5)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
               fontsize=7.5, loc='lower left')
    ax1.grid(True, color='#CCCCCC', alpha=0.4, linewidth=0.5)
    for sp in ax1.spines.values():
        sp.set_edgecolor('#AAAAAA')

    # ── Panel (0,1): SOC drop bar ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x_pos, soc_drops, color=bar_colors,
                    alpha=0.80, edgecolor='#333333',
                    linewidth=0.7, width=0.58)
    for bar, val in zip(bars2, soc_drops):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(soc_drops) * 0.02,
                 f'{val:.2f}',
                 ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#222222')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_ylim(0, max(soc_drops) * 1.22)
    ax2.set_title('Battery State of Charge\nStatistical Summary',
                  fontsize=10.5, fontweight='bold', pad=5)
    ax2.set_xlabel('Operational Condition', fontsize=9.5)
    ax2.set_ylabel('SOC Drop per 30 min (%)', fontsize=9.5)
    ax2.grid(True, axis='y', color='#CCCCCC', alpha=0.4, linewidth=0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#AAAAAA')

    # ── Panel (1,0): Motor power bar ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x_pos, pow_means, color=bar_colors,
                    alpha=0.80, edgecolor='#333333',
                    linewidth=0.7, width=0.58)
    for bar, val in zip(bars3, pow_means):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(pow_means) * 0.02,
                 f'{val:.2f}',
                 ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#222222')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=9)
    ax3.set_ylim(0, max(pow_means) * 1.22)
    ax3.set_title('Total Motor Power\nStatistical Summary',
                  fontsize=10.5, fontweight='bold', pad=5)
    ax3.set_xlabel('Operational Condition', fontsize=9.5)
    ax3.set_ylabel('Mean Total Motor Power (W)', fontsize=9.5)
    ax3.grid(True, axis='y', color='#CCCCCC', alpha=0.4, linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor('#AAAAAA')

    # ── Panel (1,1): Velocity bar ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(x_pos, vel_means, color=bar_colors,
                    alpha=0.80, edgecolor='#333333',
                    linewidth=0.7, width=0.58)
    for bar, val in zip(bars4, vel_means):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(vel_means) * 0.02,
                 f'{val:.3f}',
                 ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#222222')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, fontsize=9)
    ax4.set_ylim(0, max(vel_means) * 1.22)
    ax4.set_title('Linear Velocity\nStatistical Summary',
                  fontsize=10.5, fontweight='bold', pad=5)
    ax4.set_xlabel('Operational Condition', fontsize=9.5)
    ax4.set_ylabel('Mean |Linear Velocity| (m/s)', fontsize=9.5)
    ax4.grid(True, axis='y', color='#CCCCCC', alpha=0.4, linewidth=0.5)
    for sp in ax4.spines.values():
        sp.set_edgecolor('#AAAAAA')

    # ── Judul utama ──────────────────────────────────────────────
    fig.suptitle(
        'Dataset Statistics per Operational Condition — Polebot AMR\n',
        fontsize=11, fontweight='bold', y=1.01)

    # ── Legend bawah figure ──────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COND_COLORS[k], alpha=0.80,
                       label=COND_LABELS[k])
        for k in sorted(COND_COLORS)
    ]
    fig.legend(handles=legend_patches,
               loc='lower center', ncol=3,
               facecolor='white', edgecolor='#CCCCCC',
               fontsize=8.5, bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9)

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    path = os.path.join(OUTPUT_DIR, 'dataset_condition_statistics.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcdefaults()

    print(f"\n  ✅ Grafik statistik tersimpan: {path}")
    return path


def main():
    print("=" * 65)
    print("  Export Dataset — Polebot AMR Energy Consumption")
    print("=" * 65)

    # 1. Ambil semua data dari InfluxDB
    df_all = fetch_all_fields()
    if df_all is None:
        return

    # 2. Pisahkan per kondisi berdasarkan urutan waktu
    conditions_data = split_by_condition(df_all)

    # 3. Simpan ke CSV
    df_summary = save_datasets(conditions_data)

    # 4. Buat grafik statistik per kondisi
    print("\nMembuat grafik statistik per kondisi...")
    plot_condition_statistics(conditions_data)

    print(f"\n{'=' * 65}")
    print(f"  SELESAI — Output tersimpan di: {OUTPUT_DIR}")
    print(f"{'=' * 65}")
    print("  kondisi_1_baseline.csv")
    print("  kondisi_2_high_load.csv")
    print("  kondisi_3_stop_and_go.csv")
    print("  kondisi_4_creep.csv")
    print("  kondisi_5_burst.csv")
    print("  kondisi_6_mixed.csv")
    print("  dataset_overview.csv")
    print("  dataset_summary.txt")
    print("  dataset_condition_statistics.png  ← BARU")
    print()
    print(df_summary[['condition_id', 'condition_name', 'class',
                       'n_samples', 'SOC_drop_pct',
                       'motor_power_mean_W']].to_string(index=False))


if __name__ == '__main__':
    main()