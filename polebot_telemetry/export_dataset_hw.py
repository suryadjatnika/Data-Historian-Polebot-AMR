#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Konfigurasi
INPUT_DIR  = os.path.expanduser("~/csv_skenario17_hw")
OUTPUT_DIR = os.path.expanduser("~/polebot_hybrid_results_hw")
SKIPROWS   = 3
ACCEL_THR  = 0.10
SPEED_THR  = 0.18

# Metadata 7 skenario
CONDITIONS = [
    {'id':1,'name':'baseline',   'label':'Baseline',      'v_desc':'~0.13 m/s (maks 0.26)', 'class':'Statis'},
    {'id':2,'name':'high_load',  'label':'High Load',     'v_desc':'~0.14 m/s (maks 0.26)', 'class':'Dinamis'},
    {'id':3,'name':'stop_and_go','label':'Stop-and-Go',   'v_desc':'~0.14 m/s (maks 0.26)', 'class':'Dinamis'},
    {'id':4,'name':'creep',      'label':'Creep',         'v_desc':'~0.08 m/s (maks 0.10)', 'class':'Statis'},
    {'id':5,'name':'burst',      'label':'Burst',         'v_desc':'~0.12 m/s (maks 0.26)', 'class':'Dinamis'},
    {'id':6,'name':'mixed',      'label':'Mixed',         'v_desc':'~0.09 m/s (maks 0.18)', 'class':'Campuran'},
    {'id':7,'name':'switching',  'label':'Switching',     'v_desc':'~0.12 m/s (maks 0.26)', 'class':'Campuran'},
]

NUM_FIELDS = ['odom_v_linear','odom_accel','joint_P_total','joint_load_ratio',
              'batt_soc_percent','batt_voltage','batt_current','batt_power_draw']


def load_scenario(cid):
    """Baca satu kondisi_N.csv, bersihkan, beri label."""
    path = os.path.join(INPUT_DIR, f'kondisi_{cid}.csv')
    if not os.path.exists(path):
        print(f"{path} tidak ada, dilewati.")
        return None
    df = pd.read_csv(path, skiprows=SKIPROWS)

    # Buang data inisialisasi (51.2V default) & anomali tegangan
    vt = pd.to_numeric(df['batt_voltage'], errors='coerce')
    df = df[(vt >= 40) & (vt <= 50)].copy()

    # Konversi numerik
    for c in NUM_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['odom_v_linear','joint_P_total','batt_soc_percent'])

    # Waktu relatif
    df['_time'] = pd.to_datetime(df['_time'])
    df = df.sort_values('_time').reset_index(drop=True)
    df['t_rel'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()

    # Label kondisi statis/dinamis per baris
    v = df['odom_v_linear'].abs()
    a = df['odom_accel'].abs() if 'odom_accel' in df.columns else pd.Series(0, index=df.index)
    df['kondisi'] = np.where((v < SPEED_THR) & (a < ACCEL_THR), 'STATIS', 'DINAMIS')

    # Metadata skenario
    meta = next(c for c in CONDITIONS if c['id'] == cid)
    df['skenario_id']    = cid
    df['skenario_nama']  = meta['name']
    df['skenario_kelas'] = meta['class']
    return df


def build_statistics(all_df):
    """Tabel statistik ringkas per skenario."""
    rows = []
    for cid in range(1, 8):
        sub = all_df[all_df['skenario_id'] == cid]
        if sub.empty:
            continue
        meta = next(c for c in CONDITIONS if c['id'] == cid)
        v = sub['odom_v_linear'].abs()
        p = sub['joint_P_total']
        soc = sub['batt_soc_percent']
        n_dyn = (sub['kondisi'] == 'DINAMIS').sum()
        rows.append({
            'Skenario'      : cid,
            'Nama'          : meta['label'],
            'Kelas'         : meta['class'],
            'Durasi_s'      : round(sub['t_rel'].max(), 0),
            'v_mean_m/s'    : round(v.mean(), 3),
            'v_max_m/s'     : round(v.max(), 3),
            'P_mean_W'      : round(p.mean(), 3),
            'P_max_W'       : round(p.max(), 3),
            'SOC_awal_%'    : round(soc.iloc[0], 2),
            'SOC_akhir_%'   : round(soc.iloc[-1], 2),
            'SOC_turun_%'   : round(soc.iloc[0] - soc.iloc[-1], 2),
            'Dinamis_%'     : round(100 * n_dyn / len(sub), 1),
            'Total_baris'   : len(sub),
        })
    return pd.DataFrame(rows)


def plot_overview(all_df, stats):
    """Grafik overview multi-panel: perbandingan 7 skenario."""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':10,'figure.facecolor':'white','savefig.dpi':300})

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ids    = stats['Skenario'].values
    labels = stats['Nama'].values
    x = np.arange(len(ids))

    # Warna per kelas
    kelas_warna = {'Statis':'#2166AC','Dinamis':'#CC0000','Campuran':'#F59E0B'}
    bar_colors = [kelas_warna[k] for k in stats['Kelas'].values]

    # Panel 1: Kecepatan rata-rata & maks
    ax = axes[0,0]
    ax.bar(x-0.2, stats['v_mean_m/s'], 0.4, label='Rata-rata', color=bar_colors, alpha=0.6)
    ax.bar(x+0.2, stats['v_max_m/s'], 0.4, label='Maksimum', color=bar_colors, alpha=1.0)
    ax.axhline(SPEED_THR, color='k', ls='--', lw=1, alpha=0.6, label=f'Ambang {SPEED_THR}')
    ax.set_ylabel('Kecepatan (m/s)'); ax.set_title('Perbandingan Kecepatan per Skenario', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Panel 2: Daya motor rata-rata & maks
    ax = axes[0,1]
    ax.bar(x-0.2, stats['P_mean_W'], 0.4, label='Rata-rata', color=bar_colors, alpha=0.6)
    ax.bar(x+0.2, stats['P_max_W'], 0.4, label='Maksimum', color=bar_colors, alpha=1.0)
    ax.set_ylabel('Daya Motor (W)'); ax.set_title('Perbandingan Daya Motor per Skenario', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Panel 3: Proporsi dinamis
    ax = axes[1,0]
    bars = ax.bar(x, stats['Dinamis_%'], 0.6, color=bar_colors)
    ax.set_ylabel('Proporsi Dinamis (%)'); ax.set_title('Proporsi Kondisi Dinamis per Skenario', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, stats['Dinamis_%']):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.0f}%', ha='center', fontsize=8)

    # Panel 4: Penurunan SOC
    ax = axes[1,1]
    ax.bar(x, stats['SOC_turun_%'], 0.6, color=bar_colors)
    ax.set_ylabel('Penurunan SOC (%)'); ax.set_title('Penurunan SOC per Skenario', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Legenda kelas
    from matplotlib.patches import Patch
    leg = [Patch(color=c, label=k) for k, c in kelas_warna.items()]
    fig.legend(handles=leg, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.99), frameon=True)

    fig.suptitle('Overview Dataset 7 Skenario Operasional',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, 'dataset_overview.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  ✅ {fname}")


def main():
    print("\n" + "═"*60)
    print("  EXPORT DATASET HARDWARE 7 SKENARIO")
    print("═"*60)

    # 1. Baca semua skenario
    print("\n  Membaca file kondisi_1.csv ... kondisi_7.csv...")
    frames = []
    for cid in range(1, 8):
        df = load_scenario(cid)
        if df is not None:
            n_dyn = (df['kondisi']=='DINAMIS').sum()
            print(f"    Skenario {cid}: {len(df)} baris (dinamis {100*n_dyn/len(df):.0f}%)")
            frames.append(df)

    if not frames:
        print("Tidak ada data. Pastikan file kondisi_N.csv ada di", INPUT_DIR)
        return

    all_df = pd.concat(frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Simpan CSV gabungan berlabel
    combined_path = os.path.join(OUTPUT_DIR, 'dataset_gabungan_berlabel.csv')
    all_df.to_csv(combined_path, index=False, float_format='%.6f')
    print(f"\n CSV gabungan: {combined_path} ({len(all_df)} baris)")

    # 3. Tabel statistik
    stats = build_statistics(all_df)
    stats_path = os.path.join(OUTPUT_DIR, 'dataset_statistik.csv')
    stats.to_csv(stats_path, index=False)
    print(f"Tabel statistik: {stats_path}")
    print("\n  Ringkasan statistik per skenario:")
    print(stats.to_string(index=False))

    # 4. Grafik overview
    print("\n  Membuat grafik overview...")
    plot_overview(all_df, stats)

    print("\n" + "═"*60)
    print("  SELESAI - 3 file di", OUTPUT_DIR)
    print("    dataset_gabungan_berlabel.csv")
    print("    dataset_statistik.csv")
    print("    dataset_overview.png")
    print("═"*60)


if __name__ == '__main__':
    main()