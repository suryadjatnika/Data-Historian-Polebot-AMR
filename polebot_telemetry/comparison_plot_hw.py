#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from datetime import datetime

OUTPUT_DIR = os.path.expanduser("~/polebot_hybrid_results_hw")
ARIMA_FILE = os.path.expanduser("~/arima_results_hw/arima_hw_summary.json")
XGB_FILE   = os.path.expanduser("~/xgboost_results_hw/xgboost_hw_summary.json")

# Warna per metode
C_ARIMA = '#2166AC'
C_XGB   = '#1A9641'

VARIABLES = [
    {'field': 'joint_P_total',    'label': 'Daya Motor Total', 'unit': 'Watt'},
    {'field': 'batt_soc_percent', 'label': 'SOC Baterai',      'unit': '%'},
    {'field': 'odom_v_linear',    'label': 'Kecepatan Linear', 'unit': 'm/s'},
]

# Setup tema
AX_BG   = 'white'
AX_TEXT = '#222222'
AX_GRID = '#CCCCCC'
FIG_BG  = 'white'

import matplotlib as _mpl
_mpl.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'axes.linewidth'   : 0.8,
    'grid.linewidth'   : 0.5,
    'grid.alpha'       : 0.4,
    'grid.color'       : '#CCCCCC',
    'axes.facecolor'   : 'white',
    'figure.facecolor' : 'white',
    'axes.edgecolor'   : '#333333',
})

def load_results():
    def read_json(path, name):
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"{name}: {len(data)} variabel")
            return {r['field']: r for r in data}
        except FileNotFoundError:
            print(f"{name}: file tidak ditemukan ({path})")
            return {}
    arima = read_json(ARIMA_FILE, "ARIMA")
    xgb   = read_json(XGB_FILE,   "XGBoost")
    return arima, xgb


def get_val(d, field, mkey):
    if field not in d:
        return 0.0
    r = d[field]
    if mkey == 'smape':
        return r.get('smape', r.get('mape', 0.0))
    return r.get(mkey, 0.0)

def plot_bar_comparison(arima, xgb):
    """Bar chart komparasi ARIMA vs XGBoost"""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    metrics = [
        ('mae',   'MAE',   ''),
        ('rmse',  'RMSE',  ''),
        ('smape', 'sMAPE', '%'),
    ]
    methods = ['ARIMA', 'XGBoost']
    colors  = [C_ARIMA, C_XGB]
    x       = np.arange(len(methods))

    for row, (mkey, mname, msuffix) in enumerate(metrics):
        for col, var in enumerate(VARIABLES):
            field = var['field']
            unit  = var['unit']
            ax    = axes[row][col]

            vals = [get_val(arima, field, mkey),
                    get_val(xgb,   field, mkey)]

            if all(v == 0 for v in vals):
                ax.set_visible(False)
                continue

            bars = ax.bar(x, vals, color=colors, alpha=0.80,
                          width=0.45, edgecolor='white', linewidth=0.5)
            best = int(np.argmin(vals))
            bars[best].set_edgecolor('#333333')
            bars[best].set_linewidth(2.0)

            for b, v in zip(bars, vals):
                if v > 0:
                    fmt = f'{v:.4f}{msuffix}' if mkey != 'smape' \
                          else f'{v:.2f}{msuffix}'
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + max(vals) * 0.03,
                            fmt, ha='center', va='bottom',
                            fontsize=8.5, color='#333333', fontweight='bold')

            u = '%' if mkey == 'smape' else unit
            ax.set_title(f'{var["label"]}\n{mname} ({u})',
                         fontsize=9.5, fontweight='bold', pad=6)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=9)
            ax.set_facecolor('white')
            ax.tick_params(colors='#333333', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#AAAAAA')
            ax.grid(True, color='#CCCCCC', alpha=0.5,
                    axis='y', linewidth=0.5)
            ax.set_ylim(bottom=0)

    legend_els = [Line2D([0], [0], color=c, lw=10, label=m)
                  for c, m in zip(colors, methods)]
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               facecolor='white', edgecolor='#CCCCCC', fontsize=11,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle(
        'Comparison of ARIMA vs XGBoost\n'
        '[Dark border = best method per variable-metric combination]',
        fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.02, 0.06, 1.0, 0.98])
    path = os.path.join(OUTPUT_DIR, 'comparison_bar_all.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot bar: {path}")

def plot_winner_table(arima, xgb):
    """Tabel ringkas pemenang per variabel per metrik"""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    method_fill = {0: '#CCEEFF', 1: '#CCEECC'}
    methods_lbl = {0: 'ARIMA', 1: 'XGBoost'}

    rows, row_colors = [], []
    for var in VARIABLES:
        field = var['field']
        row, rcolor = [], []
        for mkey in ['mae', 'rmse', 'smape']:
            vals   = [get_val(arima, field, mkey),
                      get_val(xgb,   field, mkey)]
            winner = int(np.argmin(vals))
            fmt    = f'{vals[winner]:.4f}' if mkey != 'smape' \
                     else f'{vals[winner]:.2f}%'
            row.append(f'{methods_lbl[winner]}\n({fmt})')
            rcolor.append(method_fill[winner])
        rows.append(row)
        row_colors.append(rcolor)

    table = ax.table(
        cellText=rows,
        rowLabels=[f'{v["label"]} ({v["unit"]})' for v in VARIABLES],
        colLabels=['MAE ↓', 'RMSE ↓', 'sMAPE ↓'],
        cellLoc='center', loc='center',
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(2.2, 3.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#AAAAAA')
        cell.set_text_props(color='#222222')
        if r == 0:
            cell.set_facecolor('#2166AC')
            cell.set_text_props(fontweight='bold', color='white')
        elif c == -1:
            cell.set_facecolor('#EEEEEE')
            cell.set_text_props(fontweight='bold', color='#333333')

    ax.axis('off')

    legend_els = [
        Line2D([0], [0], color=C_ARIMA, lw=10, label='ARIMA wins'),
        Line2D([0], [0], color=C_XGB,   lw=10, label='XGBoost wins'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               facecolor='white', edgecolor='#CCCCCC', fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        'Winner Table per Variable and Metric ARIMA vs XGBoost',
        fontsize=11, fontweight='bold', y=1.06)

    path = os.path.join(OUTPUT_DIR, 'comparison_winner_table.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Tabel pemenang: {path}")

def _compute_hybrid_choice(arima: dict, xgb: dict) -> dict:
    """Hitung winner dan rasio otomatis dari data aktual."""
    REASONS = {
        'joint_P_total'   : 'Motor power depends on velocity,\nacceleraton & wheel load simultaneously',
        'batt_soc_percent': 'SOC temporal trend captured by\nboth models',
        'odom_v_linear'   : 'Velocity follows motor commands\n& dynamic load conditions',
    }
    result = {}
    for field in ['joint_P_total', 'batt_soc_percent', 'odom_v_linear']:
        a_mae = get_val(arima, field, 'mae')
        x_mae = get_val(xgb,   field, 'mae')
        winner = 'ARIMA' if (a_mae < x_mae and a_mae > 0) else 'XGBoost'
        w_mae  = a_mae if winner == 'ARIMA' else x_mae
        l_mae  = x_mae if winner == 'ARIMA' else a_mae
        if w_mae > 0 and l_mae > 0:
            ratio = l_mae / w_mae
            ratio_str = (f'{winner} {ratio:.0f}x more accurate'
                         if ratio >= 2
                         else f'{winner} {((l_mae-w_mae)/l_mae*100):.0f}% more accurate')
        else:
            ratio_str = f'{winner} wins'
        result[field] = (winner, REASONS.get(field, ''), ratio_str)
    return result

def plot_hybrid_justification(arima, xgb):
    from matplotlib.lines import Line2D as _Line2D

    C_SMAPE = '#D7191C'   # garis sMAPE

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    hybrid_choice = _compute_hybrid_choice(arima, xgb)

    units = {
        'joint_P_total':    'Watt',
        'batt_soc_percent': '%',
        'odom_v_linear':    'm/s',
    }

    for idx, var in enumerate(VARIABLES):
        field = var['field']
        ax    = axes[idx]
        ax.set_facecolor('white')

        a_mae  = get_val(arima, field, 'mae')
        x_mae  = get_val(xgb,   field, 'mae')
        a_smap = get_val(arima, field, 'smape')
        x_smap = get_val(xgb,   field, 'smape')

        unit       = units.get(field, '')
        winner_mae = 0 if a_mae < x_mae else 1
        x_pos      = np.arange(2)

        bars = ax.bar(x_pos, [a_mae, x_mae], 0.5,
                      color=[C_ARIMA, C_XGB], alpha=0.80,
                      edgecolor='white', linewidth=0.5)
        bars[winner_mae].set_edgecolor('#333333')
        bars[winner_mae].set_linewidth(2.0)

        for b, v in zip(bars, [a_mae, x_mae]):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(a_mae, x_mae) * 0.05,
                    f'{v:.4f} {unit}',
                    ha='center', va='bottom',
                    fontsize=8.5, color='#333333', fontweight='bold')

        ax.set_ylabel(f'MAE ({unit}) (lower is better)',
                      fontsize=8.5, color='#444444')
        ax.set_ylim(0, max(a_mae, x_mae) * 1.40)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['ARIMA', 'XGBoost'],
                           fontsize=10, fontweight='bold')
        ax.tick_params(colors='#333333')
        for sp in ax.spines.values():
            sp.set_edgecolor('#AAAAAA')
        ax.grid(True, color='#CCCCCC', alpha=0.4, axis='y')

        ax2 = ax.twinx()
        ax2.set_facecolor('white')
        ax2.plot(x_pos, [a_smap, x_smap],
                 color=C_SMAPE, marker='D', markersize=9,
                 linewidth=1.5, linestyle='--', alpha=0.85)
        for xi, sv in zip(x_pos, [a_smap, x_smap]):
            ax2.text(xi, sv + max(a_smap, x_smap) * 0.07,
                     f'{sv:.1f}%',
                     ha='center', va='bottom',
                     fontsize=8.5, color=C_SMAPE, fontweight='bold')
        ax2.set_ylabel('sMAPE (%)  (lower is better)',
                       fontsize=8.5, color=C_SMAPE)
        ax2.set_ylim(0, max(a_smap, x_smap) * 1.55)
        ax2.tick_params(colors=C_SMAPE, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_edgecolor('#AAAAAA')

        pilihan, alasan, selisih = hybrid_choice.get(field, ('?', '', ''))
        ax.set_title(f'{var["label"]}\nSelected: {pilihan}\n({selisih})',
                     fontsize=9.5, fontweight='bold', pad=8)
        ax.text(0.5, -0.22, alasan,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8.5, color='#555555',
                style='italic', multialignment='center')

    legend_els = [
        _Line2D([0],[0], color=C_ARIMA, lw=10, label='ARIMA (bar)'),
        _Line2D([0],[0], color=C_XGB,   lw=10, label='XGBoost (bar)'),
        _Line2D([0],[0], color=C_SMAPE, lw=2,
                marker='D', markersize=7,
                linestyle='--', label='sMAPE % (diamond marker)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               facecolor='white', edgecolor='#CCCCCC', fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle(
        'Hybrid Model Selection Justification ARIMA + XGBoost\n'
        'Bar = MAE (left axis)  |  Diamond marker = sMAPE (right axis)'
        '  |  Dark border = winner',
        fontsize=10, fontweight='bold', y=1.03)

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, 'comparison_hybrid_justification.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Plot justifikasi hybrid (dual Y-axis): {path}")

def save_txt(arima, xgb):
    path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')

    winners = {}
    for var in VARIABLES:
        field = var['field']
        a_mae = get_val(arima, field, 'mae')
        x_mae = get_val(xgb,   field, 'mae')
        winners[field] = 'ARIMA' if (a_mae < x_mae and a_mae > 0) else 'XGBoost'

    with open(path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("KOMPARASI ARIMA vs XGBoost\n")
        f.write(f"Dijalankan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")

        for var in VARIABLES:
            field = var['field']
            f.write(f"{'─' * 65}\n  {var['label']} ({var['unit']})\n"
                    f"{'─' * 65}\n")
            f.write(f"  {'Metode':<12}{'MAE':>12}{'RMSE':>12}{'sMAPE':>10}\n")
            f.write(f"  {'-' * 46}\n")
            for name, d in [('ARIMA', arima), ('XGBoost', xgb)]:
                if field in d:
                    r  = d[field]
                    sm = r.get('smape', r.get('mape', 0))
                    f.write(
                        f"  {name:<12}{r.get('mae', 0):>12.4f}"
                        f"{r.get('rmse', 0):>12.4f}{sm:>9.2f}%\n")
            f.write("\n")

        f.write("KESIMPULAN\n")
        for var in VARIABLES:
            field   = var['field']
            a_mae   = get_val(arima, field, 'mae')
            x_mae   = get_val(xgb,   field, 'mae')
            winner  = 'ARIMA' if (a_mae > 0 and a_mae < x_mae) else 'XGBoost'
            f.write(f"  {var['label']:<25} → {winner}\n")
            f.write(f"    ARIMA MAE={a_mae:.4f} {var['unit']} | "
                    f"XGBoost MAE={x_mae:.4f} {var['unit']}\n")
        f.write("=" * 65 + "\n")
    print(f"Ringkasan TXT: {path}")

def plot_forecast_overlay(arima: dict, xgb: dict):
    import pandas as pd
    from influxdb_client import InfluxDBClient
    import matplotlib.dates as mdates

    INFLUXDB_URL    = "http://localhost:8086"
    INFLUXDB_TOKEN  = ("SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_"
                       "TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ==")
    INFLUXDB_ORG    = "polman"
    INFLUXDB_BUCKET = "polebot_hw"
    DATA_RANGE      = "start: 2026-07-10T09:29:00Z, stop: 2026-07-10T10:09:30Z"


    ARIMA_DIR = os.path.expanduser("~/arima_results_hw")
    XGB_DIR   = os.path.expanduser("~/xgboost_results_hw")

    COLOR_HIST    = '#888888'   # data historis
    COLOR_ARIMA_F = '#2166AC'   # forecast ARIMA
    COLOR_XGB_F   = '#1A9641'   # forecast XGBoost
    COLOR_CI      = '#AACCEE'   # CI ARIMA

    # Label, unit, dan rentang waktu historis yang ditampilkan (detik terakhir)
    var_config = [
        {'field': 'batt_soc_percent', 'label': 'Battery SOC',    'unit': '%',
         'history_s': 600},
        {'field': 'joint_P_total',    'label': 'Motor Power',     'unit': 'W',
         'history_s': 300},
        {'field': 'odom_v_linear',    'label': 'Linear Velocity', 'unit': 'm/s',
         'history_s': 300},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor('white')

    client = InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    for idx, vc in enumerate(var_config):
        field     = vc['field']
        label     = vc['label']
        unit      = vc['unit']
        hist_s    = vc['history_s']
        ax        = axes[idx]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#AAAAAA')
        ax.grid(True, color='#CCCCCC', alpha=0.4, linewidth=0.5)

        # 1. Ambil data historis dari InfluxDB
        query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range({DATA_RANGE})
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => r.source == "hardware_pzem")
  |> filter(fn: (r) => r._field == "{field}")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> sort(columns: ["_time"])
'''
        hist_drawn = False
        try:
            df = client.query_api().query_data_frame(query)
            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
            if not df.empty:
                df['_time'] = pd.to_datetime(df['_time'], utc=True)
                df = df.set_index('_time').sort_index()
                df.index = df.index.tz_convert('Asia/Jakarta').tz_localize(None)
                df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
                df = df.dropna(subset=['_value'])

                # Ambil N detik terakhir untuk tampilan
                hist = df['_value'].iloc[-hist_s:]
                ax.plot(hist.index, hist.values,
                        color=COLOR_HIST, linewidth=0.8, alpha=0.70,
                        label='Historical data', zorder=1)
                hist_drawn = True
        except Exception as e:
            print(f"Gagal ambil historis {field}: {e}")

        # 2. Baca forecast ARIMA
        arima_csv = os.path.join(ARIMA_DIR, f'forecast_{field}.csv')
        arima_drawn = False
        if os.path.exists(arima_csv):
            try:
                fa = pd.read_csv(arima_csv, parse_dates=['timestamp'])
                fa = fa.set_index('timestamp').sort_index()
                ax.plot(fa.index, fa['forecast_value'],
                        color=COLOR_ARIMA_F, linewidth=1.5,
                        linestyle='--', label='ARIMA forecast', zorder=3)
                # CI ARIMA jika tersedia
                if 'ci_lower' in fa.columns and 'ci_upper' in fa.columns:
                    ax.fill_between(fa.index,
                                    fa['ci_lower'], fa['ci_upper'],
                                    color=COLOR_CI, alpha=0.20,
                                    label='ARIMA 95% CI', zorder=2)
                # Garis vertikal pemisah historis/forecast
                ax.axvline(x=fa.index[0], color=COLOR_ARIMA_F,
                           linewidth=0.8, linestyle=':', alpha=0.6)
                arima_drawn = True
            except Exception as e:
                print(f"Gagal baca ARIMA forecast {field}: {e}")

        # 3. Baca forecast XGBoost
        xgb_csv = os.path.join(XGB_DIR, f'xgboost_hw_forecast_{field}.csv')
        xgb_drawn = False
        if os.path.exists(xgb_csv):
            try:
                fx = pd.read_csv(xgb_csv, parse_dates=['timestamp'])
                fx = fx.set_index('timestamp').sort_index()
                ax.plot(fx.index, fx['forecast_value'],
                        color=COLOR_XGB_F, linewidth=1.5,
                        linestyle='--', label='XGBoost forecast', zorder=3)
                ax.axvline(x=fx.index[0], color=COLOR_XGB_F,
                           linewidth=0.8, linestyle=':', alpha=0.6)
                xgb_drawn = True
            except Exception as e:
                print(f"Gagal baca XGBoost forecast {field}: {e}")

        # 4. Anotasi metrik
        a_mae   = get_val(arima, field, 'mae')
        x_mae   = get_val(xgb,   field, 'mae')
        a_smape = get_val(arima, field, 'smape')
        x_smape = get_val(xgb,   field, 'smape')

        anno = (
            f"ARIMA   | MAE: {a_mae:.4f} {unit}  |  sMAPE: {a_smape:.1f}%\n"
            f"XGBoost | MAE: {x_mae:.4f} {unit}  |  sMAPE: {x_smape:.1f}%"
        )
        ax.text(0.01, 0.97, anno,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8.5, color='#333333',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#F5F5F5',
                          edgecolor='#CCCCCC',
                          alpha=0.90))

        # 5. Format sumbu
        ax.set_ylabel(f'{label} ({unit})', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(labelsize=8.5)
        ax.legend(frameon=True, framealpha=0.9,
                  edgecolor='#CCCCCC', fontsize=8.5,
                  loc='lower left', ncol=2)

        # Tentukan pemenang untuk judul
        winner = 'ARIMA' if a_mae < x_mae else 'XGBoost'
        ax.set_title(
            f'{label} ARIMA vs XGBoost Forecast  '
            f'[Selected: {winner}]',
            fontsize=10.5, fontweight='bold', pad=6)

    client.close()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.suptitle(
        'ARIMA vs XGBoost Forecast Overlay Comparison\n',
        fontsize=12, fontweight='bold', y=1.00)

    path = os.path.join(OUTPUT_DIR, 'comparison_forecast_overlay.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Forecast overlay: {path}")

def main():
    print("  Comparison Plot ARIMA vs XGBoost")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nMemuat hasil...")
    arima, xgb = load_results()

    if not arima and not xgb:
        print("Tidak ada file hasil. Jalankan predictor terlebih dahulu.")
        return

    print("\n[1/5] Membuat plot bar komparasi...")
    plot_bar_comparison(arima, xgb)

    print("\n[2/5] Membuat tabel pemenang...")
    plot_winner_table(arima, xgb)

    print("\n[3/5] Membuat plot justifikasi hybrid...")
    plot_hybrid_justification(arima, xgb)

    print("\n[4/5] Membuat grafik forecast overlay...")
    plot_forecast_overlay(arima, xgb)

    print("\n[5/5] Menyimpan ringkasan teks...")
    save_txt(arima, xgb)

    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print("  comparison_bar_all.png")
    print("  comparison_winner_table.png")
    print("  comparison_hybrid_justification.png")
    print("  comparison_forecast_overlay.png")
    print("  comparison_summary.txt")


if __name__ == '__main__':
    main()