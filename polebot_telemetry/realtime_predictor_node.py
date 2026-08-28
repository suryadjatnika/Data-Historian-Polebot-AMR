#!/usr/bin/env python3
"""
Node ROS 2 CBTS (Condition-Based Temporal Switching) REAL-TIME.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')
from collections import deque

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

import rclpy
from rclpy.node import Node


# ── Konfigurasi ──
INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_hw"

MODEL_DIR = os.path.expanduser("~/polebot_hybrid_results_hw/polebot_realtime_models")

TARGETS = ['batt_soc_percent', 'joint_P_total', 'odom_v_linear']
CROSS_FIELDS = ['odom_v_linear', 'odom_accel', 'joint_P_total', 'batt_current']

TICK_HZ = 1.0            # laju prediksi: 1 kali per detik
BUFFER_MAXLEN = 300      # sama dengan ARIMA_BUFFER di train_realtime_f0.py


class RealtimePredictorNode(Node):

    def __init__(self):
        super().__init__('realtime_predictor_node')

        # ── Koneksi InfluxDB (baca + tulis) ──
        self.client = InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

        # ── Muat metadata & model XGBoost ──
        with open(os.path.join(MODEL_DIR, 'model_metadata.json')) as f:
            self.meta = json.load(f)

        self.accel_thr = self.meta['accel_threshold']
        self.speed_thr = self.meta['speed_threshold']
        self.arima_order = self.meta['arima_order']
        self.xgb_feature_order = self.meta['xgb_features']

        self.xgb_models = {}
        for t in TARGETS:
            m = xgb.XGBRegressor()
            m.load_model(os.path.join(MODEL_DIR, f'xgb_{t}.json'))
            self.xgb_models[t] = m
        self.get_logger().info(f"Model XGBoost dimuat untuk: {TARGETS}")

        # ── Buffer bergulir riwayat (untuk fitur lag & ARIMA) ──
        self.buffer = deque(maxlen=BUFFER_MAXLEN)

        # ── Timer prediksi 1 Hz ──
        self.timer = self.create_timer(1.0 / TICK_HZ, self._tick)

        self.tick_count = 0
        self.get_logger().info("="*58)
        self.get_logger().info("  Realtime Predictor Node (CBTS)")
        self.get_logger().info("="*58)
        self.get_logger().info(f"  Ambang: |v|<{self.speed_thr} & |a|<{self.accel_thr} -> STATIS")
        self.get_logger().info(f"  Laju prediksi: {TICK_HZ} Hz")
        self.get_logger().info("  Menunggu data pertama dari InfluxDB...")

    def _fetch_latest_1s(self):
        """Ambil rata-rata 18 field dalam 1 detik terakhir dari InfluxDB."""
        flux = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: -2s)
  |> filter(fn: (r) => r._measurement == "polebot_telemetry")
  |> filter(fn: (r) => r.source == "hardware_pzem")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
  |> last()
'''
        try:
            df = self.query_api.query_data_frame(flux)
            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
            if df.empty:
                return None
            row = df.iloc[-1].to_dict()
            return row
        except Exception as e:
            self.get_logger().warn(f"Gagal ambil data InfluxDB: {e}")
            return None

    def _classify(self, v, a):
        """STATIS jika |v|<ambang DAN |a|<ambang, else DINAMIS."""
        return abs(v) < self.speed_thr and abs(a) < self.accel_thr

    def _build_xgb_feature_vector(self, target):
        """
        Susun satu baris fitur dari buffer
        """
        if len(self.buffer) < 6:
            return None  # buffer belum cukup untuk lag3 + rollmean5

        hist = pd.DataFrame(list(self.buffer))
        row = {}
        row[f'{target}_lag1'] = hist[target].iloc[-1]
        row[f'{target}_lag2'] = hist[target].iloc[-2]
        row[f'{target}_lag3'] = hist[target].iloc[-3]
        row[f'{target}_rollmean5'] = hist[target].iloc[-5:].mean()
        for c in CROSS_FIELDS:
            if c == target:
                continue
            row[f'{c}_lag1'] = hist[c].iloc[-1]

        urutan = self.xgb_feature_order[target]
        vektor = np.array([[row[f] for f in urutan]])
        return vektor

    def _predict_arima(self, target):
        """Latih ulang ARIMA pada buffer STATIS terkini, ramal 1 langkah."""
        hist = pd.DataFrame(list(self.buffer))
        statis_only = hist[hist['is_static']][target].dropna().values
        if len(statis_only) < 30:
            return None
        order = tuple(self.arima_order[target])
        try:
            model = ARIMA(statis_only, order=order).fit()
            forecast = model.forecast(steps=1)
            return float(forecast[0])
        except Exception as e:
            self.get_logger().warn(f"ARIMA {target} gagal: {e}")
            return None

    def _tick(self):
        row = self._fetch_latest_1s()
        if row is None:
            return

        v = row.get('odom_v_linear', 0.0) or 0.0
        a = row.get('odom_accel', 0.0) or 0.0
        is_static = self._classify(v, a)
        row['is_static'] = is_static
        self.buffer.append(row)

        self.tick_count += 1
        hasil = {}
        model_aktif = 'ARIMA' if is_static else 'XGBoost'

        for target in TARGETS:
            if is_static:
                pred = self._predict_arima(target)
            else:
                fv = self._build_xgb_feature_vector(target)
                pred = float(self.xgb_models[target].predict(fv)[0]) if fv is not None else None
            hasil[target] = pred

        # Tulis ke InfluxDB (measurement baru, terpisah dari data mentah)
        try:
            point = (
                Point("polebot_realtime_prediction")
                .tag("robot", "polebot_amr")
                .tag("active_model", model_aktif)
                .field("kondisi_dinamis", 0.0 if is_static else 1.0)
            )
            for target, pred in hasil.items():
                if pred is not None:
                    point = point.field(f"pred_{target}", pred)
            self.write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            self.get_logger().warn(f"Gagal tulis prediksi: {e}")

        if self.tick_count % 5 == 0:
            soc_txt = f"{hasil['batt_soc_percent']:.2f}%" if hasil['batt_soc_percent'] else "N/A"
            p_txt = f"{hasil['joint_P_total']:.3f}W" if hasil['joint_P_total'] else "N/A"
            v_txt = f"{hasil['odom_v_linear']:.3f}m/s" if hasil['odom_v_linear'] else "N/A"
            self.get_logger().info(
                f"[{self.tick_count}] {model_aktif:8s} | v={v:.3f} a={a:.3f} | "
                f"SOC~{soc_txt} P~{p_txt} v~{v_txt}")

    def destroy_node(self):
        self.client.close()
        self.get_logger().info(
            f"[realtime_predictor_node] Node dimatikan. Total tick: {self.tick_count}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimePredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()