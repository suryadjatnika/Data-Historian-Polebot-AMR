#!/usr/bin/env python3
import math
import json
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
try:
    from visualization_msgs.msg import MarkerArray, Marker
    from std_msgs.msg import ColorRGBA
    HAVE_MARKERS = True
except ImportError:
    HAVE_MARKERS = False


# ─── Parameter Fisik Polebot AMR (sama dengan energy_path_predictor.py) ────
POLEBOT_PARAMS = {
    "battery_voltage_V"   : 48.0,
    "battery_capacity_Ah" : 32.0,
    "battery_capacity_Wh" : 1536.0,   # 48V × 32Ah
    "motor_rated_W"       : 2000.0,   # 2 × 1000W Tongyi 80SV-10030BA
    "wheel_radius_m"      : 0.078,
    "wheelbase_m"         : 0.587,
    "mass_kg"             : 100.845,
    "v_max_ms"            : 0.7,      # sesuai skenario 2 kita
    "v_default_ms"        : 0.3,      # sesuai skenario 1
    "v_min_ms"            : 0.1,
    "accel_default"       : 0.1,
    "d_min_obstacle"      : 0.5,      # threshold WARN kita
}

# ─── Koefisien Model XGBoost (dari hasil komparasi) ─────────────────────────
XGBOOST_COEF = {
    "coef_v"    : 8.234,
    "coef_accel": 15.12,
    "coef_load" : 4.67,
    "coef_d_inv": 2.31,
    "intercept" : 1.42,
}

# ─── Model ARIMA(0,1,0) — drift rate dari data historis ─────────────────────
ARIMA_SOC_DRAIN_RATE = 0.00278  # %/s


class PathPlanningEnergyNode(Node):
    """
    ROS2 Node: menerima path dari Nav2, memprediksi konsumsi energi,
    dan mempublikasikan rekomendasi kecepatan optimal.
    """

    def __init__(self):
        super().__init__("path_planning_energy_node")

        self.get_logger().info("="*55)
        self.get_logger().info("  Polebot AMR — Path Planning Energy Node")
        self.get_logger().info("  Hybrid ARIMA + XGBoost Integration")
        self.get_logger().info("="*55)

        # ── State ──
        self.soc_current    = 100.0   # % (update dari /batt_soc)
        self.v_current      = 0.0     # m/s (update dari /odom)
        self.last_plan_hash = None    # untuk deteksi plan baru
        self.last_prediction = {}

        # ── Publishers ──
        self.pub_prediction = self.create_publisher(
            String, "/energy_prediction", 10)
        self.pub_cmd_optimal = self.create_publisher(
            Twist, "/cmd_vel_optimal", 10)

        if HAVE_MARKERS:
            self.pub_markers = self.create_publisher(
                MarkerArray, "/energy_marker", 10)

        # ── Subscribers ──
        self.sub_plan = self.create_subscription(
            Path, "/plan", self.plan_callback, 10)
        self.sub_soc = self.create_subscription(
            Float32, "/batt_soc", self.soc_callback, 10)
        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        # ── Timer: publish status setiap 5 detik ──
        self.timer_status = self.create_timer(5.0, self.status_callback)

        self.get_logger().info("  Node siap. Menunggu /plan dari Nav2...")
        self.get_logger().info("  Subscribe: /plan, /batt_soc, /odom")
        self.get_logger().info("  Publish  : /energy_prediction, /cmd_vel_optimal")

    # ─── Callback: terima SOC dari telemetri ────────────────────────────────
    def soc_callback(self, msg: Float32):
        self.soc_current = float(msg.data)

    # ─── Callback: terima kecepatan dari odometri ────────────────────────────
    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_current = math.sqrt(vx**2 + vy**2)

    # ─── Callback UTAMA: terima path dari Nav2 ───────────────────────────────
    def plan_callback(self, msg: Path):
        """
        Dipanggil setiap kali Nav2 menghasilkan path baru.
        Path berisi array PoseStamped dari posisi robot sekarang ke goal.
        """
        poses = msg.poses
        n = len(poses)

        if n < 2:
            self.get_logger().warn("Path terlalu pendek (< 2 poses), skip.")
            return

        # Deteksi path baru (hash dari titik awal & akhir)
        p_start = poses[0].pose.position
        p_end   = poses[-1].pose.position
        plan_hash = f"{p_start.x:.2f},{p_start.y:.2f},{p_end.x:.2f},{p_end.y:.2f}"

        if plan_hash == self.last_plan_hash:
            return  # path sama, skip
        self.last_plan_hash = plan_hash

        self.get_logger().info(f"Path baru diterima: {n} poses dari "
                               f"({p_start.x:.2f},{p_start.y:.2f}) → "
                               f"({p_end.x:.2f},{p_end.y:.2f})")

        # ── Hitung segmen ──
        # Nav2 memberikan banyak poses (densely sampled)
        # Kita downsample menjadi waypoints yang bermakna
        waypoints = self._downsample_path(poses, max_waypoints=10)
        segments  = self._analyze_path_segments(waypoints)

        if not segments:
            return

        # ── Ringkasan prediksi ──
        total_distance  = sum(s["distance_m"] for s in segments)
        total_time_s    = sum(s["duration_s"] for s in segments)
        total_energy_Wh = sum(s["E_Wh"]      for s in segments)
        total_delta_soc = sum(s["delta_soc"]  for s in segments)
        soc_akhir       = max(0.0, self.soc_current - total_delta_soc)

        # ── Bangun pesan JSON ──
        prediction = {
            "timestamp"       : self.get_clock().now().to_msg().sec,
            "soc_current_pct" : self.soc_current,
            "soc_akhir_pct"   : round(soc_akhir, 3),
            "soc_terpakai_pct": round(total_delta_soc, 4),
            "total_distance_m": round(total_distance, 3),
            "total_time_s"    : round(total_time_s, 1),
            "total_energy_Wh" : round(total_energy_Wh, 4),
            "n_poses"         : n,
            "n_segments"      : len(segments),
            "model"           : "Hybrid ARIMA(0,1,0) + XGBoost",
            "world"           : "depot",
            "status"          : "OK" if soc_akhir > 20 else "WARNING_LOW_SOC",
            "segments"        : segments,
        }
        self.last_prediction = prediction

        # Publish JSON
        msg_out = String()
        msg_out.data = json.dumps(prediction, indent=None)
        self.pub_prediction.publish(msg_out)

        # Publish kecepatan optimal untuk segmen pertama
        self._publish_optimal_velocity(segments)

        # Publish markers RViz
        if HAVE_MARKERS:
            self._publish_energy_markers(poses, segments)

        # Log ringkasan
        self.get_logger().info(
            f"Prediksi: {total_distance:.1f}m, {total_time_s:.1f}s, "
            f"{total_energy_Wh:.4f}Wh, SOC {self.soc_current:.1f}%→{soc_akhir:.2f}%"
        )

        if soc_akhir < 20:
            self.get_logger().warn(
                f"⚠️  SOC akhir {soc_akhir:.1f}% < 20% — baterai hampir habis!"
            )

    def _downsample_path(self, poses, max_waypoints=10):
        """
        Downsample path Nav2 (bisa 100+ poses) menjadi waypoints bermakna.
        Pilih poses yang memiliki perubahan heading signifikan atau jarak minimal.
        """
        if len(poses) <= max_waypoints:
            return poses

        # Ambil poses dengan interval merata + selalu masukkan awal & akhir
        step = max(1, len(poses) // (max_waypoints - 1))
        selected = [poses[i] for i in range(0, len(poses), step)]
        if poses[-1] not in selected:
            selected.append(poses[-1])
        return selected[:max_waypoints]

    def _analyze_path_segments(self, poses):
        """
        Analisis setiap segmen dari path.
        Return list of dict dengan prediksi energi per segmen.
        """
        segments = []
        p = POLEBOT_PARAMS
        soc_running = self.soc_current

        for i in range(len(poses) - 1):
            pos1 = poses[i].pose.position
            pos2 = poses[i+1].pose.position

            # Jarak
            dx = pos2.x - pos1.x
            dy = pos2.y - pos1.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < 0.01:  # skip segmen terlalu pendek
                continue

            # Heading change
            yaw1 = self._quat_to_yaw(poses[i].pose.orientation)
            yaw2 = self._quat_to_yaw(poses[i+1].pose.orientation)
            dh = abs(yaw2 - yaw1)
            if dh > math.pi:
                dh = 2*math.pi - dh

            # Kecepatan adaptif per segmen
            v = self._adaptive_velocity(dist, dh)

            # Prediksi daya (aproksimasi XGBoost)
            P = self._predict_power(v, p["accel_default"], dh, dist)

            # Waktu tempuh
            t_accel = v / p["accel_default"]
            d_accel = 0.5 * p["accel_default"] * t_accel**2
            d_cruise = max(0, dist - 2*d_accel)
            duration = 2*t_accel + (d_cruise/v if v > 0 else 0)

            # Energi dan SOC (ARIMA)
            E_Wh = P * duration / 3600.0
            delta_soc = (E_Wh / p["battery_capacity_Wh"]) * 70 + \
                        ARIMA_SOC_DRAIN_RATE * duration * 30
            delta_soc /= 100  # normalisasi bobot

            soc_running -= delta_soc

            segments.append({
                "seg_idx"   : i + 1,
                "x1"        : round(pos1.x, 3),
                "y1"        : round(pos1.y, 3),
                "x2"        : round(pos2.x, 3),
                "y2"        : round(pos2.y, 3),
                "distance_m": round(dist, 3),
                "heading_delta_deg": round(math.degrees(dh), 1),
                "v_optimal_ms"    : round(v, 3),
                "duration_s"      : round(duration, 2),
                "P_total_W"       : round(P, 3),
                "E_Wh"            : round(E_Wh, 5),
                "delta_soc"       : round(delta_soc, 5),
                "soc_after"       : round(max(0, soc_running), 3),
            })

        return segments

    def _quat_to_yaw(self, q):
        """Konversi quaternion → yaw."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _adaptive_velocity(self, distance, delta_heading):
        """Kecepatan adaptif berdasarkan karakteristik segmen."""
        p = POLEBOT_PARAMS
        v = p["v_default_ms"]

        # Faktor jarak
        if distance < 0.5:
            v *= 0.4
        elif distance < 2.0:
            v *= 0.75

        # Faktor heading
        if delta_heading > math.pi / 4:
            v *= 0.5
        elif delta_heading > math.pi / 8:
            v *= 0.75

        return max(p["v_min_ms"], min(p["v_max_ms"], v))

    def _predict_power(self, v, accel, delta_heading, distance):
        """Prediksi daya total (aproksimasi XGBoost)."""
        c = XGBOOST_COEF
        p = POLEBOT_PARAMS

        if distance > 0 and v > 0:
            omega = delta_heading / (distance / v)
        else:
            omega = 0.0

        load_ratio = (v / p["v_max_ms"]) * 0.8 + 0.1
        d_min = p["d_min_obstacle"]

        P = (c["intercept"] +
             c["coef_v"] * v +
             c["coef_accel"] * accel +
             c["coef_load"] * load_ratio +
             c["coef_d_inv"] * (omega / (d_min + 0.5)))

        return max(0.1, P)

    def _publish_optimal_velocity(self, segments):
        """
        Publish kecepatan optimal untuk segmen pertama (yang akan ditempuh sekarang).
        """
        if not segments:
            return
        s = segments[0]
        v_opt = s["v_optimal_ms"]

        msg = Twist()
        msg.linear.x  = v_opt
        msg.angular.z = 0.0  # dikontrol Nav2
        self.pub_cmd_optimal.publish(msg)

    def _publish_energy_markers(self, all_poses, segments):
        """
        Publish MarkerArray untuk visualisasi di RViz.
        Warna segmen berdasarkan konsumsi energi (hijau=hemat, merah=boros).
        """
        if not HAVE_MARKERS:
            return

        marker_array = MarkerArray()
        max_E = max(s["E_Wh"] for s in segments) if segments else 1.0

        for i, seg in enumerate(segments):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "energy_segments"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05  # lebar garis

            # Warna: hijau (hemat) → merah (boros)
            ratio = seg["E_Wh"] / max_E if max_E > 0 else 0
            m.color.r = ratio
            m.color.g = 1.0 - ratio
            m.color.b = 0.0
            m.color.a = 0.8

            from geometry_msgs.msg import Point
            p1 = Point()
            p1.x, p1.y, p1.z = seg["x1"], seg["y1"], 0.05
            p2 = Point()
            p2.x, p2.y, p2.z = seg["x2"], seg["y2"], 0.05
            m.points = [p1, p2]

            marker_array.markers.append(m)

        self.pub_markers.publish(marker_array)

    def status_callback(self):
        """Log status node setiap 5 detik."""
        if self.last_prediction:
            p = self.last_prediction
            self.get_logger().info(
                f"Status: SOC={self.soc_current:.1f}%, "
                f"v={self.v_current:.3f}m/s, "
                f"Prediksi aktif: {p.get('n_segments',0)} segmen, "
                f"SOC akhir prediksi: {p.get('soc_akhir_pct','-')}%"
            )
        else:
            self.get_logger().info(
                f"Menunggu path... SOC={self.soc_current:.1f}%, v={self.v_current:.3f}m/s"
            )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningEnergyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node dihentikan.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()