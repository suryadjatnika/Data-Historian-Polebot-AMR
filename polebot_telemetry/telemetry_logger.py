#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters

from sensor_msgs.msg import JointState, BatteryState
from nav_msgs.msg import Odometry

import math
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

import time as time_module

INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
INFLUXDB_ORG    = "polman"
INFLUXDB_BUCKET = "polebot_hw"

# KONSTANTA ROBOT diambil langsung dari URDF polebot.urdf.xacro
# Motor  : (PMSM, 48VDC, 1kW, τ_nom=3.2N·m, τ_max=6.4N·m)
# Driver : Tongyi IxLII 30.60 (20-80VDC, 30A continuous, 60A peak)
WHEEL_MASS      = 3.610          # kg  (massa satu roda)
WHEEL_RADIUS    = 0.078          # m   (jari-jari roda)
WHEEL_WIDTH     = 0.055          # m   (lebar roda)
BASE_MASS       = 84.837         # kg  (massa base_link)
MONITOR_MASS    = 16.008         # kg  (massa monitor_link)
ROBOT_MASS      = BASE_MASS + MONITOR_MASS  # ≈ 100.845 kg total

# Momen inersia roda - silinder pejal: I = ½ × m × r²
I_WHEEL = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2  # = 0.01125 kg·m²

# Koefisien gesek kinetik lantai epoxy halus
MU_KINETIC = 0.02

# Gravitasi
G = 9.81           # m/s²

# Motor  : 80SV-10030BA Rated Power 1000W, Rated Current 29A
# Driver : Max Continuous 30A, Input 48VDC
# Config : 1 driver untuk 2 motor = P_max = 48V × 30A
P_RATED         = 2000.0            # Watt (daya mekanik nominal)
                                    # 2 motor × 1000W (Tongyi 80SV-10030BA)
                                    # NB: daya listrik max = 2 × (48V × 30A) = 2880W

# ApproximateTimeSynchronizer
SYNC_QUEUE_SIZE         = 10
SYNC_SLOP               = 0.1    # toleransi selisih timestamp antar topic

# NODE UTAMA
class TelemetryLogger(Node):

    def __init__(self):
        super().__init__('telemetry_logger')
        self.get_logger().info("=" * 55)
        self.get_logger().info("  Polebot AMR — Telemetry Logger Node")
        self.get_logger().info("  Data Historian ROS2 · Polman Bandung 2025/2026")
        self.get_logger().info("=" * 55)

        # Inisialisasi InfluxDB
        self._init_influxdb()

        # State variables /odom
        self.v_prev    = 0.0      # kecepatan linear sebelumnya (m/s)
        self.t_prev    = None     # timestamp sebelumnya (dari header.stamp)
        self.x_prev    = 0.0     # posisi X sebelumnya (m)
        self.y_prev    = 0.0     # posisi Y sebelumnya (m)
        self.S_total   = 0.0     # jarak kumulatif total (m)

        # State variable /battery — menyimpan data terakhir yang diterima
        self.latest_battery = {
            'voltage'            : 51.2,
            'current'            : 0.5,
            'soc_percent'        : 100.0,
            'soc_status_code'    : 3,        # 3 = FULL
            'soc_status_text'    : 'FULL',
            'power_draw'         : 0.0,
            'energy_consumed_wh' : 0.0,
            'charge_remaining_ah': 32.0,
        }

        # Counter diagnostik
        self.callback_count = 0
        self.write_success  = 0
        self.write_fail     = 0

        # QoS Profile
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber via message_filters
        # Sinkronisasi hanya /odom + /joint_states (daya motor butuh akselerasi
        # dari odom, jadi keduanya harus sinkron dalam satu timestamp).
        self.odom_sub    = message_filters.Subscriber(
            self, Odometry,     '/odom',                   qos_profile=qos_sensor)
        self.joint_sub   = message_filters.Subscriber(
            self, JointState,   '/joint_states',            qos_profile=qos_sensor)
        self.battery_sub = self.create_subscription(BatteryState, '/polebot/battery_status', self._battery_callback, 10)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.joint_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP
        )

        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info("Subscriber aktif — menunggu data dari 3 topic...")
        self.get_logger().info(
            f"  /odom · /joint_states · /polebot/battery_status"
            f"  (slop={SYNC_SLOP}s)")


    # INISIALISASI INFLUXDB
    def _init_influxdb(self):
        """Buat koneksi ke InfluxDB dan verifikasi koneksi berhasil."""
        try:
            self.influx_client = InfluxDBClient(
                url=INFLUXDB_URL,
                token=INFLUXDB_TOKEN,
                org=INFLUXDB_ORG
            )
            self.write_api = self.influx_client.write_api(
                write_options=SYNCHRONOUS
            )
            # Ping untuk verifikasi koneksi
            health = self.influx_client.health()
            if health.status == "pass":
                self.get_logger().info(
                    f"InfluxDB terhubung → {INFLUXDB_URL} | bucket: {INFLUXDB_BUCKET}")
            else:
                self.get_logger().warn(f"InfluxDB status: {health.status}")
        except Exception as e:
            self.get_logger().error(f"GAGAL terhubung ke InfluxDB: {e}")
            self.get_logger().error(
                "Pastikan InfluxDB berjalan dan token sudah benar.")
            self.write_api = None

    def _battery_callback(self, msg: BatteryState):
        """Callback independen — simpan data baterai terbaru."""
        self.latest_battery = self._transform_battery(msg)

    # SYNCHRONIZED CALLBACK - dipanggil saat /odom + /joint_states siap
    def synchronized_callback(self, odom_msg, joint_msg):

        self.callback_count += 1

        # EXTRACT
        # Timestamp referensi sistem (nanosecond)
        t_ns = int(time_module.time() * 1_000_000_000)

        # TRANSFORM
        odom_data  = self._transform_odom(odom_msg)
        joint_data = self._transform_joint(joint_msg, odom_data['acceleration'], odom_data['v_linear'])
        battery_data = self.latest_battery   # ambil nilai terakhir yang tersimpan

        # LOAD
        self._write_to_influxdb(t_ns, odom_data, joint_data, battery_data)

        # Log ringkasan setiap 50 callback (~5 detik pada 10 Hz)
        if self.callback_count % 50 == 0:
            self.get_logger().info(
                f"[CB #{self.callback_count}] "
                f"v={odom_data['v_linear']:.2f}m/s | "
                f"P={joint_data['P_total']:.2f}W | "
                f"SOC={battery_data['soc_percent']:.1f}% | "
                f"V={battery_data['voltage']:.2f}V | "
                f"DB ok/fail={self.write_success}/{self.write_fail}"
            )


    # TRANSFORM 1: /odom — Differential Drive Kinematics
    def _transform_odom(self, msg: Odometry) -> dict:
        """
        Model: Differential Drive Kinematics
        Input : nav_msgs/Odometry
        Output: v_linear, omega, acceleration, S_total, delta_t

        Rumus:
          Δt      = t(k) - t(k-1)
          a(t)    = (v(t) - v(t-1)) / Δt
          Δd      = √( (Xt-Xt-1)² + (Yt-Yt-1)² )
          S_total = S_total + Δd 
        """

        # Kecepatan langsung dari pesan
        v_linear = msg.twist.twist.linear.x
        omega    = msg.twist.twist.angular.z

        # Posisi dari pose
        x_curr = msg.pose.pose.position.x
        y_curr = msg.pose.pose.position.y

        # Timestamp dari header (nanosecond → detik)
        t_curr = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Rumus 1 — delta waktu
        if self.t_prev is None:
            delta_t      = 0.0
            acceleration = 0.0
        else:
            delta_t = t_curr - self.t_prev
            if delta_t > 0.0:
                # Rumus 2 — akselerasi linear (derivasi numerik Euler)
                acceleration = (v_linear - self.v_prev) / delta_t
            else:
                acceleration = 0.0

        # Rumus 3 — Euclidean step distance
        delta_d = math.sqrt(
            (x_curr - self.x_prev)**2 + (y_curr - self.y_prev)**2
        )

        # Rumus 4 — jarak kumulatif total (virtual odometer)
        self.S_total += delta_d

        # Update state variables untuk iterasi berikutnya
        self.v_prev = v_linear
        self.t_prev = t_curr
        self.x_prev = x_curr
        self.y_prev = y_curr

        return {
            'v_linear'    : round(v_linear,    6),
            'omega'       : round(omega,        6),
            'acceleration': round(acceleration, 6),
            'S_total'     : round(self.S_total, 4),
            'delta_t'     : round(delta_t,      6),
        }


    # TRANSFORM 2: /joint_states — Power Estimation
    def _transform_joint(self, msg: JointState, acceleration: float, v_linear: float) -> dict:

        # Ekstrak velocity dari pesan
        # Nama joint: ['drivewhl_l_joint', 'drivewhl_r_joint']
        try:
            idx_l = msg.name.index('drivewhl_l_joint')
            idx_r = msg.name.index('drivewhl_r_joint')
            omega_L = msg.velocity[idx_l]
            omega_R = msg.velocity[idx_r]
        except (ValueError, IndexError):
            # Fallback kalau nama joint tidak ditemukan
            omega_L = msg.velocity[0] if len(msg.velocity) > 0 else 0.0
            omega_R = msg.velocity[1] if len(msg.velocity) > 1 else 0.0

        # Rumus 1 — akselerasi sudut roda dari akselerasi linear robot
        # α = a / r  (kinematika roda tidak slip)
        if WHEEL_RADIUS > 0:
            alpha = acceleration / WHEEL_RADIUS   # rad/s²
        else:
            alpha = 0.0

        # Rumus 2 — estimasi torsi per roda (τ = I × α)
        tau_L = I_WHEEL * alpha   # N·m
        tau_R = I_WHEEL * alpha   # N·m

        # Rumus 3 — daya inersia per roda (hanya saat akselerasi)
        P_inersia = (tau_L * abs(omega_L)) + (tau_R * abs(omega_R))

        # Rumus 4 — daya gesek (selalu ada selama robot bergerak)
        # P_gesek = μ × m × g × |v|
        # = 0.02 × 100.845 × 9.81 × |v| → ≈ 9.89W saat v=0.5m/s
        P_gesek = MU_KINETIC * ROBOT_MASS * G * abs(v_linear)

        # Rumus 5 — total daya mekanik
        P_total = P_inersia + P_gesek

        # Rumus 6 — rasio beban motor (normalisasi)
        load_ratio = P_total / P_RATED if P_RATED > 0 else 0.0
        load_ratio = max(0.0, min(1.0, load_ratio))   # clamp 0.0 – 1.0

        return {
            'omega_L'   : round(omega_L,    6),
            'omega_R'   : round(omega_R,    6),
            'tau_L'     : round(tau_L,      6),
            'tau_R'     : round(tau_R,      6),
            'P_total'   : round(P_total,    4),
            'load_ratio': round(load_ratio, 4),
        }

    # TRANSFORM 3: /polebot/battery_status — Energy Analysis
    def _transform_battery(self, msg: BatteryState) -> dict:
        voltage    = msg.voltage
        current    = msg.current
        soc        = msg.percentage * 100.0   # konversi 0.0–1.0 → 0–100%
        capacity   = msg.capacity             # Ah
        charge     = msg.charge               # Ah tersisa

        # Daya listrik sesaat
        P_draw = voltage * current            # Watt

        # Energi yang sudah dikonsumsi (Wh)
        # capacity_ah - charge_ah = consumed_ah
        consumed_ah = capacity - charge
        energy_consumed_wh = consumed_ah * voltage   # Wh

        # Status SOC berdasarkan threshold proposal
        # OPSI B: status disimpan sebagai KODE ANGKA (bukan teks) agar tidak
        # memecah query mean() di InfluxDB. Pemetaan kode:
        #   0 = CRITICAL, 1 = WARNING, 2 = NORMAL, 3 = FULL
        if voltage >= 50.5:
            soc_status_text = "FULL"
            soc_status_code = 3
        elif voltage < 42.0:
            soc_status_text = "CRITICAL"
            soc_status_code = 0
        elif voltage < 46.0:
            soc_status_text = "WARNING"
            soc_status_code = 1
        else:
            soc_status_text = "NORMAL"
            soc_status_code = 2

        return {
            'voltage'           : round(voltage,            3),
            'current'           : round(current,            3),
            'soc_percent'       : round(soc,                2),
            'soc_status_code'   : soc_status_code,   # angka → aman untuk InfluxDB
            'soc_status_text'   : soc_status_text,   # teks → hanya untuk log terminal
            'power_draw'        : round(P_draw,             3),
            'energy_consumed_wh': round(energy_consumed_wh, 4),
            'charge_remaining_ah': round(charge,            3),
        }


    # LOAD — Tulis ke InfluxDB
    def _write_to_influxdb(self, timestamp_ns: int,
                           odom: dict, joint: dict, battery: dict):
        """
        Tulis satu record ke InfluxDB yang berisi semua field
        dari ketiga sumber (odom, joint, battery) dalam satu timestamp.

        Measurement: polebot_telemetry
        Tags       : robot = polebot_amr
        Fields     : semua output dari ketiga transform
        """
        if self.write_api is None:
            return

        try:
            point = (
                Point("polebot_telemetry")
                .tag("robot", "polebot_amr")
                .tag("source", "hardware_pzem")

                #/odom — Differential Drive Kinematics
                .field("odom_v_linear",    odom['v_linear'])
                .field("odom_omega",       odom['omega'])
                .field("odom_accel",       odom['acceleration'])
                .field("odom_S_total",     odom['S_total'])
                .field("odom_delta_t",     odom['delta_t'])

                #/joint_states — Power Estimation
                .field("joint_omega_L",    joint['omega_L'])
                .field("joint_omega_R",    joint['omega_R'])
                .field("joint_tau_L",      joint['tau_L'])
                .field("joint_tau_R",      joint['tau_R'])
                .field("joint_P_total",    joint['P_total'])
                .field("joint_load_ratio", joint['load_ratio'])

                #/polebot/battery_status — Energy Analysis
                .field("batt_voltage",            battery['voltage'])
                .field("batt_current",            battery['current'])
                .field("batt_soc_percent",        battery['soc_percent'])
                .field("batt_soc_status_code",    battery['soc_status_code'])
                .field("batt_power_draw",         battery['power_draw'])
                .field("batt_energy_consumed_wh", battery['energy_consumed_wh'])
                .field("batt_charge_remaining_ah",battery['charge_remaining_ah'])

                .time(timestamp_ns, WritePrecision.NS)
            )

            self.write_api.write(
                bucket=INFLUXDB_BUCKET,
                org=INFLUXDB_ORG,
                record=point
            )
            self.write_success += 1

        except Exception as e:
            self.write_fail += 1
            if self.write_fail <= 5:      # batasi pesan error agar tidak spam
                self.get_logger().error(f"Gagal tulis ke InfluxDB: {e}")


    # CLEANUP
    def destroy_node(self):
        print(
            f"\n[telemetry_logger] Node dimatikan."
            f"\n  Total callback : {self.callback_count}"
            f"\n  DB sukses      : {self.write_success}"
            f"\n  DB gagal       : {self.write_fail}"
        )
        if hasattr(self, 'influx_client'):
            self.influx_client.close()
            print("[telemetry_logger] Koneksi InfluxDB ditutup.")
        super().destroy_node()


# ENTRY POINT
def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()