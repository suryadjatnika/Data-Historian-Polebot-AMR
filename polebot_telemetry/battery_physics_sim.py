#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import Twist
import math
import random

class BatteryPhysicsSim(Node):
    def __init__(self):
        super().__init__('battery_physics_sim')
        
        # Publisher ke Dashboard
        self.publisher_ = self.create_publisher(BatteryState, '/polebot/battery_status', 10)
        
        # Subscriber: Mendengarkan perintah Keyboard/Gazebo
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
            
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Mengambil parameter "scenario" dari terminal (Default: "NEW")
        self.declare_parameter('scenario', 'NEW')
        self.scenario = self.get_parameter('scenario').get_parameter_value().string_value
        
        # --- SPESIFIKASI ASLI POLEBOT AMR (48V 32Ah) ---
        if self.scenario == "NEW":
            self.capacity_ah = 32.0       # 4× GS Astra NS40 kondisi baru
            self.internal_r  = 0.05       # hambatan dalam rendah
            self.peukert_k   = 1.1        # efisiensi baik
        else:
            self.capacity_ah = 16.0       # kondisi degradasi
            self.internal_r  = 0.20       # hambatan dalam tinggi
            self.peukert_k   = 1.3        # efisiensi memburuk

        # Tongyi IxLII 30.60 — 48VDC, 2 driver × 30A = 60A continuous, 2 × 60A = 120A peak
        self.I_max_continuous = 60.0      # Ampere (2 driver × 30A)
        self.I_peak           = 120.0      # Ampere (2 driver × 60A peak)
        self.V_nominal        = 48.0      # Volt

        self.voltage        = 51.2        # tegangan penuh 4× 12.8V
        self.current        = 0.5         # arus standby
        self.target_current = 0.5
        self.soc            = 100.0
        self.consumed_ah    = 0.0

        self.get_logger().info(f'Polebot Digital Twin Started. Mode: {self.scenario}')

    def get_ocv_curve(self, soc):
        # Kurva Voltase Aki Kering Seri 48V (Kosong: 42V, Penuh: 51.2V)
        soc_norm = soc / 100.0
        v_ocv = 42.0 + (9.2 * soc_norm)
        
        # Drop tegangan drastis jika baterai mau habis (di bawah 20%)
        if soc < 20: 
            v_ocv -= 2.0 * (1 - (soc/20.0))
        return v_ocv

    def cmd_vel_callback(self, msg):
        # Kecepatan linear aktual dari spesifikasi
        speed = abs(msg.linear.x)
        turn = abs(msg.angular.z)
        
        # Max speed Polebot = 0.8 m/s. 
        # Arus maksimum dengan beban 560kg beroperasi 45 menit = ~42.6 Ampere
        speed_factor = min(speed / 0.8, 1.0) 
        turn_factor = min(turn / 1.0, 1.0) 
        
        # Arus total = Standby + Beban Maju + Beban Belok
        self.target_current = 0.5 + (speed_factor * 15.0) + (turn_factor * 5.0)
        self.target_current = min(self.target_current, self.I_peak)

    def timer_callback(self):
        # Filter lonjakan arus agar grafik mulus
        self.current = (0.9 * self.current) + (0.1 * self.target_current)
        
        # Simulasi noise pembacaan sensor tegangan/arus di lapangan
        noise = random.uniform(-0.1, 0.1) 
        real_current = self.current + noise
        if real_current < 0: real_current = 0
        
        # Hitung Fisika (Hukum Peukert untuk penurunan kapasitas)
        effective_current = math.pow(real_current, self.peukert_k)
        ah_step = effective_current / 36000.0 
        self.consumed_ah += ah_step
        
        # Hitung State of Charge (SOC)
        self.soc = ((self.capacity_ah - self.consumed_ah) / self.capacity_ah) * 100.0
        if self.soc < 0: self.soc = 0
        
        # Hitung Voltage Drop (Tegangan Turun akibat tarikan motor)
        v_open = self.get_ocv_curve(self.soc)
        v_drop = real_current * self.internal_r
        self.voltage = v_open - v_drop + random.uniform(-0.05, 0.05)

        # Siapkan paket data untuk InfluxDB/Grafana
        msg = BatteryState()
        msg.header.stamp        = self.get_clock().now().to_msg()
        msg.voltage             = float(round(self.voltage, 3))
        msg.current             = float(round(real_current, 3))
        msg.percentage          = float(round(self.soc / 100.0, 4))  # 0.0–1.0
        msg.capacity            = float(self.capacity_ah)
        msg.charge              = float(round(
            (self.capacity_ah - self.consumed_ah), 3))
        msg.present             = True
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BatteryPhysicsSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()