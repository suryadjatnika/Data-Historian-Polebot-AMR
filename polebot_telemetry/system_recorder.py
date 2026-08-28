#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import json
import time
from rclpy.qos import qos_profile_sensor_data
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class SystemRecorder(Node):
    def __init__(self):
        super().__init__('system_recorder')
        
        # KONFIGURASI INFLUXDB
        self.token = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="
        self.org = "polman"
        self.bucket = "polebot_data"
        self.url = "http://localhost:8086"
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # SUBSCRIBER
        # 1. Sedot Data Baterai (Dari battery_physics)
        self.create_subscription(String, '/polebot/battery_status', self.battery_callback, 10)
        
        # 2. Sedot Data Posisi/Gerak (Dari Gazebo)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

        # PENYIMPANAN SEMENTARA
        # Data di-update terus menerus oleh callback
        self.current_voltage = 0.0
        self.current_ampere = 0.0
        self.current_soc = 0.0
        self.speed_linear = 0.0
        
        # Timer Perekaman (Simpan ke DB setiap 0.5 detik)
        self.timer = self.create_timer(0.5, self.record_to_db)
        
        self.get_logger().info('System Recorder ACTIVE. Listening to Odom & Battery')

    def battery_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_voltage = data['voltage']
            self.current_ampere = data['current']
            self.current_soc = data['soc']
        except:
            pass

    def odom_callback(self, msg):
        # Ambil kecepatan linear robot (m/s)
        self.speed_linear = msg.twist.twist.linear.x

    def record_to_db(self):
        point = Point("telemetry_full") \
            .tag("robot_id", "polebot_v1") \
            .field("voltage", float(self.current_voltage)) \
            .field("current", float(self.current_ampere)) \
            .field("soc", float(self.current_soc)) \
            .field("speed_mps", float(self.speed_linear))
            
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            # Notes
            print(f"Saved: {self.current_voltage}V | Speed: {self.speed_linear:.2f} m/s", end="\r")
        except Exception as e:
            self.get_logger().error(f"Gagal simpan ke DB: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SystemRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()