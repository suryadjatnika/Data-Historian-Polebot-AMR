import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxBridgeNode(Node):
    def __init__(self):
        super().__init__('influx_bridge_node')

        # --- KONFIGURASI INFLUXDB (SESUAIKAN DISINI!) ---
        self.token = "SYcl0AdCw24pzzbtK5DV70HSko6zDalLqPCEHKLRjNB1t_TuVDkGe7w-Bdirll5eGUXVyNFbCdiE3Ku6Wh07aQ=="  # <--- GANTI INI
        self.org = "polman"
        self.bucket = "polebot_data"
        self.url = "http://localhost:8086"

        # Koneksi ke Database
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Subscribe ke Topik Baterai
        self.subscription = self.create_subscription(
            BatteryState,
            '/polebot/battery_status',
            self.listener_callback,
            10)
        
        self.get_logger().info("InfluxDB Bridge Started! Menunggu data...")

    def listener_callback(self, msg):
        # Setiap ada data baru dari simulasi, fungsi ini jalan
        
        # 1. Siapkan Titik Data (Point)
        point = Point("energy_system") \
            .tag("robot_id", "polebot_01") \
            .field("voltage", msg.voltage) \
            .field("current", msg.current) \
            .field("soc", msg.percentage)

        try:
            # 2. Tulis ke Database
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            # self.get_logger().info(f"💾 Saved: {msg.voltage:.2f} V") # Un-comment kalau mau lihat log
        except Exception as e:
            self.get_logger().error(f"Gagal menulis ke DB: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = InfluxBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()