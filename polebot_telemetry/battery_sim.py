import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
import random

class BatterySimNode(Node):
    def __init__(self):
        super().__init__('battery_sim_node')
        
        # Publisher ke topik /polebot/battery_status
        self.publisher_ = self.create_publisher(BatteryState, '/polebot/battery_status', 10)
        
        # Timer berjalan setiap 1 detik
        self.timer = self.create_timer(1.0, self.publish_battery_data)
        
        # Setting Awal: Baterai Penuh (50.5 V)
        self.current_voltage = 50.5
        self.get_logger().info("🔋 Simulasi Baterai Dimulai pada 50.5V")

    def publish_battery_data(self):
        msg = BatteryState()
        
        # --- LOGIKA SIMULASI ---
        # Tegangan turun acak (0.01 - 0.05 V) per detik
        drop = random.uniform(0.01, 0.05)
        self.current_voltage -= drop
        
        # Jika habis (< 41V), reset lagi ke penuh (Looping)
        if self.current_voltage < 41.0:
            self.current_voltage = 50.5
            self.get_logger().info("🔄 Charge Ulang (Reset Simulasi)")

        # Isi pesan ROS
        msg.voltage = self.current_voltage
        msg.current = 2.0 + random.uniform(-0.1, 0.1) # Simulasi Arus 2 Ampere
        msg.percentage = (self.current_voltage - 42.0) / (50.5 - 42.0) * 100.0
        
        # Kirim data
        self.publisher_.publish(msg)
        self.get_logger().info(f'Mengirim: {msg.voltage:.2f} V | {msg.percentage:.1f} %')

def main(args=None):
    rclpy.init(args=args)
    node = BatterySimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()