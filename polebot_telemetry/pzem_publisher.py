#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState

import minimalmodbus
import serial
import time


class PzemPublisher(Node):

    def __init__(self):
        super().__init__('pzem_publisher')

        # PARAMETER
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('slave_address', 1)
        self.declare_parameter('baudrate', 9600)
        self.declare_parameter('capacity_ah', 32.0)
        self.declare_parameter('publish_rate_hz', 10.0)

        # Ambang arus untuk membedakan kondisi statis dan dinamis (Ampere).
        # Di bawah nilai robot dianggap idle SOC pakai OCV.
        # Di atas nilai robot menarik daya SOC pakai Coulomb counting.
        self.declare_parameter('static_current_threshold', 2.0)

        # Eksponen Hukum Peukert (koreksi penurunan kapasitas efektif pada arus tinggi. k=1.1 untuk baterai lead-acid kondisi baik
        self.declare_parameter('peukert_k', 1.1)

        self.port        = self.get_parameter('port').value
        self.slave_addr  = self.get_parameter('slave_address').value
        self.baudrate    = self.get_parameter('baudrate').value
        self.capacity_ah = self.get_parameter('capacity_ah').value
        rate             = self.get_parameter('publish_rate_hz').value
        self.i_static_th = self.get_parameter('static_current_threshold').value
        self.peukert_k   = self.get_parameter('peukert_k').value

        # KONSTANTA KURVA OCV (lead-acid 48V)
        # Kosong (0%)  = 20.0 V
        # Penuh (100%) = 51.2 V
        # v_ocv = empty + span × soc_norm
        self.OCV_EMPTY = 20.0    # Volt saat SOC 0%
        self.OCV_SPAN  = 31.2    # rentang tegangan 0 -> 100%

        # STATE ESTIMASI SOC
        self.consumed_ah = 0.0
        self.soc         = None
        self.last_time   = None
        self.soc_method  = 'INIT'

        # KONEKSI PZEM-017
        self.get_logger().info("PZEM-017 Publisher Node (HARDWARE)")
        self.get_logger().info("=" * 58)

        self.pzem = self._connect_pzem()

        # KALIBRASI SOC AWAL (dari OCV)
        self._calibrate_initial_soc()

        # PUBLISHER + TIMER
        self.publisher_ = self.create_publisher(
            BatteryState, '/polebot/battery_status', 10)

        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

        # Counter diagnostik
        self.read_ok   = 0
        self.read_fail = 0

        self.get_logger().info(
            f"Publisher aktif /polebot/battery_status @ {rate:.0f} Hz")
        self.get_logger().info(
            f"  Port={self.port} | Kapasitas={self.capacity_ah}Ah | "
            f"Ambang statis={self.i_static_th}A | Peukert k={self.peukert_k}")

    # KONEKSI MODBUS
    def _connect_pzem(self):
        try:
            inst = minimalmodbus.Instrument(self.port, self.slave_addr)
            inst.serial.baudrate = self.baudrate
            inst.serial.bytesize = 8
            inst.serial.parity   = serial.PARITY_NONE
            inst.serial.stopbits = 2
            inst.serial.timeout  = 1.0
            inst.mode = minimalmodbus.MODE_RTU
            inst.clear_buffers_before_each_transaction = True
            self.get_logger().info(
                f"PZEM-017 terhubung di {self.port} "
                f"(slave={self.slave_addr}, baud={self.baudrate})")
            return inst
        except Exception as e:
            self.get_logger().error(f"GAGAL koneksi PZEM-017: {e}")
            self.get_logger().error(
                "Cek: (1) port benar? ls /dev/ttyUSB*  "
                "(2) PZEM dapat 5V?  (3) V+/V- tersambung baterai?")
            raise

    # PEMBACAAN PZEM (register map sesuai datasheet resmi)
    def _read_pzem(self):
        """
        Peta register resmi PZEM-017:
          0x0000 Voltage   16-bit  ×0.01 V
          0x0001 Current   16-bit  ×0.01 A
          0x0002 Power Lo  32-bit  ×0.1  W
          0x0003 Power Hi
          0x0004 Energy Lo 32-bit  ×1    Wh
          0x0005 Energy Hi
          0x0006 HV Alarm  0xFFFF=alarm
          0x0007 LV Alarm  0xFFFF=alarm
        """
        try:
            r = self.pzem.read_registers(0x0000, 8, functioncode=4)
            return {
                'voltage'  : r[0] * 0.01,
                'current'  : r[1] * 0.01,
                'power'    : (r[2] + (r[3] << 16)) * 0.1,
                'energy'   : r[4] + (r[5] << 16),
                'hv_alarm' : r[6] == 0xFFFF,
                'lv_alarm' : r[7] == 0xFFFF,
            }
        except Exception as e:
            self.read_fail += 1
            if self.read_fail <= 5:
                self.get_logger().warn(f"Gagal baca PZEM: {e}")
            return None

    # KURVA OCV konversi tegangan
    def _ocv_to_soc(self, voltage):
        soc_norm = (voltage - self.OCV_EMPTY) / self.OCV_SPAN
        soc = soc_norm * 100.0
        return max(0.0, min(100.0, soc))

    def _soc_to_charge_ah(self, soc):
        """Konversi SOC % → sisa muatan Ah (untuk field msg.charge)."""
        return (soc / 100.0) * self.capacity_ah

    # KALIBRASI SOC AWAL
    def _calibrate_initial_soc(self):
        self.get_logger().info("Kalibrasi SOC awal dari OCV (robot diam)...")
        samples = []
        for _ in range(10):
            data = self._read_pzem()
            if data:
                samples.append(data['voltage'])
            time.sleep(0.1)

        if not samples:
            self.soc = 100.0
            self.get_logger().warn(
                "Kalibrasi gagal (tidak ada data). SOC awal di-set 100%.")
        else:
            v_avg = sum(samples) / len(samples)
            self.soc = self._ocv_to_soc(v_avg)
            self.get_logger().info(
                f"  V_avg={v_avg:.2f}V → SOC awal = {self.soc:.1f}%")

        # Sinkronkan consumed_ah dengan SOC awal
        self.consumed_ah = self.capacity_ah * (1.0 - self.soc / 100.0)
        self.last_time = time.time()

    # ESTIMASI SOC (OCV saat statis, Coulomb saat dinamis)
    def _estimate_soc(self, voltage, current):
        """
        Pilih metode SOC berdasarkan kondisi operasi:
          |I| < ambang = STATIS maka OCV (sekaligus re-kalibrasi consumed_ah)
          |I| ≥ ambang = DINAMIS maka Coulomb counting
        """
        now = time.time()
        dt = now - self.last_time if self.last_time else 0.0
        self.last_time = now

        i_abs = abs(current)

        if i_abs < self.i_static_th:
            # KONDISI STATIS
            self.soc = self._ocv_to_soc(voltage)
            # Re-sinkronkan akumulator Coulomb agar konsisten saat nanti bergerak
            self.consumed_ah = self.capacity_ah * (1.0 - self.soc / 100.0)
            self.soc_method = 'OCV'
        else:
            # KONDISI DINAMIS = COULOMB COUNTING + PEUKERT
            # Coulomb counting dasar : Ah = I x dt / 3600
            # Koreksi Hukum Peukert  : arus efektif = I^k
            i_effective = i_abs ** self.peukert_k
            self.consumed_ah += i_effective * dt / 3600.0
            self.soc = (
                (self.capacity_ah - self.consumed_ah) / self.capacity_ah
            ) * 100.0
            self.soc = max(0.0, min(100.0, self.soc))
            self.soc_method = 'COULOMB+PEUKERT'

        return self.soc

    # TIMER CALLBACK baca, estimasi, publish
    def timer_callback(self):
        data = self._read_pzem()
        if data is None:
            return

        self.read_ok += 1

        voltage = data['voltage']
        current = data['current']

        # Estimasi SOC
        soc = self._estimate_soc(voltage, current)
        charge_ah = self._soc_to_charge_ah(soc)

        # Susun pesan BatteryState
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage      = float(round(voltage, 3))
        msg.current      = float(round(current, 3))
        msg.percentage   = float(round(soc / 100.0, 4))   # 0.0–1.0
        msg.capacity     = float(self.capacity_ah)
        msg.charge       = float(round(charge_ah, 3))
        msg.present      = True
        self.publisher_.publish(msg)

        # Log ringkasan tiap 50 siklus (~5 detik @ 10 Hz)
        if self.read_ok % 50 == 0:
            self.get_logger().info(
                f"[#{self.read_ok}] V={voltage:.2f}V I={current:.2f}A "
                f"P={data['power']:.1f}W | SOC={soc:.1f}% "
                f"[{self.soc_method}] | ok/fail={self.read_ok}/{self.read_fail}"
            )

    def destroy_node(self):
        print(
            f"\n[pzem_publisher] Node dimatikan."
            f"\n  Pembacaan sukses : {self.read_ok}"
            f"\n  Pembacaan gagal  : {self.read_fail}"
            f"\n  SOC terakhir     : {self.soc:.1f}%"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PzemPublisher()
    except Exception:
        rclpy.shutdown()
        return
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