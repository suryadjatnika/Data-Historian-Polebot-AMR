#!/usr/bin/env python3
"""
scenario_runner_cl.py — Scenario Runner CLOSED-LOOP (HARDWARE)
==============================================================
Versi closed-loop dari scenario_runner2.py. Perbedaan utama:
  - Fase gerak berhenti berdasarkan JARAK NYATA (dari /odom),
    BUKAN durasi waktu. Ini mencegah robot menjauh dari titik awal
    dan menabrak dinding.
  - Robot maju sampai jarak target, lalu MUNDUR kembali ke titik awal
    (via odom), sehingga drift antar-siklus minimal.

Cara kerja pelacakan posisi:
  signed_pos += sign(kecepatan_perintah) x jarak_langkah_euclidean
  → maju menambah posisi, mundur mengurangi. Robust terhadap arah hadap.

KEAMANAN (WAJIB robot nyata):
  - Batas jarak keras: kalau |posisi| > max_forward + margin → STOP darurat
  - Timeout keselamatan per fase
  - Pengawas dengan Emergency Stop WAJIB

Cara pakai:
    python3 scenario_runner_cl.py --ros-args -p scenario:=1
    python3 scenario_runner_cl.py --ros-args \
        -p scenario:=2 -p duration:=300 -p max_forward:=2.5
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class ScenarioRunnerCL(Node):

    def __init__(self):
        super().__init__('scenario_runner_cl')

        # ── Parameter ──
        self.declare_parameter('scenario', 1)
        self.declare_parameter('duration', 300)        # detik total per skenario
        self.declare_parameter('max_forward', 2.5)     # jarak maju maks (m, via odom)
        self.declare_parameter('safety_margin', 1.0)   # margin batas keras (m)
        # Laju perubahan kecepatan (m/s per detik) untuk ramping halus.
        # accel_statis dibuat SANGAT landai agar |a| tetap < 0.10 (ambang statis),
        # sehingga segmen statis tidak "bocor" terklasifikasi dinamis.
        self.declare_parameter('accel_statis',  0.06)   # ramp landai untuk statis
        self.declare_parameter('accel_dinamis', 0.50)   # ramp tegas untuk dinamis

        self.scenario     = self.get_parameter('scenario').value
        self.duration     = self.get_parameter('duration').value
        self.max_forward  = self.get_parameter('max_forward').value
        self.safety_margin= self.get_parameter('safety_margin').value
        self.accel_statis = self.get_parameter('accel_statis').value
        self.accel_dinamis= self.get_parameter('accel_dinamis').value
        self.hard_limit   = self.max_forward + self.safety_margin

        # Kecepatan robot yang sedang diperintahkan saat ini (untuk ramping).
        # Dinaikkan/diturunkan bertahap tiap tick menuju target, bukan lompat.
        self.cmd_speed_now = 0.0

        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                 self._odom_cb, 10)

        # ── State pelacakan posisi ──
        self.last_x     = None
        self.last_y     = None
        self.signed_pos = 0.0      # posisi bertanda dari titik awal (m)
        self.euclid_from_origin = 0.0
        self.origin_x   = None
        self.origin_y   = None
        self.current_cmd_sign = 0

        # ── State eksekusi segmen ──
        self.segments    = self._build_segments()
        self.seg_index   = 0
        self.seg_start_time = None
        self.dwell_until = None
        self.finished    = False
        self.odom_ready  = False

        self.start_time  = time.time()
        self.timer = self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  Scenario Runner CLOSED-LOOP — Skenario {self.scenario}")
        self.get_logger().info("=" * 60)
        self._log_scenario_info()
        self.get_logger().info(
            f"  Jarak maju maks : {self.max_forward:.1f} m (via odom)")
        self.get_logger().info(
            f"  Batas keras     : {self.hard_limit:.1f} m (STOP darurat)")
        self.get_logger().info(f"  Durasi          : {self.duration//60} menit")
        self.get_logger().warn(
            "  PASTIKAN pengawas siap dengan Emergency Stop!")
        self.get_logger().info("  Menunggu /odom pertama...")

    # ──────────────────────────────────────────────────────────
    # ODOM: hitung posisi bertanda dari titik awal
    # ──────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.origin_x is None:
            self.origin_x = x
            self.origin_y = y
            self.last_x = x
            self.last_y = y
            self.odom_ready = True
            self.get_logger().info(
                f"  Odom awal: x={x:.3f}, y={y:.3f}. Mulai!")
            self.seg_start_time = time.time()
            return

        self.last_x = x
        self.last_y = y

        # Jarak euclidean absolut dari titik awal (untuk batas keselamatan).
        # Dihitung LANGSUNG dari posisi absolut, tidak diakumulasi, sehingga
        # tidak menumpuk error meski sesi berlangsung lama.
        self.euclid_from_origin = math.sqrt(
            (x - self.origin_x)**2 + (y - self.origin_y)**2)

        # PERBAIKAN BUG SESI PANJANG:
        # signed_pos TIDAK LAGI diakumulasi dari step positif (yang dulu
        # menumpuk error tiap lengkungan kecil, sampai melonjak liar seperti
        # -22.9m pada sesi 20 menit). Sekarang signed_pos = euclid_from_origin
        # yang diberi TANDA sesuai arah gerak robot relatif titik awal.
        # Karena dihitung dari posisi absolut tiap saat, error tidak menumpuk.
        # Tanda ditentukan dari arah perpindahan dominan (proyeksi sederhana):
        # jika robot berada "di depan" titik awal pada sumbu geraknya -> positif.
        if self.euclid_from_origin < 0.03:
            self.signed_pos = 0.0
        else:
            # Tanda mengikuti arah perintah terakhir saat menjauh/mendekat.
            # Karena gerak robot maju-mundur pada satu garis, |euclid| sudah
            # merepresentasikan jarak; tanda dipakai hanya untuk logika mundur.
            self.signed_pos = self.euclid_from_origin

    # ──────────────────────────────────────────────────────────
    # BUILDER SEGMEN — 7 SKENARIO
    #   segmen 'move' : gerak sampai signed_pos capai target
    #   segmen 'dwell': diam selama N detik
    # ──────────────────────────────────────────────────────────
    def _build_segments(self):
        """
        Segmen euclid-based:
          'out'  : maju sampai euclid_from_origin >= target_euclid
          'back' : mundur sampai euclid dekat 0 (atau timeout keselamatan)
          'dwell': diam N detik
        Jarak dibuat konservatif karena robot cenderung melengkung.
        """
        D = self.max_forward   # jarak maju maks (euclid)

        if self.scenario == 1:
            return [
                {'type':'out','target_euclid': D*0.8,'speed':0.3,'label':'Maju 0.3 m/s'},
                {'type':'dwell','duration':1.0,'label':'Berhenti'},
                {'type':'back','speed':0.3,'max_time':12.0,'label':'Mundur ke titik awal'},
                {'type':'dwell','duration':1.0,'label':'Berhenti'},
            ]
        elif self.scenario == 2:
            return [
                {'type':'out','target_euclid': D,'speed':0.42,'label':'Sprint maju'},
                {'type':'dwell','duration':0.5,'label':'Rem'},
                {'type':'back','speed':0.42,'max_time':12.0,'label':'Sprint mundur'},
                {'type':'dwell','duration':0.5,'label':'Rem'},
            ]
        elif self.scenario == 3:
            return [
                {'type':'out','target_euclid': D*0.4,'speed':0.4,'label':'Maju segmen 1'},
                {'type':'dwell','duration':1.5,'label':'BERHENTI'},
                {'type':'out','target_euclid': D*0.75,'speed':0.4,'label':'Maju segmen 2'},
                {'type':'dwell','duration':1.5,'label':'BERHENTI'},
                {'type':'out','target_euclid': D,'speed':0.4,'label':'Maju segmen 3'},
                {'type':'dwell','duration':1.5,'label':'BERHENTI'},
                {'type':'back','speed':0.4,'max_time':15.0,'label':'Mundur ke awal'},
                {'type':'dwell','duration':1.5,'label':'BERHENTI'},
            ]
        elif self.scenario == 4:
            return [
                {'type':'out','target_euclid': D*0.5,'speed':0.1,'label':'Creep maju 0.1'},
                {'type':'dwell','duration':1.5,'label':'Berhenti'},
                {'type':'back','speed':0.1,'max_time':15.0,'label':'Creep mundur'},
                {'type':'dwell','duration':1.5,'label':'Berhenti'},
            ]
        elif self.scenario == 5:
            return [
                {'type':'out','target_euclid': D*0.6,'speed':0.42,'label':'BURST maju'},
                {'type':'dwell','duration':1.5,'label':'REM mendadak'},
                {'type':'back','speed':0.42,'max_time':12.0,'label':'BURST mundur'},
                {'type':'dwell','duration':1.5,'label':'REM mendadak'},
            ]
        elif self.scenario == 6:
            # Mixed — TIAP kecepatan pola maju-mundur pendek (tidak akumulatif)
            return [
                {'type':'dwell','duration':3.0,'label':'[STATIS] IDLE'},
                {'type':'out','target_euclid': D*0.5,'speed':0.15,'label':'[DINAMIS] Lambat 0.15'},
                {'type':'back','speed':0.15,'max_time':12.0,'label':'Mundur'},
                {'type':'dwell','duration':1.5,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D*0.5,'speed':0.30,'label':'[DINAMIS] Sedang 0.30'},
                {'type':'back','speed':0.30,'max_time':12.0,'label':'Mundur'},
                {'type':'dwell','duration':1.5,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D*0.5,'speed':0.42,'label':'[DINAMIS] Cepat 0.42'},
                {'type':'back','speed':0.42,'max_time':12.0,'label':'Mundur'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] IDLE'},
            ]
        elif self.scenario == 7:
            # Mixed Switching — STATIS <-> DINAMIS bergantian.
            # POLA: sprint/burst dinamis dilakukan saat MAJU (0.42, terkendali),
            # dan kembali ke titik awal dengan MUNDUR PELAN (0.15) agar robot
            # tidak melengkung parah seperti saat mundur cepat.
            # Target maju dibuat lebih pendek (D*0.5) untuk toleransi lengkungan.
            return [
                # Blok creep (statis) — maju & mundur sama-sama pelan
                {'type':'out','target_euclid': D*0.5,'speed':0.12,'label':'[STATIS] Creep maju'},
                {'type':'back','speed':0.12,'max_time':14.0,'label':'[STATIS] Creep mundur pelan'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Stop'},
                # Blok sprint (dinamis) — MAJU cepat, MUNDUR pelan
                {'type':'out','target_euclid': D*0.5,'speed':0.42,'label':'[DINAMIS] Sprint MAJU cepat'},
                {'type':'back','speed':0.15,'max_time':16.0,'label':'[STATIS] Mundur pelan (kembali)'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Stop'},
                # Blok normal (statis)
                {'type':'out','target_euclid': D*0.5,'speed':0.25,'label':'[STATIS] Normal maju'},
                {'type':'back','speed':0.15,'max_time':16.0,'label':'[STATIS] Mundur pelan'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Stop'},
                # Blok burst (dinamis) — MAJU cepat, MUNDUR pelan
                {'type':'out','target_euclid': D*0.5,'speed':0.42,'label':'[DINAMIS] Burst MAJU cepat'},
                {'type':'back','speed':0.15,'max_time':16.0,'label':'[STATIS] Mundur pelan (kembali)'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] IDLE'},
            ]
        elif self.scenario == 8:
            # DEMO SIDANG — ringkas, 2 siklus STATIS->DINAMIS->STATIS yang jelas.
            # Dirancang untuk ruang demo (lorong ~4.5-5m di antara 2 meja),
            # jarak dipangkas (max_forward default 2.0m) untuk margin ekstra
            # hari-H. Total durasi gerak ~60-90 detik, sisa waktu untuk narasi.
            return [
                {'type':'dwell','duration':3.0,'label':'[STATIS] Robot diam - amati panel CBTS (biru)'},
                {'type':'out','target_euclid': D*0.4,'speed':0.15,'label':'[STATIS] Creep pelan'},
                {'type':'back','speed':0.15,'max_time':10.0,'label':'[STATIS] Kembali pelan'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D,'speed':0.42,'label':'[DINAMIS] SPRINT - amati panel berubah oranye'},
                {'type':'back','speed':0.15,'max_time':16.0,'label':'[STATIS] Kembali pelan (aman)'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D,'speed':0.42,'label':'[DINAMIS] SPRINT ke-2 - konfirmasi switching'},
                {'type':'back','speed':0.15,'max_time':16.0,'label':'[STATIS] Kembali pelan (aman)'},
                {'type':'dwell','duration':2.0,'label':'[STATIS] Selesai - robot diam'},
            ]
        elif self.scenario == 9:
            # ============================================================
            # SKENARIO REVISI (Penguji 1 & 2) — SIKLUS TUNGGAL BERULANG
            # ============================================================
            # Tujuan revisi:
            #  1. Satu bentuk siklus yang diulang terus sampai 'duration' habis
            #     (5/10/20/30 menit diatur via parameter duration, bukan ganti
            #     skenario). Karena _build_segments berputar (modulo), siklus ini
            #     otomatis berulang hingga durasi total tercapai.
            #  2. Gerakan HALUS — pakai 'accel' per segmen untuk ramping bertahap,
            #     bukan sentakan. Ini menjawab kritik "gerakan kurang mulus".
            #  3. Transisi STATIS<->DINAMIS JELAS DAN SINKRON DENGAN MATA:
            #     - STATIS: accel landai (0.06) + kecepatan rendah (0.12) yang
            #       ditahan. |a| tetap < 0.10 sepanjang segmen sehingga TIDAK
            #       bocor jadi dinamis. Robot terlihat benar-benar kalem.
            #     - DINAMIS: accel tegas (0.50) + kecepatan tinggi (0.42) yang
            #       jelas ngebut. |v| dan |a| sama-sama lewat ambang.
            #     Ini menjawab kritik "statis kok tanpa akselerasi berubah dinamis".
            #  4. Porsi dinamis sengaja diperbanyak dari skenario lama agar
            #     transisi lebih sering terlihat (permintaan: tidak harus ikut
            #     proporsi jurnal, murni untuk KTI).
            #
            # Satu siklus = STATIS pelan -> DINAMIS ngebut -> STATIS pelan -> jeda.
            # Pola maju/mundur mempertahankan prinsip aman: kembali selalu pelan.
            return [
                {'type':'dwell','duration':4.0,'label':'[STATIS] Diam (baseline biru)'},
                {'type':'out','target_euclid': D*0.35,'speed':0.12,
                 'accel':0.06,'label':'[STATIS] Maju pelan-halus (a landai)'},
                {'type':'back','speed':0.12,'accel':0.06,'max_time':14.0,
                 'label':'[STATIS] Mundur pelan-halus'},
                {'type':'dwell','duration':3.0,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D,'speed':0.42,
                 'accel':0.50,'label':'[DINAMIS] NGEBUT maju (a tegas, oranye)'},
                {'type':'back','speed':0.15,'accel':0.10,'max_time':16.0,
                 'label':'[STATIS] Mundur pelan (aman)'},
                {'type':'dwell','duration':3.0,'label':'[STATIS] Jeda'},
                {'type':'out','target_euclid': D,'speed':0.42,
                 'accel':0.50,'label':'[DINAMIS] NGEBUT maju ke-2 (oranye)'},
                {'type':'back','speed':0.15,'accel':0.10,'max_time':16.0,
                 'label':'[STATIS] Mundur pelan (aman)'},
                {'type':'dwell','duration':4.0,'label':'[STATIS] Diam (tutup siklus)'},
            ]
        else:
            self.get_logger().error(f"Skenario {self.scenario} tidak dikenal!")
            return [{'type':'dwell','duration':1.0,'label':'IDLE (error)'}]

    # ──────────────────────────────────────────────────────────
    # CONTROL LOOP
    # ──────────────────────────────────────────────────────────
    def _control_loop(self):
        if self.finished:
            return
        if not self.odom_ready:
            self._publish_zero()
            return

        elapsed_total = time.time() - self.start_time

        # --- Durasi total habis ---
        if elapsed_total >= self.duration:
            self._finish("Durasi tercapai")
            return

        # --- BATAS KESELAMATAN KERAS ---
        # Pakai |signed_pos| agar konsisten dengan target gerak (bukan euclid
        # terpisah yang bisa melonjak akibat noise heading odom).
        if self.euclid_from_origin > self.hard_limit:
            self.get_logger().error(
                f"BATAS KERAS TERLEWATI! euclid={self.euclid_from_origin:.2f}m "
                f"> {self.hard_limit:.2f}m. STOP DARURAT (anti-tabrak).")
            self._finish("Batas keselamatan terlewati")
            return

        seg = self.segments[self.seg_index]

        # ===== SEGMEN DWELL =====
        if seg['type'] == 'dwell':
            self.current_cmd_sign = 0
            self.cmd_speed_now = 0.0   # reset ramping saat berhenti
            self._publish_zero()
            if self.dwell_until is None:
                self.dwell_until = time.time() + seg['duration']
                self.get_logger().info(
                    f"  [{int(elapsed_total)}s] {seg['label']} "
                    f"(euclid={self.euclid_from_origin:.2f}m)")
            if time.time() >= self.dwell_until:
                self.dwell_until = None
                self._next_segment()
            return

        # ===== SEGMEN MOVE (euclid-based) =====
        # Tipe segmen: 'out' (maju sampai euclid capai target) atau
        #              'back' (mundur sampai euclid dekat 0)
        seg_type = seg['type']
        speed    = seg['speed']

        if seg_type == 'out':
            target_euclid = seg['target_euclid']
            direction = 1
            # Sampai kalau jarak nyata dari awal >= target
            reached = self.euclid_from_origin >= target_euclid
        else:  # 'back'
            direction = -1
            # Sampai kalau sudah dekat titik awal
            reached = self.euclid_from_origin <= 0.20
            # Cap keselamatan: kalau mundur terlalu lama (robot melengkung
            # dan tidak bisa kembali), berhenti daripada berputar terus.
            if self.seg_start_time and (time.time() - self.seg_start_time) > seg.get('max_time', 15.0):
                self.get_logger().warn(
                    f"  Mundur melebihi batas waktu (euclid={self.euclid_from_origin:.2f}m). "
                    f"Lanjut ke segmen berikutnya.")
                reached = True

        if reached:
            self.current_cmd_sign = 0
            self.cmd_speed_now = 0.0   # reset ramping saat segmen selesai
            self._publish_zero()
            self._next_segment()
            return

        # Publish perintah gerak DENGAN RAMPING BERTAHAP.
        # Kecepatan tidak dilempar langsung ke target, tapi dinaikkan sedikit
        # demi sedikit tiap tick (20 Hz). Laju kenaikan (accel) ditentukan per
        # segmen: landai untuk statis (agar |a|<0.10), tegas untuk dinamis.
        self.current_cmd_sign = direction
        target_speed = abs(speed)
        accel = seg.get('accel', self.accel_dinamis)  # m/s per detik
        dt = 0.05  # periode timer (20 Hz)
        step = accel * dt
        # Ramp menuju target_speed
        if self.cmd_speed_now < target_speed:
            self.cmd_speed_now = min(target_speed, self.cmd_speed_now + step)
        elif self.cmd_speed_now > target_speed:
            self.cmd_speed_now = max(target_speed, self.cmd_speed_now - step)

        twist = Twist()
        twist.linear.x = direction * self.cmd_speed_now
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        # Diagnostik odom tiap ~3 detik
        now = time.time()
        if not hasattr(self, '_last_diag') or (now - self._last_diag) > 3.0:
            self._last_diag = now
            tgt = seg.get('target_euclid', 0.0)
            self.get_logger().info(
                f"    [odom] euclid={self.euclid_from_origin:.2f}m | "
                f"arah={'MAJU' if direction>0 else 'MUNDUR'} | "
                f"target_euclid={tgt:.2f}m")

        # Log transisi segmen (sekali per segmen)
        if self.seg_start_time is not None and \
           (time.time() - self.seg_start_time) < 0.1:
            tgt_txt = (f"target {seg['target_euclid']:.2f}m"
                       if seg_type == 'out' else "kembali ke awal")
            self.get_logger().info(
                f"  [{int(elapsed_total)}s] {seg['label']} "
                f"→ {tgt_txt} @ {speed:.2f}m/s")

    def _next_segment(self):
        self.seg_index = (self.seg_index + 1) % len(self.segments)
        self.seg_start_time = time.time()

    def _publish_zero(self):
        t = Twist()
        self.cmd_pub.publish(t)

    def _finish(self, alasan):
        self.current_cmd_sign = 0
        for _ in range(10):
            self._publish_zero()
        self.finished = True
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"SELESAI ({alasan}) — Skenario {self.scenario}")
        self.get_logger().info(f"  Posisi akhir: {self.signed_pos:.2f}m dari awal")
        self.get_logger().info("=" * 60)
        self.create_timer(1.0, self._shutdown_once)

    def _shutdown_once(self):
        raise SystemExit(0)

    def _log_scenario_info(self):
        info = {
            1:("Baseline","STATIS","ARIMA"),2:("Beban Tinggi","DINAMIS","XGBoost"),
            3:("Stop-and-Go","DINAMIS","XGBoost"),4:("Creep","STATIS","ARIMA"),
            5:("Burst","DINAMIS EKSTREM","XGBoost"),6:("Mixed","CAMPURAN","keduanya"),
            7:("Mixed Switching","CAMPURAN SWITCHING","CBTS validation"),
            8:("DEMO SIDANG","CAMPURAN SWITCHING RINGKAS","CBTS demo 10 menit"),
            9:("REVISI - Siklus Halus Berulang","STATIS kalem <-> DINAMIS ngebut",
               "durasi via -p duration (5/10/20/30 menit)"),
        }
        nama,kondisi,model = info.get(self.scenario,("?","?","?"))
        self.get_logger().info(f"  Nama    : {nama}")
        self.get_logger().info(f"  Kondisi : {kondisi} → {model}")

    def destroy_node(self):
        try:
            for _ in range(10):
                self._publish_zero()
        except Exception:
            pass
        self.get_logger().info(
            f"[scenario_runner_cl] Skenario {self.scenario} dihentikan.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioRunnerCL()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()