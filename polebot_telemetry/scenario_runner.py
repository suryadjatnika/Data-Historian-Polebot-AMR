#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

# ─────────────────────────────────────────────────────────────
# PARAMETER GLOBAL
# ─────────────────────────────────────────────────────────────
DURATION_SECONDS = 1800   # 30 menit per skenario
TURN_SPEED       = 0.5    # rad/s — kecepatan angular untuk belok
PUBLISH_RATE     = 0.1    # detik — 10 Hz

# Durasi putar pada TURN_SPEED = 0.5 rad/s:
#   90°  → π/2 / 0.5 = 3.14s ≈ 3.2s
#   180° → π   / 0.5 = 6.28s ≈ 6.4s
T90  = 3.2    # detik untuk belok 90°
T180 = 6.4    # detik untuk putar 180°


# ─────────────────────────────────────────────────────────────
# NODE UTAMA
# ─────────────────────────────────────────────────────────────
class ScenarioRunner(Node):

    def __init__(self):
        super().__init__('scenario_runner')

        self.declare_parameter('scenario', 1)
        self.scenario = self.get_parameter('scenario') \
                            .get_parameter_value().integer_value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.start_time      = time.time()
        self.phase_start     = time.time()
        self.phase_index     = 0
        self.running         = True
        self.last_log_second = -1

        self.phases = self._build_phases()

        self.timer = self.create_timer(PUBLISH_RATE, self._timer_callback)

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  Scenario Runner — Skenario {self.scenario}")
        self.get_logger().info(
            f"  Durasi: {DURATION_SECONDS // 60} menit "
            f"({DURATION_SECONDS} detik)")
        self.get_logger().info("=" * 60)
        self._log_scenario_info()
        self.get_logger().info("  Memulai dalam 3 detik...")
        time.sleep(3.0)
        self.start_time  = time.time()
        self.phase_start = time.time()
        self.get_logger().info("▶ Skenario dimulai!")


    # ──────────────────────────────────────────────────────────
    # BUILDER FASE — 6 SKENARIO
    # ──────────────────────────────────────────────────────────
    def _build_phases(self) -> list:
        """
        Bangun daftar fase gerakan per skenario.
        Setiap fase: dict {linear (m/s), angular (rad/s),
                           duration (detik), label (string)}.
        Fase diulang otomatis sampai DURATION_SECONDS tercapai.

        Prinsip keamanan (depot world):
        - Segmen lurus max 1.4 m (cukup di koridor ~2 m)
        - Pola selalu kembali ke posisi + orientasi awal dalam 1 siklus
        - Robot dimensi: 1.172 × 0.670 m
        """

        if self.scenario == 1:
            # ──────────────────────────────────────────────
            # Skenario 1 — Normal / Baseline
            # Kondisi: STATIS (kecepatan konstan rendah, akselerasi minimal)
            # Tujuan  : Data SOC baseline → ARIMA dapat memodelkan tren linier
            # Pola    : Kotak kecil, v = 0.3 m/s
            # Jarak   : 0.3 × 3.0 = 0.9 m/sisi → aman di koridor
            # Siklus  : ~25.6 detik (4 sisi + 4 belokan)
            # ──────────────────────────────────────────────
            return [
                {'linear': 0.3, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Maju lurus 0.3 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90°'},
                {'linear': 0.3, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Maju lurus 0.3 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90°'},
                {'linear': 0.3, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Maju lurus 0.3 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90°'},
                {'linear': 0.3, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Maju lurus 0.3 m/s — kembali ke titik awal'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90° — orientasi kembali ke awal'},
            ]

        elif self.scenario == 2:
            # ──────────────────────────────────────────────
            # Skenario 2 — Beban Tinggi
            # Kondisi: DINAMIS (kecepatan tinggi)
            # Tujuan  : Data SOC saat drain tinggi → XGBoost (kondisi dinamis)
            # Pola    : Bolak-balik (ping-pong), v = 0.7 m/s
            # Jarak   : 0.7 × 2.0 = 1.4 m/segmen → batas aman
            # Siklus  : ~17.8 detik
            # ──────────────────────────────────────────────
            return [
                {'linear': 0.7, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Sprint maju 0.7 m/s'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Berhenti sejenak'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180° kiri'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Berhenti sejenak'},
                {'linear': 0.7, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Sprint balik 0.7 m/s'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Berhenti sejenak'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180° kiri — orientasi awal'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Berhenti sejenak'},
            ]

        elif self.scenario == 3:
            # ──────────────────────────────────────────────
            # Skenario 3 — Stop and Go
            # Kondisi: DINAMIS (akselerasi & deselerasi berulang)
            # Tujuan  : Data P_inersia tinggi → XGBoost (kondisi dinamis)
            #           Periode "BERHENTI" menghasilkan data statis pendek
            # Pola    : Maju → berhenti → mundur → berhenti → putar
            # ──────────────────────────────────────────────
            return [
                {'linear':  0.5, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Maju 0.5 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear': -0.3, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Mundur 0.3 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear':  0.5, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Maju 0.5 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear': -0.3, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Mundur 0.3 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear':  0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Putar kiri 90° di tempat'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'BERHENTI sejenak'},
                {'linear':  0.5, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Maju 0.5 m/s (arah baru)'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear': -0.3, 'angular':  0.0, 'duration': 2.0,
                 'label': 'Mundur 0.3 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': 'BERHENTI'},
                {'linear':  0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Putar kiri 90° — kembali ke arah awal'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'BERHENTI sejenak'},
            ]

        elif self.scenario == 4:
            # ──────────────────────────────────────────────
            # Skenario 4 — Creep / Sangat Lambat
            # Kondisi: STATIS (kecepatan sangat rendah, P_gesek minimal)
            # Tujuan  : Data SOC baseline batas bawah → ARIMA
            #           Membuktikan ARIMA mampu menangkap tren penurunan
            #           SOC yang sangat lambat dan linier
            # Pola    : Kotak kecil + bolak-balik pelan, v = 0.1 m/s
            # Jarak   : 0.1 × 4.0 = 0.4 m/sisi → sangat aman
            # ──────────────────────────────────────────────
            return [
                {'linear': 0.1, 'angular':  0.0, 'duration': 4.0,
                 'label': 'Creep maju 0.1 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90° pelan'},
                {'linear': 0.1, 'angular':  0.0, 'duration': 4.0,
                 'label': 'Creep maju 0.1 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90° pelan'},
                {'linear': 0.1, 'angular':  0.0, 'duration': 4.0,
                 'label': 'Creep maju 0.1 m/s'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90° pelan'},
                {'linear': 0.1, 'angular':  0.0, 'duration': 4.0,
                 'label': 'Creep maju 0.1 m/s — kembali ke titik awal'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T90,
                 'label': 'Belok kiri 90° — orientasi kembali ke awal'},
                # Variasi bolak-balik lambat
                {'linear':  0.1, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Creep maju 0.1 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'BERHENTI'},
                {'linear': -0.1, 'angular':  0.0, 'duration': 3.0,
                 'label': 'Creep mundur 0.1 m/s'},
                {'linear':  0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'BERHENTI'},
            ]

        elif self.scenario == 5:
            # ──────────────────────────────────────────────
            # Skenario 5 — Akselerasi Agresif / Burst
            # Kondisi: DINAMIS EKSTREM (akselerasi + deselerasi mendadak)
            # Tujuan  : Membuktikan SOC drop tidak linier saat P_inersia
            #           spike → XGBoost jauh lebih baik dari ARIMA di sini
            # Pola    : Sprint 0.6 m/s → rem mendadak → putar → ulangi
            # Jarak   : 0.6 × 1.5 = 0.9 m/sprint → aman
            # Siklus  : ~17.3 detik
            # ──────────────────────────────────────────────
            return [
                {'linear': 0.6, 'angular':  0.0, 'duration': 1.5,
                 'label': 'BURST maju 0.6 m/s'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'REM mendadak — BERHENTI'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180° kiri'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Jeda stabilisasi'},
                {'linear': 0.6, 'angular':  0.0, 'duration': 1.5,
                 'label': 'BURST balik 0.6 m/s'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': 'REM mendadak — BERHENTI'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180° kiri — orientasi awal'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 0.5,
                 'label': 'Jeda stabilisasi'},
            ]

        elif self.scenario == 6:
            # ──────────────────────────────────────────────
            # Skenario 6 — Kecepatan Campuran (Mixed)
            # Kondisi: CAMPURAN — berisi periode STATIS dan DINAMIS
            # Tujuan  : Memvalidasi sistem hybrid ARIMA+XGBoost pada
            #           skenario operasional nyata (idle → jalan pelan
            #           → sedang → cepat → berhenti → ulang)
            # Pola    : IDLE → lambat → sedang → cepat → putar → ulang
            # Siklus  : ~40 detik — variasi lengkap dalam 1 siklus
            # ──────────────────────────────────────────────
            return [
                # ── Periode STATIS (ARIMA) ──
                {'linear': 0.0, 'angular':  0.0, 'duration': 4.0,
                 'label': '[STATIS] IDLE diam'},
                # ── Periode DINAMIS naik ──
                {'linear': 0.2, 'angular':  0.0, 'duration': 3.0,
                 'label': '[DINAMIS] Lambat 0.2 m/s (0.6m)'},
                {'linear': 0.4, 'angular':  0.0, 'duration': 2.5,
                 'label': '[DINAMIS] Sedang 0.4 m/s (1.0m)'},
                {'linear': 0.6, 'angular':  0.0, 'duration': 1.5,
                 'label': '[DINAMIS] Cepat 0.6 m/s (0.9m)'},
                # ── Berhenti + putar ──
                {'linear': 0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': '[STATIS] Berhenti setelah sprint'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180°'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 1.0,
                 'label': '[STATIS] Jeda putar'},
                # ── Periode DINAMIS balik (urutan terbalik) ──
                {'linear': 0.6, 'angular':  0.0, 'duration': 1.5,
                 'label': '[DINAMIS] Cepat balik 0.6 m/s (0.9m)'},
                {'linear': 0.4, 'angular':  0.0, 'duration': 2.5,
                 'label': '[DINAMIS] Sedang balik 0.4 m/s (1.0m)'},
                {'linear': 0.2, 'angular':  0.0, 'duration': 3.0,
                 'label': '[DINAMIS] Lambat balik 0.2 m/s (0.6m)'},
                # ── Kembali ke orientasi awal ──
                {'linear': 0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': '[STATIS] Berhenti'},
                {'linear': 0.0, 'angular':  0.5, 'duration': T180,
                 'label': 'Putar 180° — orientasi awal'},
                {'linear': 0.0, 'angular':  0.0, 'duration': 2.0,
                 'label': '[STATIS] IDLE akhir siklus'},
            ]

        elif self.scenario == 7:
            # ──────────────────────────────────────────────
            # Skenario 7 — Mixed Switching Demo
            # Kondisi: CAMPURAN (statis dan dinamis BERGANTIAN dalam satu siklus)
            # Tujuan  : Membuktikan Condition-Based Temporal Switching:
            #           ARIMA aktif saat statis, XGBoost aktif saat dinamis,
            #           bergantian dalam satu garis waktu kontinu.
            # Siklus  : ~82 detik → ±22 siklus dalam 1800 detik
            #           → 8+ transisi statis↔dinamis per siklus
            # Filter  : STATIS  = |a| < 0.15 m/s²  DAN  v < 0.6 m/s
            #           DINAMIS = salah satu melewati ambang batas
            # ──────────────────────────────────────────────
            return [
                # ═══ BLOK A — CREEP (STATIS) ═══════════════════════
                {'linear': 0.15, 'angular': 0.0, 'duration': 12.0,
                 'label': '[STATIS] Creep konstan 0.15 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.5,
                 'label': '[STATIS] Stop'},

                # ═══ BLOK B — SPRINT CEPAT (DINAMIS: v > 0.6) ═══════
                {'linear': 0.65, 'angular': 0.0, 'duration': 2.0,
                 'label': '[DINAMIS] Sprint 0.65 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.5,
                 'label': '[DINAMIS→STATIS] Rem mendadak'},
                {'linear': 0.0,  'angular': 0.5, 'duration': T180,
                 'label': '[DINAMIS] Putar 180° kiri'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.0,
                 'label': '[STATIS] Stop sejenak'},
                {'linear': 0.65, 'angular': 0.0, 'duration': 2.0,
                 'label': '[DINAMIS] Sprint balik 0.65 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 2.0,
                 'label': '[STATIS] Stop'},

                # ═══ BLOK C — KECEPATAN NORMAL (STATIS) ═════════════
                {'linear': 0.0,  'angular': 0.5, 'duration': T90,
                 'label': '[DINAMIS] Belok 90° kiri'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.0,
                 'label': '[STATIS] Stop'},
                {'linear': 0.25, 'angular': 0.0, 'duration': 8.0,
                 'label': '[STATIS] Konstan 0.25 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.5,
                 'label': '[STATIS] Stop'},

                # ═══ BLOK D — BURST AGRESIF (DINAMIS) ═══════════════
                {'linear': 0.7,  'angular': 0.0, 'duration': 1.5,
                 'label': '[DINAMIS] Burst 0.7 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 2.0,
                 'label': '[STATIS] Rem + stop'},
                {'linear': 0.0,  'angular': 0.5, 'duration': T180,
                 'label': '[DINAMIS] Putar 180° kiri'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 0.5,
                 'label': '[STATIS] Stop'},
                {'linear': 0.7,  'angular': 0.0, 'duration': 1.5,
                 'label': '[DINAMIS] Burst balik 0.7 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 2.0,
                 'label': '[STATIS] Stop'},

                # ═══ BLOK E — CREEP PELAN (STATIS) ══════════════════
                {'linear': 0.0,  'angular': 0.5, 'duration': T90,
                 'label': '[DINAMIS] Belok 90° kiri'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.0,
                 'label': '[STATIS] Stop'},
                {'linear': 0.1,  'angular': 0.0, 'duration': 15.0,
                 'label': '[STATIS] Creep 0.1 m/s'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.5,
                 'label': '[STATIS] Stop'},

                # ═══ BLOK F — KEMBALI ORIENTASI AWAL ════════════════
                {'linear': 0.0,  'angular': 0.5, 'duration': T90,
                 'label': '[DINAMIS] Belok 90° kiri — kembali orientasi'},
                {'linear': 0.0,  'angular': 0.0, 'duration': 1.0,
                 'label': '[STATIS] IDLE akhir siklus'},
            ]

        else:
            self.get_logger().error(
                f"Skenario {self.scenario} tidak dikenal! "
                f"Pilih antara 1 sampai 7.")
            return [{'linear': 0.0, 'angular': 0.0,
                     'duration': 1.0, 'label': 'IDLE (error)'}]


    # ──────────────────────────────────────────────────────────
    # TIMER CALLBACK
    # ──────────────────────────────────────────────────────────
    def _timer_callback(self):
        """Callback utama — publish cmd_vel sesuai fase aktif."""

        elapsed_total = time.time() - self.start_time

        # Durasi habis → hentikan robot dan keluar
        if elapsed_total >= DURATION_SECONDS:
            if self.running:
                self._stop_robot()
                self.running = False
                menit = int(elapsed_total // 60)
                detik = int(elapsed_total % 60)
                self.get_logger().info("=" * 60)
                self.get_logger().info(
                    f"✅ Skenario {self.scenario} SELESAI! "
                    f"Durasi: {menit}m {detik}s")
                self.get_logger().info(
                    "   Data tersimpan di InfluxDB bucket: polebot_data")
                self.get_logger().info(
                    "   Skenario berikutnya akan dimulai otomatis.")
                self.get_logger().info("=" * 60)
                raise SystemExit(0)
            return

        # Ambil dan eksekusi fase aktif
        phase = self.phases[self.phase_index]
        elapsed_phase = time.time() - self.phase_start

        if elapsed_phase >= phase['duration']:
            self.phase_index = (self.phase_index + 1) % len(self.phases)
            self.phase_start = time.time()
            phase = self.phases[self.phase_index]
            menit_e = int(elapsed_total // 60)
            detik_e = int(elapsed_total % 60)
            self.get_logger().info(
                f"  [{menit_e:02d}:{detik_e:02d}] → {phase['label']}")

        # Publish Twist
        twist = Twist()
        twist.linear.x  = phase['linear']
        twist.angular.z = phase['angular']
        self.cmd_pub.publish(twist)

        # Log progress setiap 5 menit
        current_second = int(elapsed_total)
        if (current_second % 300 == 0 and
                current_second > 0 and
                current_second != self.last_log_second):
            self.last_log_second = current_second
            menit_total = DURATION_SECONDS // 60
            menit_e = current_second // 60
            persen  = (elapsed_total / DURATION_SECONDS) * 100
            sisa    = DURATION_SECONDS - elapsed_total
            sisa_menit = int(sisa // 60)
            sisa_detik = int(sisa % 60)
            self.get_logger().info(
                f"📊 Progress: {menit_e}/{menit_total} menit "
                f"({persen:.0f}%) — "
                f"sisa {sisa_menit}m {sisa_detik}s — "
                f"v={phase['linear']:.1f} m/s"
            )


    # ──────────────────────────────────────────────────────────
    # HELPER FUNCTIONS
    # ──────────────────────────────────────────────────────────
    def _stop_robot(self):
        """Hentikan robot — publish Twist nol beberapa kali."""
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        for _ in range(10):
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

    def _log_scenario_info(self):
        """Log deskripsi singkat skenario yang dipilih."""
        info = {
            1: ("Normal / Baseline",
                "Kotak kecil, v=0.3 m/s",
                "SOC statis → data latih ARIMA (baseline)",
                "STATIS"),
            2: ("Beban Tinggi",
                "Bolak-balik cepat, v=0.7 m/s",
                "SOC drain tinggi → data latih XGBoost (kecepatan tinggi)",
                "DINAMIS"),
            3: ("Stop and Go",
                "Maju-berhenti-mundur berulang, v=0.5 m/s",
                "P_inersia dominan → data latih XGBoost (akselerasi)",
                "DINAMIS"),
            4: ("Creep / Sangat Lambat",
                "Kotak mini + bolak-balik, v=0.1 m/s",
                "SOC tren linier sangat lambat → data latih ARIMA",
                "STATIS"),
            5: ("Akselerasi Agresif / Burst",
                "Sprint pendek + rem mendadak, v=0.6 m/s",
                "Spike P_inersia → membuktikan ARIMA gagal, XGBoost unggul",
                "DINAMIS EKSTREM"),
            6: ("Kecepatan Campuran / Mixed",
                "IDLE → 0.2 → 0.4 → 0.6 m/s → putar → ulang",
                "Validasi sistem hybrid ARIMA+XGBoost operasional nyata",
                "CAMPURAN"),
            7: ("Mixed Switching Demo",
                "Creep 0.15 → Sprint 0.65 → Normal 0.25 → Burst 0.7 → Creep 0.1, berulang",
                "Pembuktian Condition-Based Temporal Switching: "
                "ARIMA (statis) ↔ XGBoost (dinamis) bergantian",
                "CAMPURAN SWITCHING"),
        }
        nama, pola, tujuan, kondisi = info.get(
            self.scenario, ("Unknown", "-", "-", "-"))
        self.get_logger().info(f"  Nama     : {nama}")
        self.get_logger().info(f"  Kondisi  : {kondisi}")
        self.get_logger().info(f"  Pola     : {pola}")
        self.get_logger().info(f"  Tujuan   : {tujuan}")
        self.get_logger().info(
            f"  Durasi   : {DURATION_SECONDS // 60} menit")

        # Penanda kondisi untuk referensi filter data
        self.get_logger().info("")
        if kondisi == "STATIS":
            self.get_logger().info(
                "  📘 [ARIMA] Data skenario ini digunakan untuk "
                "melatih ARIMA (SOC kondisi statis)")
        elif kondisi == "DINAMIS" or kondisi == "DINAMIS EKSTREM":
            self.get_logger().info(
                "  📗 [XGBoost] Data skenario ini digunakan untuk "
                "melatih XGBoost (SOC kondisi dinamis)")
        elif kondisi == "CAMPURAN SWITCHING":
            self.get_logger().info(
                "  🔀 [HYBRID SWITCHING] Skenario ini dirancang khusus "
                "memperlihatkan ARIMA ↔ XGBoost bergantian per segmen.")
            self.get_logger().info(
                "  ▶ Setelah selesai, jalankan: hybrid_switching_predictor.py")
        else:
            self.get_logger().info(
                "  📙 [HYBRID] Skenario ini berisi keduanya — "
                "periode statis (ARIMA) & dinamis (XGBoost)")


    # ──────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────
    def destroy_node(self):
        try:
            self._stop_robot()
        except Exception:
            pass
        self.get_logger().info(
            f"[scenario_runner] Skenario {self.scenario} "
            f"dihentikan — robot dimatikan.")
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ScenarioRunner()
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