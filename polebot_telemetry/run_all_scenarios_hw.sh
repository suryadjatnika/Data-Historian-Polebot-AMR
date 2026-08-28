#!/bin/bash
# ============================================================
# run_all_scenarios_hw.sh — Jalankan 7 skenario BERURUTAN (HARDWARE)
# ============================================================
# Versi HARDWARE: TIDAK bergantung Gazebo/battery_physics_sim.
# Mengecek node hardware (driver Tongyi, joint_states_bridge,
# pzem_publisher, telemetry_logger), lalu menjalankan
# scenario_runner2.py untuk skenario 1..7 secara berurutan.
#
# PENTING: Script ini mencatat timestamp START/STOP tiap skenario
#          ke file log, supaya nanti bisa dipisah per-skenario
#          saat export dari InfluxDB.
#
# Prasyarat (jalankan DULU di terminal terpisah, biarkan hidup):
#   T1: setup CAN               (setup_can0_500k.sh)
#   T2: driver Tongyi           (tongyi_bringup.launch.py)
#   T3: enable motor            (service enable)
#   T4: joint_states_bridge.py
#   T5: pzem_publisher.py
#   T6: telemetry_logger.py
#   T7: SCRIPT INI
# ============================================================

# ── Konfigurasi ──────────────────────────────────────────────
START_SCENARIO=1
END_SCENARIO=7
DURASI_PER_SKENARIO=300      # detik (5 menit). Ubah ke 180 utk 3 menit.
JEDA_ANTAR_SKENARIO=10       # detik jeda antar skenario
SCENARIO_SCRIPT="$HOME/polebot_ws/src/polebot_telemetry/polebot_telemetry/scenario_runner2.py"
LOG_DIR="$HOME/polebot_scenario_logs"
WORKSPACE="$HOME/polebot_ws/polebot_dev_ws"

# ── Warna ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

mkdir -p "$LOG_DIR"
source "$WORKSPACE/install/setup.bash" 2>/dev/null
TIMESTAMP_LOG="$LOG_DIR/scenario_timestamps_$(date +%Y%m%d_%H%M%S).txt"

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════${NC}"
}
print_ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }

countdown() {
    local detik=$1; local pesan=$2
    for ((i=detik; i>0; i--)); do
        printf "\r${YELLOW}[WAIT]${NC} $pesan: ${BOLD}%3d detik${NC}" $i
        sleep 1
    done
    printf "\r${GREEN}[GO]${NC}   $pesan: SELESAI           \n"
}

# ── Cek node hardware aktif ──────────────────────────────────
print_header "CEK PRASYARAT NODE HARDWARE"
nodes=$(ros2 node list 2>/dev/null)

check_node() {
    local pattern=$1; local nama=$2; local cmd=$3
    if echo "$nodes" | grep -q "$pattern"; then
        print_ok "$nama → AKTIF"
        return 0
    else
        print_error "$nama → TIDAK AKTIF"
        print_warn "  Jalankan dulu: $cmd"
        return 1
    fi
}

siap=0
check_node "tongyi_canopen_node" "Driver Tongyi     " "ros2 launch tongyi_canopen_driver tongyi_bringup.launch.py" || siap=1
check_node "joint_states_bridge" "Joint States Bridge" "python3 .../joint_states_bridge.py" || siap=1
check_node "pzem_publisher"      "PZEM Publisher    " "python3 .../pzem_publisher.py" || siap=1
check_node "telemetry_logger"    "Telemetry Logger  " "python3 .../telemetry_logger.py" || siap=1

# Cek topik /odom ada publisher
odom_pub=$(ros2 topic info /odom 2>/dev/null | grep -i "Publisher count" | grep -o '[0-9]*')
if [ "$odom_pub" -ge 1 ] 2>/dev/null; then
    print_ok "/odom punya publisher ($odom_pub)"
else
    print_error "/odom TIDAK ada publisher — driver/robot belum siap"
    siap=1
fi

if [ $siap -ne 0 ]; then
    print_error "Ada prasyarat belum siap. Perbaiki dulu sebelum lanjut."
    exit 1
fi

# ── Konfirmasi keamanan ──────────────────────────────────────
print_header "SIAP MENJALANKAN ${END_SCENARIO} SKENARIO"
total_menit=$(( (END_SCENARIO - START_SCENARIO + 1) * DURASI_PER_SKENARIO / 60 ))
print_info "Total estimasi waktu: ~${total_menit} menit"
print_info "Durasi per skenario : $((DURASI_PER_SKENARIO / 60)) menit"
print_info "Log timestamp       : $TIMESTAMP_LOG"
echo ""
print_warn "PASTIKAN:"
print_warn "  - Pengawas siap dengan Emergency Stop"
print_warn "  - Lorong lurus bebas ~2-2.5 meter"
print_warn "  - Baterai cukup untuk ~${total_menit} menit operasi"
echo ""
read -p "$(echo -e ${BOLD}Tekan ENTER untuk MULAI, atau Ctrl+C untuk batal...${NC})"

echo "# Log Timestamp Skenario Hardware — $(date)" > "$TIMESTAMP_LOG"
echo "# Format: skenario | start_utc | stop_utc" >> "$TIMESTAMP_LOG"

# ── Loop skenario ────────────────────────────────────────────
waktu_mulai_total=$(date +%s)

for skenario in $(seq $START_SCENARIO $END_SCENARIO); do
    print_header "SKENARIO $skenario / $END_SCENARIO"

    start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    print_info "START: $start_utc"

    # Jalankan scenario_runner2.py via python3
    python3 "$SCENARIO_SCRIPT" --ros-args \
        -p scenario:=$skenario \
        -p duration:=$DURASI_PER_SKENARIO

    stop_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    print_ok "STOP: $stop_utc"

    # Catat timestamp untuk split data nanti
    echo "$skenario | $start_utc | $stop_utc" >> "$TIMESTAMP_LOG"

    # Jeda antar skenario (kecuali skenario terakhir)
    if [ $skenario -lt $END_SCENARIO ]; then
        countdown $JEDA_ANTAR_SKENARIO "Jeda sebelum skenario berikutnya"
    fi
done

# ── Selesai ──────────────────────────────────────────────────
waktu_selesai_total=$(date +%s)
durasi_total=$(( waktu_selesai_total - waktu_mulai_total ))

print_header "SEMUA SKENARIO SELESAI"
print_ok "Total durasi: $((durasi_total / 60)) menit $((durasi_total % 60)) detik"
print_ok "Timestamp tersimpan: $TIMESTAMP_LOG"
echo ""
print_info "Isi timestamp log (untuk split data saat export):"
cat "$TIMESTAMP_LOG"
echo ""
print_info "Langkah berikutnya: export data per skenario dari InfluxDB"
print_info "menggunakan timestamp di atas."