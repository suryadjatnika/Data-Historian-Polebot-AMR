#!/bin/bash
# ── Konfigurasi ───────────────────────────────────────────────
START_SCENARIO=1
END_SCENARIO=6
JEDA_ANTAR_SKENARIO=20   # detik jeda antar skenario (stabilisasi)
JEDA_SEBELUM_MATIKAN=10  # detik tunggu sebelum matikan logger (flush data)
LOG_DIR="$HOME/polebot_scenario_logs"
WORKSPACE="$HOME/polebot_ws"

# ── Warna terminal ─────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Setup ──────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
source "$WORKSPACE/install/setup.bash" 2>/dev/null

# ── Helper functions ───────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════${NC}"
}
print_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
print_ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

countdown() {
    local detik=$1
    local pesan=$2
    for ((i=detik; i>0; i--)); do
        printf "\r${YELLOW}[WAIT]${NC} $pesan: ${BOLD}%3d detik${NC}" $i
        sleep 1
    done
    printf "\r${GREEN}[GO]${NC}   $pesan: SELESAI           \n"
}

format_durasi() {
    local total_detik=$1
    local jam=$((total_detik / 3600))
    local menit=$(( (total_detik % 3600) / 60 ))
    local detik=$((total_detik % 60))
    if [ $jam -gt 0 ]; then
        printf "%d jam %d menit %d detik" $jam $menit $detik
    else
        printf "%d menit %d detik" $menit $detik
    fi
}

# ── Cek ROS 2 ─────────────────────────────────────────────────
check_ros() {
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS 2 tidak ditemukan!"
        exit 1
    fi
    print_ok "ROS 2 tersedia"
}

# ── Cek prerequisites ──────────────────────────────────────────
check_prerequisites() {
    print_info "Mengecek node yang sedang berjalan..."
    sleep 2
    local nodes
    nodes=$(ros2 node list 2>/dev/null)
    local ok=true

    if echo "$nodes" | grep -q "telemetry_logger"; then
        print_ok "telemetry_logger     → AKTIF ✓"
    else
        print_warn "telemetry_logger     → TIDAK AKTIF ⚠"
        print_warn "  Jalankan: python3 ~/polebot_ws/src/polebot_telemetry/polebot_telemetry/telemetry_logger.py"
        ok=false
    fi

    if echo "$nodes" | grep -q "battery_physics_sim"; then
        print_ok "battery_physics_sim  → AKTIF ✓"
    else
        print_warn "battery_physics_sim  → TIDAK AKTIF ⚠"
        print_warn "  Jalankan: python3 ~/polebot_ws/src/polebot_telemetry/polebot_telemetry/battery_physics_sim.py"
        ok=false
    fi

    if echo "$nodes" | grep -qE "gz|gazebo|sim"; then
        print_ok "Gazebo               → AKTIF ✓"
    else
        print_warn "Gazebo               → TIDAK TERDETEKSI ⚠"
        ok=false
    fi

    if [ "$ok" = false ]; then
        echo ""
        print_warn "Ada node yang belum aktif."
        echo -e "${YELLOW}Lanjutkan tetap? (y/N):${NC} \c"
        read -r jawab
        if [[ ! "$jawab" =~ ^[Yy]$ ]]; then
            print_info "Dibatalkan. Jalankan node yang diperlukan terlebih dahulu."
            exit 0
        fi
        print_warn "Melanjutkan meski ada node yang tidak aktif..."
    fi
}

# ══════════════════════════════════════════════════════════════
# FUNGSI AUTO-SHUTDOWN NODE PENDUKUNG
# ══════════════════════════════════════════════════════════════
shutdown_support_nodes() {
    echo ""
    print_header "AUTO-SHUTDOWN NODE PENDUKUNG"
    print_info "Menunggu ${JEDA_SEBELUM_MATIKAN} detik agar data terakhir ter-flush ke InfluxDB..."
    countdown $JEDA_SEBELUM_MATIKAN "Flush data terakhir"

    # Matikan telemetry_logger.py
    if pkill -f "telemetry_logger.py" 2>/dev/null; then
        print_ok "telemetry_logger.py  → DIHENTIKAN ✓"
    else
        print_warn "telemetry_logger.py  → tidak ditemukan (mungkin sudah mati)"
    fi

    # Matikan battery_physics_sim.py
    if pkill -f "battery_physics_sim.py" 2>/dev/null; then
        print_ok "battery_physics_sim.py → DIHENTIKAN ✓"
    else
        print_warn "battery_physics_sim.py → tidak ditemukan (mungkin sudah mati)"
    fi

    # Tunggu sebentar dan verifikasi
    sleep 3
    local masih_hidup=false
    if pgrep -f "telemetry_logger.py" > /dev/null 2>&1; then
        print_warn "telemetry_logger masih berjalan — force kill..."
        pkill -9 -f "telemetry_logger.py" 2>/dev/null
        masih_hidup=true
    fi
    if pgrep -f "battery_physics_sim.py" > /dev/null 2>&1; then
        print_warn "battery_physics_sim masih berjalan — force kill..."
        pkill -9 -f "battery_physics_sim.py" 2>/dev/null
        masih_hidup=true
    fi

    if [ "$masih_hidup" = false ]; then
        print_ok "Semua node pendukung berhasil dihentikan"
    else
        print_warn "Force kill sudah dilakukan"
    fi

    print_info "Data NOL idle tidak akan tersimpan karena logger sudah mati ✓"
}


# ══════════════════════════════════════════════════════════════
# MULAI EKSEKUSI
# ══════════════════════════════════════════════════════════════
clear
print_info "Skenario        : $START_SCENARIO sampai $END_SCENARIO"
print_info "Durasi total    : $(format_durasi $(( (END_SCENARIO - START_SCENARIO + 1) * 1800 )))"
print_info "Jeda antar sk.  : $JEDA_ANTAR_SKENARIO detik"
print_info "Log disimpan    : $LOG_DIR"
echo ""

check_ros
check_prerequisites

echo ""
echo -e "${BOLD}Semua cek selesai. Mulai pengumpulan data? (y/N):${NC} \c"
read -r konfirmasi
if [[ ! "$konfirmasi" =~ ^[Yy]$ ]]; then
    print_info "Dibatalkan."
    exit 0
fi

WAKTU_MULAI=$(date +%s)
WAKTU_MULAI_STR=$(date '+%H:%M:%S %d/%m/%Y')

echo ""
print_ok "Memulai pada $WAKTU_MULAI_STR"
print_info "Perkiraan selesai: $(date -d "+$(( (END_SCENARIO - START_SCENARIO + 1) * 1800 )) seconds" '+%H:%M:%S %d/%m/%Y' 2>/dev/null || echo 'hitung manual')"
echo ""

# Nama skenario
declare -A NAMA_SKENARIO=(
    [1]="Normal / Baseline (0.3 m/s) [STATIS]"
    [2]="Beban Tinggi (0.7 m/s) [DINAMIS]"
    [3]="Stop and Go (0.5 m/s) [DINAMIS]"
    [4]="Creep / Sangat Lambat (0.1 m/s) [STATIS]"
    [5]="Akselerasi Agresif/Burst (0.6 m/s) [DINAMIS EKSTREM]"
    [6]="Kecepatan Campuran / Mixed [CAMPURAN]"
)

# ── Loop utama ────────────────────────────────────────────────
SKENARIO_BERHASIL=0
SKENARIO_GAGAL=0

for ((s=START_SCENARIO; s<=END_SCENARIO; s++)); do

    # Jeda antar skenario (kecuali skenario pertama)
    if [ $s -gt $START_SCENARIO ]; then
        echo ""
        print_info "Jeda $JEDA_ANTAR_SKENARIO detik — stabilisasi simulator..."
        countdown $JEDA_ANTAR_SKENARIO "Lanjut ke skenario $s"
    fi

    LOG_FILE="$LOG_DIR/scenario_${s}_$(date +%Y%m%d_%H%M%S).log"

    print_header "SKENARIO $s — ${NAMA_SKENARIO[$s]}"
    print_info "Mulai   : $(date '+%H:%M:%S %d/%m/%Y')"
    print_info "Log     : $LOG_FILE"
    print_info "Durasi  : 30 menit"
    echo ""

    ros2 run polebot_telemetry scenario_runner2 \
        --ros-args -p scenario:=$s \
        2>&1 | tee "$LOG_FILE"

    hasil=${PIPESTATUS[0]}

    if [ $hasil -eq 0 ] || [ $hasil -eq 130 ]; then
        print_ok "Skenario $s selesai pada $(date '+%H:%M:%S') ✓"
        SKENARIO_BERHASIL=$((SKENARIO_BERHASIL + 1))
    else
        print_warn "Skenario $s exit code=$hasil — lanjut ke skenario berikutnya..."
        SKENARIO_GAGAL=$((SKENARIO_GAGAL + 1))
    fi

done

# ── AUTO-SHUTDOWN setelah semua skenario selesai ──────────────
shutdown_support_nodes

# ── Ringkasan akhir ───────────────────────────────────────────
WAKTU_SELESAI=$(date +%s)
TOTAL_DETIK=$((WAKTU_SELESAI - WAKTU_MULAI))

echo ""
print_header "PENGUMPULAN DATA SELESAI"
print_info "Mulai    : $WAKTU_MULAI_STR"
print_info "Selesai  : $(date '+%H:%M:%S %d/%m/%Y')"
print_info "Durasi   : $(format_durasi $TOTAL_DETIK)"
echo ""
print_ok "Skenario berhasil : $SKENARIO_BERHASIL"
[ $SKENARIO_GAGAL -gt 0 ] && print_warn "Skenario gagal    : $SKENARIO_GAGAL"
echo ""
print_info "Log tersimpan di  : $LOG_DIR"
print_info "Data tersimpan di : InfluxDB bucket polebot_data"
echo ""
echo -e "${BOLD}${GREEN}Langkah berikutnya (setelah bangun tidur): 😊${NC}"
echo "  1. Cek data di Grafana       : http://localhost:3000"
echo "  2. Latih ulang ARIMA         :"
echo "     python3 ~/polebot_ws/src/polebot_telemetry/polebot_telemetry/arima_predictor.py"
echo "  3. Latih ulang XGBoost       :"
echo "     python3 ~/polebot_ws/src/polebot_telemetry/polebot_telemetry/xgboost_predictor.py"
echo "  4. Generate comparison plot  :"
echo "     python3 ~/polebot_ws/src/polebot_telemetry/polebot_telemetry/comparison_plot.py"
echo ""