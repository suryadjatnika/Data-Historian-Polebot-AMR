import serial
import time
import struct

PORT     = '/dev/ttyUSB0'
BAUDRATE = 9600


def crc16_modbus(data: bytes) -> int:
    """Hitung CRC-16 standar Modbus."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_frame(slave_addr: int, func_code: int,
                reg_start: int, reg_count: int) -> bytes:
    """Bangun frame Modbus RTU lengkap dengan CRC."""
    payload = struct.pack('>BBHH', slave_addr, func_code,
                          reg_start, reg_count)
    crc = crc16_modbus(payload)
    # CRC dikirim Little-Endian (Low byte dulu)
    return payload + struct.pack('<H', crc)


def parse_response(data: bytes) -> dict | None:
    """
    Parse respons register 0x0000 (Voltage) dari PZEM-017.
    Respons yang diharapkan: 07 bytes
    [addr][fc][byte_count][val_hi][val_lo][crc_lo][crc_hi]
    """
    if len(data) < 7:
        return None
    addr      = data[0]
    fc        = data[1]
    byte_cnt  = data[2]
    raw_val   = (data[3] << 8) | data[4]
    crc_recv  = (data[6] << 8) | data[5]
    crc_calc  = crc16_modbus(data[:5])

    return {
        'addr'     : addr,
        'fc'       : fc,
        'byte_cnt' : byte_cnt,
        'raw'      : raw_val,
        'voltage'  : raw_val * 0.01,
        'crc_ok'   : crc_recv == crc_calc,
    }


def try_read(ser: serial.Serial, slave_addr: int,
             delay_ms: int) -> bytes:
    """Kirim 1 request, tunggu delay_ms, baca semua respons."""
    frame = build_frame(slave_addr, 0x04, 0x0000, 0x0001)

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(frame)
    ser.flush()

    time.sleep(delay_ms / 1000)

    return ser.read(ser.in_waiting or 32)


print("=" * 60)
print("RAW SERIAL DIAGNOSTIC — PZEM-017")
print("=" * 60)

try:
    ser = serial.Serial(
        port     = PORT,
        baudrate = BAUDRATE,
        bytesize = 8,
        parity   = serial.PARITY_NONE,
        stopbits = 2,
        timeout  = 0.5,
    )
except Exception as e:
    print(f"[FATAL] Tidak bisa buka port: {e}")
    exit(1)

print(f"Port terbuka: {PORT} @ {BAUDRATE} baud\n")

# ── Test 1: Kirim frame mentah, lihat ada respons atau tidak ──
print("TEST 1 — Cek apakah ada respons sama sekali")
print("-" * 50)

frame_slave1 = build_frame(1, 0x04, 0x0000, 0x0001)
print(f"Frame dikirim: {frame_slave1.hex(' ').upper()}")

for delay in [100, 200, 500, 1000]:
    ser.reset_input_buffer()
    ser.write(frame_slave1)
    ser.flush()
    time.sleep(delay / 1000)
    raw = ser.read(ser.in_waiting or 64)

    if raw:
        print(f"\n  [HIT] delay={delay}ms → terima {len(raw)} byte: "
              f"{raw.hex(' ').upper()}")
        result = parse_response(raw)
        if result:
            status = "CRC OK" if result['crc_ok'] else "CRC FAIL"
            print(f"         Voltage = {result['voltage']:.2f} V ({status})")
        break
    else:
        print(f"  delay={delay}ms → tidak ada respons")

print()

# ── Test 2: Coba slave address lain ──
print("TEST 2 — Scan slave address 1–4 dan 248")
print("-" * 50)

for addr in [1, 2, 3, 4, 0xF8]:
    frame = build_frame(addr, 0x04, 0x0000, 0x0001)
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    time.sleep(0.5)
    raw = ser.read(ser.in_waiting or 64)

    status = f"{len(raw)} byte: {raw.hex(' ').upper()}" if raw else "tidak ada respons"
    print(f"  addr={addr:3d}: {status}")

print()

# ── Test 3: Broadcast (addr=0) ──
print("TEST 3 — Broadcast address 0x00")
print("-" * 50)
frame_bc = build_frame(0, 0x04, 0x0000, 0x0001)
ser.reset_input_buffer()
ser.write(frame_bc)
ser.flush()
time.sleep(0.5)
raw = ser.read(ser.in_waiting or 64)
status = f"{len(raw)} byte: {raw.hex(' ').upper()}" if raw else "tidak ada respons"
print(f"  broadcast: {status}")

ser.close()
print("\nSelesai. Port ditutup.")
print()
print("INTERPRETASI:")
print("  Ada byte respons    → timing fix berhasil, lanjut ke langkah berikutnya")
print("  Tidak ada sama sekali → kemungkinan RX path USB485 bermasalah secara hardware")