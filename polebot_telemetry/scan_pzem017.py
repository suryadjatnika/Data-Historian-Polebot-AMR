import minimalmodbus
import serial
import time

PORT = '/dev/ttyUSB0'

BAUDRATES       = [9600, 4800, 19200]
PARITIES        = [serial.PARITY_NONE, serial.PARITY_EVEN]
PARITY_NAMES    = {serial.PARITY_NONE: 'None', serial.PARITY_EVEN: 'Even'}
STOPBITS_LIST   = [2, 1]
SLAVE_ADDRESSES = [1, 2, 3, 0xF8]

total = len(BAUDRATES) * len(PARITIES) * len(STOPBITS_LIST) * len(SLAVE_ADDRESSES)

print("=" * 65)
print("SCAN KOMPREHENSIF PZEM-017")
print("=" * 65)
print(f"Port           : {PORT}")
print(f"Total kombinasi: {total}")
print("=" * 65)

found      = False
attempt    = 0

for baudrate in BAUDRATES:
    for parity in PARITIES:
        for stopbits in STOPBITS_LIST:
            for addr in SLAVE_ADDRESSES:
                attempt += 1
                label = (f"[{attempt:02d}/{total}] "
                         f"baud={baudrate}, parity={PARITY_NAMES[parity]}, "
                         f"stop={stopbits}, addr={addr}")
                try:
                    inst = minimalmodbus.Instrument(PORT, addr)
                    inst.serial.baudrate = baudrate
                    inst.serial.bytesize = 8
                    inst.serial.parity   = parity
                    inst.serial.stopbits = stopbits
                    inst.serial.timeout  = 0.8
                    inst.mode = minimalmodbus.MODE_RTU
                    inst.clear_buffers_before_each_transaction = True

                    val = inst.read_register(0x0000, 2, functioncode=4)

                    print(f"\n{'='*65}")
                    print(f"  BERHASIL! {label}")
                    print(f"  Voltage terbaca = {val:.2f} V")
                    print(f"{'='*65}")
                    print(f"\n  Konfigurasi yang benar untuk test_pzem017.py:")
                    print(f"  PORT          = '{PORT}'")
                    print(f"  SLAVE_ADDRESS = {addr}")
                    print(f"  BAUDRATE      = {baudrate}")
                    print(f"  PARITY        = serial.{parity!r}")
                    print(f"  STOPBITS      = {stopbits}")
                    found = True
                    break

                except Exception as e:
                    err = type(e).__name__
                    print(f"  MISS {label} → {err}")

                time.sleep(0.2)

            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print("\n" + "="*65)
    print("SEMUA KOMBINASI GAGAL.")
    print("="*65)
    print("\nIni bukan masalah software. Kemungkinan hardware:")
    print()
    print("1. KABEL SALAH TERMINAL")
    print("   Pastikan kabel 4-pin terhubung ke terminal RS485")
    print("   (yang berlabel '5V B A GND'), bukan ke terminal V/I")
    print()
    print("2. A DAN B TERBALIK")
    print("   Tukar posisi kabel A dan B di terminal USB485,")
    print("   lalu jalankan scan ini lagi.")
    print()
    print("3. PZEM-017 TIDAK MENYALA")
    print("   Cek apakah ada LED atau layar di PZEM yang menyala.")
    print("   Kalau tidak ada tanda kehidupan, suplai 5V belum masuk.")
    print()
    print("4. KABEL PUTUS / KONEKSI LONGGAR")
    print("   Cek apakah semua 4 kabel sudah masuk dan dikencangkan")
    print("   dengan benar di terminal block (putar sekrup terminal).")