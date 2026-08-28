import minimalmodbus
import serial
import time

PORT          = '/dev/ttyUSB0'
SLAVE_ADDRESS = 1
BAUDRATE      = 9600

SHUNT_RANGE_REGISTER = 0x0003
SHUNT_50A            = 0x0001   # nilai untuk 50A


def connect_pzem():
    inst = minimalmodbus.Instrument(PORT, SLAVE_ADDRESS)
    inst.serial.baudrate = BAUDRATE
    inst.serial.bytesize = 8
    inst.serial.parity   = serial.PARITY_NONE
    inst.serial.stopbits = 2
    inst.serial.timeout  = 1
    inst.mode = minimalmodbus.MODE_RTU
    inst.clear_buffers_before_each_transaction = True
    return inst


def read_shunt_range(inst):
    """Baca setting range shunt saat ini (Holding Register 0x0003)."""
    try:
        val = inst.read_register(SHUNT_RANGE_REGISTER,
                                 0, functioncode=3)
        names = {0: '100A', 1: '50A', 2: '200A', 3: '300A'}
        return val, names.get(val, f'Unknown ({val})')
    except Exception as e:
        return None, str(e)


def set_shunt_range(inst, value):
    """Tulis range shunt ke Holding Register 0x0003."""
    try:
        inst.write_register(SHUNT_RANGE_REGISTER,
                            value, functioncode=6)
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("SET RANGE SHUNT PZEM-017 → 50A")
    print("=" * 50)

    print(f"\nMenghubungkan ke {PORT} ...")
    try:
        inst = connect_pzem()
    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)
    print("Terhubung.\n")

    # Baca setting saat ini
    val, name = read_shunt_range(inst)
    if val is None:
        print(f"[ERROR] Gagal baca setting awal: {name}")
        exit(1)
    print(f"Setting shunt saat ini : {name} (register value = {val})")

    if val == SHUNT_50A:
        print("\nShunt sudah diset ke 50A. Tidak perlu diubah.")
        exit(0)

    # Tulis setting baru
    print(f"Menulis setting baru   : 50A (value = {SHUNT_50A}) ...")
    ok = set_shunt_range(inst, SHUNT_50A)
    if not ok:
        print("[FATAL] Gagal menulis setting.")
        exit(1)

    # Tunggu PZEM menyimpan
    time.sleep(0.5)

    # Verifikasi
    val2, name2 = read_shunt_range(inst)
    print(f"Verifikasi setting baru: {name2} (register value = {val2})")

    if val2 == SHUNT_50A:
        print("\n✓ BERHASIL — Range shunt tersimpan ke 50A.")
        print("  Setting ini permanen, tidak perlu diulang.")
    else:
        print("\n✗ GAGAL — Nilai tidak berubah. Coba jalankan ulang.")