import minimalmodbus
import serial
import time

PORT          = '/dev/ttyUSB1'
SLAVE_ADDRESS = 1
BAUDRATE      = 9600


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


def read_pzem(inst):
    try:
        regs = inst.read_registers(0x0000, 8, functioncode=4)

        return {
            'voltage_V' : regs[0] * 0.01,
            'current_A' : regs[1] * 0.01,
            'power_W'   : (regs[2] + (regs[3] << 16)) * 0.1,
            'energy_Wh' : regs[4] + (regs[5] << 16),
            'hv_alarm'  : regs[6] == 0xFFFF,
            'lv_alarm'  : regs[7] == 0xFFFF,
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


if __name__ == '__main__':
    print(f"Menghubungkan ke PZEM-017 di {PORT} ...")
    try:
        inst = connect_pzem()
    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)

    print(f"Terhubung. Slave={SLAVE_ADDRESS}, Baud={BAUDRATE}")
    print("─" * 65)
    print(f"{'Tegangan':>10} {'Arus':>10} {'Daya':>12} {'Energi':>10} {'HV':>5} {'LV':>5}")
    print(f"{'(V)':>10} {'(A)':>10} {'(W)':>12} {'(Wh)':>10} {'Alrm':>5} {'Alrm':>5}")
    print("─" * 65)

    try:
        while True:
            d = read_pzem(inst)
            if d:
                print(
                    f"{d['voltage_V']:>10.2f} "
                    f"{d['current_A']:>10.2f} "
                    f"{d['power_W']:>12.1f} "
                    f"{d['energy_Wh']:>10d} "
                    f"{'YES' if d['hv_alarm'] else '-':>5} "
                    f"{'YES' if d['lv_alarm'] else '-':>5}"
                )
            else:
                print("Tidak ada data")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDihentikan.")