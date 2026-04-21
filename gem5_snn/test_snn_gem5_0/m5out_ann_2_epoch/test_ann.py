import m5
from m5.objects import *

# ==============================
# Система
# ==============================
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"             # можно подобрать под твой целевой SoC
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"                   # timing simulation
system.mem_ranges = [AddrRange("1GB")]       # 1 GB DDR4

# ==============================
# CPU
# ==============================
# Для ARMv7‑A в SE‑режиме используем TimingSimpleCPU
# В ARM‑конфигах обычно: TimingSimpleCPU
system.cpu = TimingSimpleCPU()

# Включаем Thumb‑поддержку – если gem5 даёт такую опцию, можно обозначить
# (обычно ISA‑поддержка задаётся самим ARM‑CPU типом, не тут).

# Для SE‑модели устанавливаем workload
process = Process()
process.cmd = ["ann_test"]                     # твой SNN‑бинарник, скомпилированный под ARMv7‑A
system.cpu.workload = process
system.cpu.createThreads()

# ==============================
# Кэши L1 (I+D)
# ==============================
class L1ICache(Cache):
    size = "8kB"
    assoc = 4
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 20
    writeback_clean = True
    # 64‑байтные строки по умолчанию

class L1DCache(Cache):
    size = "8kB"
    assoc = 4
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 20
    writeback_clean = True

system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

# ==============================
# Системная когерентная шина
# ==============================
system.membus = SystemXBar()

system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.mem_side = system.membus.cpu_side_ports

# ==============================
# DDR4‑память (DDR4_2400)
# ==============================
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]  # 0–1 GB

system.mem_ctrl.port = system.membus.mem_side_ports

# ==============================
# Рут‑система
# ==============================
root = Root(full_system=False, system=system)
m5.instantiate()
print("Beginning simulation!")
exit_event = m5.simulate()
print(f"Exit event: {exit_event.getCause()}")
print(f"Simulation took {m5.curTick()} ticks")
