#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gem5 SE mode simulation with SiFive U74 core.
Uses the prebuilt RISC-V Matched board from gem5 standard library.
"""

from gem5.prebuilt.riscvmatched.riscvmatched_board import RiscvMatchedBoard
from gem5.resources.resource import Resource
from gem5.simulate.simulator import Simulator

# 1. Создание платы с ядром U74
#    RiscvMatchedBoard автоматически использует U74Processor
board = RiscvMatchedBoard()

# Плата сконфигурирована следующим образом:
# - CPU: U74Core (модель SiFive U74) [citation:1][citation:4]
# - Branch predictor: TournamentBP [citation:1]
# - Latency: IntFU=1, IntMulFU=3, IntDivFU=6, MemReadFU=2, MemWriteFU=2 [citation:1]
# - Поддержка векторных инструкций отключена (U74 их не поддерживает) [citation:7]

# 2. Вывод информации о CPU (опционально)

print(f"Core type: U74 (SiFive U64 Core Complex)")

# 3. Загрузка рабочей нагрузки
#    Используем стандартную программу riscv-hello
workload = Resource("riscv-hello")
board.set_se_binary_workload(workload)

# 4. Настройка симулятора
simulator = Simulator(board=board)

# 5. Запуск
print("\nStarting simulation with U74 core...")
simulator.run()
