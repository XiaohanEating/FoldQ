"""
加载 HAL 仓库自带的 qLDPC 校验矩阵（HAL/hal/codes/qTanner_pcms）

文件格式：0/1 矩阵，每行一个稳定子，列为数据比特。
"""

import os
from typing import Dict, Tuple

import numpy as np

from foldq.core.quantum_code import QuantumCode, PauliOperator, Stabilizer
from foldq.codes.gf2 import css_check

HAL_PCM_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'HAL', 'hal', 'codes', 'qTanner_pcms'))

DEFAULT_HAL_CODES = ['D4_3', 'D6_3', 'D8_3', 'D8_5', 'D10_5']


def _read_pcm(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        rows = [list(map(int, line.split())) for line in f if line.strip()]
    return np.array(rows, dtype=np.uint8)


def code_from_pcm(Hx: np.ndarray, Hz: np.ndarray) -> QuantumCode:
    """由 (Hx, Hz) 构造 QuantumCode；数据比特 0..n-1，辅助比特按 X、Z 顺序排在后面"""
    commute, k = css_check(Hx, Hz)
    if not commute:
        raise ValueError("Hx·Hz^T != 0，稳定子不对易，不是有效 CSS 码")

    n = Hx.shape[1]
    num_physical = n + Hx.shape[0] + Hz.shape[0]
    code = QuantumCode(num_physical, k, distance=0)  # 码距未知，按需单独计算
    code.num_data = n

    ancilla = n
    for H, stype in ((Hx, 'X'), (Hz, 'Z')):
        for row in H:
            ops = ['I'] * num_physical
            for q in np.nonzero(row)[0]:
                ops[q] = stype
            code.add_stabilizer(Stabilizer(PauliOperator(ops), ancilla, stype))
            ancilla += 1
    return code


def load_hal_code(name: str) -> QuantumCode:
    Hx = _read_pcm(os.path.join(HAL_PCM_DIR, f"{name}_Xstabs"))
    Hz = _read_pcm(os.path.join(HAL_PCM_DIR, f"{name}_Zstabs"))
    code = code_from_pcm(Hx, Hz)
    code.name = name
    return code


def load_hal_codes(names=DEFAULT_HAL_CODES) -> Dict[str, QuantumCode]:
    return {name: load_hal_code(name) for name in names}
