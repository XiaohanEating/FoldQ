"""
GF(2) 线性代数工具：用于验证 CSS 码的有效性

- css_check: 验证 Hx·Hz^T = 0（稳定子对易），并计算逻辑比特数 k = n - rank(Hx) - rank(Hz)
- min_logical_weight: 小规模穷举最小权重逻辑算符（码距上界/精确值）
"""

from itertools import combinations
from typing import Optional, Tuple

import numpy as np


def _echelon(H: np.ndarray) -> Tuple[np.ndarray, list]:
    """GF(2) 行阶梯化，返回 (阶梯矩阵, 主元列表)"""
    M = (np.array(H, dtype=np.uint8) % 2).copy()
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        M[others] ^= M[r]
        pivots.append(c)
        r += 1
    return M[:r], pivots


def gf2_rank(H: np.ndarray) -> int:
    if H.size == 0:
        return 0
    return len(_echelon(H)[1])


def css_check(Hx: np.ndarray, Hz: np.ndarray) -> Tuple[bool, int]:
    """
    返回 (稳定子是否全部对易, 逻辑比特数 k)
    """
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError("Hx 与 Hz 的列数（数据比特数）不一致")
    commute = not np.any((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2)
    n = Hx.shape[1]
    k = n - gf2_rank(Hx) - gf2_rank(Hz)
    return bool(commute), int(k)


def _in_rowspace(echelon: np.ndarray, pivots: list, v: np.ndarray) -> bool:
    w = v.copy()
    for row, c in zip(echelon, pivots):
        if w[c]:
            w ^= row
    return not w.any()


def gf2_nullspace(H: np.ndarray) -> np.ndarray:
    """GF(2) 零空间的一组基，每行一个向量"""
    n = H.shape[1]
    ech, piv = _echelon(H)
    free = [c for c in range(n) if c not in set(piv)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for row, c in zip(ech, piv):
            basis[i, c] = row[f]
    return basis


def logical_basis(H_check: np.ndarray, H_stab: np.ndarray) -> np.ndarray:
    """
    逻辑算符的一组代表：位于 ker(H_check)、且模 rowspace(H_stab) 线性无关。
    逻辑 Z 取 (H_check, H_stab) = (Hx, Hz)，逻辑 X 取 (Hz, Hx)。返回 k 行。
    """
    span = (np.array(H_stab, dtype=np.uint8) % 2)
    chosen = []
    for v in gf2_nullspace(H_check):
        ech, piv = _echelon(span)
        if not _in_rowspace(ech, piv, v):
            chosen.append(v)
            span = np.vstack([span, v])
    return np.array(chosen, dtype=np.uint8).reshape(len(chosen), H_check.shape[1])


def _min_weight_one_side(H_check: np.ndarray, H_stab: np.ndarray,
                         max_weight: int, max_combos: int) -> Optional[int]:
    """
    找最小权重向量 v：H_check·v = 0 且 v 不在 rowspace(H_stab) 中
    （即一个非平凡逻辑算符）。超过 max_weight 或组合数上限仍未找到则返回 None。
    """
    n = H_check.shape[1]
    cols = (H_check.T % 2).astype(np.uint8)
    ech, piv = _echelon(H_stab)
    for w in range(1, max_weight + 1):
        count = 0
        for support in combinations(range(n), w):
            count += 1
            if count > max_combos:
                return None
            syndrome = np.bitwise_xor.reduce(cols[list(support)], axis=0)
            if syndrome.any():
                continue
            v = np.zeros(n, dtype=np.uint8)
            v[list(support)] = 1
            if not _in_rowspace(ech, piv, v):
                return w
    return None


def min_logical_weight(Hx: np.ndarray, Hz: np.ndarray, max_weight: int = 4,
                       max_combos: int = 2_000_000) -> Optional[int]:
    """
    穷举两类逻辑算符（Z 型：ker Hx 中不属于 rowspace Hz 的；X 型：反之），
    返回最小权重。返回 None 表示在 max_weight 以内没找到（码距 > max_weight，
    或组合数超限）。只用于小规模码的正确性检查。
    """
    dz = _min_weight_one_side(Hx, Hz, max_weight, max_combos)
    dx = _min_weight_one_side(Hz, Hx, max_weight, max_combos)
    found = [d for d in (dz, dx) if d is not None]
    return min(found) if found else None
