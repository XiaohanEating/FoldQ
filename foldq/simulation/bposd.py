"""
BP-OSD 解码：从 Stim 的探测器错误模型构造校验矩阵并解码

qLDPC 码的单个电路错误常同时触发多于 2 个探测器，pymatching 处理不了，
这里用 ldpc 包的 BpOsdDecoder。相同探测器/观测量组合的错误机制合并为一列，
概率按 p = p1(1-p2) + p2(1-p1) 合成。
"""

from typing import Dict, Tuple

import numpy as np
import stim
from scipy.sparse import csc_matrix
from scipy.stats import beta

from ldpc import BpOsdDecoder


def dem_matrices(dem: stim.DetectorErrorModel) -> Tuple[csc_matrix, csc_matrix, np.ndarray]:
    """返回 (H 探测器×错误, L 观测量×错误, 先验概率)"""
    cols: Dict[Tuple[tuple, tuple], float] = {}
    for inst in dem.flattened():
        if inst.type != 'error':
            continue
        p = inst.args_copy()[0]
        dets = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        key = (dets, obs)
        q = cols.get(key, 0.0)
        cols[key] = q * (1 - p) + p * (1 - q)

    keys = list(cols)
    def build(idx, n_rows):
        rows, cs = [], []
        for j, k in enumerate(keys):
            for i in k[idx]:
                rows.append(i)
                cs.append(j)
        return csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cs)),
                          shape=(n_rows, len(keys)))
    H = build(0, dem.num_detectors)
    L = build(1, dem.num_observables)
    return H, L, np.array([cols[k] for k in keys])


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


def logical_error_rate(circuit: stim.Circuit, shots: int, seed: int = 0,
                       max_iter: int = 100, osd_order: int = 4) -> Dict:
    """
    采样 shots 次，BP-OSD 解码，任一逻辑观测量预测错即计一次失败。
    相同综合征只解码一次。
    """
    dem = circuit.detector_error_model()
    H, L, priors = dem_matrices(dem)
    dec = BpOsdDecoder(H, error_channel=list(priors), max_iter=max_iter,
                       bp_method='minimum_sum', osd_method='osd_cs', osd_order=osd_order)

    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    uniq, inv = np.unique(det.astype(np.uint8), axis=0, return_inverse=True)
    pred = np.zeros((len(uniq), L.shape[0]), dtype=np.uint8)
    Ld = L.toarray()
    for i, s in enumerate(uniq):
        if s.any():
            e = dec.decode(s)
            pred[i] = (Ld @ e) % 2
    fail = np.any(pred[inv.ravel()] != obs, axis=1)
    k = int(fail.sum())
    lo, hi = clopper_pearson(k, shots)
    return {'failures': k, 'shots': shots, 'ler': k / shots, 'ci95': [lo, hi],
            'n_detectors': dem.num_detectors, 'n_error_mechanisms': H.shape[1],
            'n_unique_syndromes': int(len(uniq))}
