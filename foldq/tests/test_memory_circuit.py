"""
存储实验电路与 BP-OSD 解码的正确性测试

运行：python -m pytest foldq/tests/test_memory_circuit.py -q
"""

import numpy as np
import pymatching
import pytest
import stim

from foldq.core import SurfaceCode
from foldq.codes import load_hal_code, logical_basis, gf2_rank
from foldq.optimization.layouts import native_layout, spring_layout, default_grid_size
from foldq.routing import CouplingGraph, positions_to_phys
from foldq.simulation.memory_circuit import route_rounds, build_memory_circuit
from foldq.simulation.bposd import logical_error_rate


def _routed(code, layout='spring', n_rounds=2):
    if layout == 'native':
        pos, size = native_layout(code)
    else:
        size = default_grid_size(code)
        pos = spring_layout(code, size, 0)
    phys = positions_to_phys(pos, size)
    return route_rounds(code.get_all_interactions(), phys, CouplingGraph(size), n_rounds)


def _single_shot(circuit):
    det, obs = circuit.compile_detector_sampler().sample(1, separate_observables=True)
    return det[0].astype(int), obs[0].astype(int)


@pytest.mark.parametrize('make', [lambda: SurfaceCode(3), lambda: load_hal_code('D4_3'),
                                  lambda: load_hal_code('D6_3')])
def test_logical_basis_has_k_independent_operators(make):
    code = make()
    Hx, Hz = code.parity_check_matrices()
    Lz = logical_basis(Hx, Hz)
    assert len(Lz) == code.num_logical
    assert not ((Hx.astype(int) @ Lz.T.astype(int)) % 2).any()
    assert gf2_rank(np.vstack([Hz, Lz])) == gf2_rank(Hz) + code.num_logical


@pytest.mark.parametrize('x_det', [False, True])
@pytest.mark.parametrize('make,layout', [(lambda: SurfaceCode(3), 'native'),
                                         (lambda: load_hal_code('D4_3'), 'spring')])
def test_noiseless_detectors_and_observables_are_zero(make, layout, x_det):
    code = make()
    rounds = _routed(code, layout, n_rounds=2)
    c = build_memory_circuit(code, rounds, 0.0, 1.0, 0.0, x_detectors=x_det)
    det, obs = c.compile_detector_sampler().sample(200, separate_observables=True)
    assert not det.any() and not obs.any()


def test_data_x_error_fires_exactly_its_z_checks_in_first_round():
    code = load_hal_code('D4_3')
    rounds = _routed(code, 'spring', n_rounds=2)
    c0 = build_memory_circuit(code, rounds, 0.0, 1.0, 0.0)
    phys0 = rounds[0][1]
    z_stabs = [s for s in code.stabilizers if s.stabilizer_type == 'Z']
    for q in (0, 7, code.num_data - 1):
        c = stim.Circuit()
        c.append('X_ERROR', [phys0[q]], 1.0)
        c += c0
        det, obs = _single_shot(c)
        expected = np.zeros_like(det)
        for j, s in enumerate(z_stabs):  # 第 1 轮探测器按 Z 型辅助比特顺序排列
            expected[j] = int(q in s.data_qubits)
        assert expected.any()
        assert np.array_equal(det, expected)


def test_undetectable_logical_x_flips_observables_only():
    code = load_hal_code('D4_3')
    rounds = _routed(code, 'spring', n_rounds=2)
    c0 = build_memory_circuit(code, rounds, 0.0, 1.0, 0.0)
    Hx, Hz = code.parity_check_matrices()
    lx = logical_basis(Hz, Hx)[0]
    Lz = logical_basis(Hx, Hz)
    c = stim.Circuit()
    c.append('X_ERROR', [rounds[0][1][int(q)] for q in np.nonzero(lx)[0]], 1.0)
    c += c0
    det, obs = _single_shot(c)
    assert not det.any()
    assert np.array_equal(obs, (Lz.astype(int) @ lx.astype(int)) % 2)
    assert obs.any()


def test_bposd_agrees_with_pymatching_on_surface_code():
    code = SurfaceCode(3)
    rounds = _routed(code, 'native', n_rounds=3)
    c = build_memory_circuit(code, rounds, 3e-3, 1.0, 3e-3)
    shots = 20000
    bp = logical_error_rate(c, shots, seed=1)

    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    det, obs = c.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
    mw = np.any(m.decode_batch(det) != obs, axis=1).mean()

    lo, hi = bp['ci95']
    # 功效下限：失败数只需足够支撑下面的区间比较。不用严格 >50——
    # stim 小版本间的随机流差异会让固定种子的失败数在边界附近翻转。
    assert bp['failures'] >= 40
    assert lo * 0.8 <= mw <= hi * 1.25, (bp, mw)


# ---------------- exp29：带时间片的电路构建器 ----------------

def _routed_timed(code, layout='spring', n_rounds=2, beta=1.0):
    from foldq.simulation.memory_circuit import route_rounds_timed
    from foldq.routing import CouplingGraph, positions_to_phys
    from foldq.optimization.layouts import spring_layout, default_grid_size
    if layout == 'native':
        from foldq.optimization.layouts import native_layout
        pos, size = native_layout(code)
    else:
        size = default_grid_size(code)
        pos = spring_layout(code, size, 0)
    return route_rounds_timed(code.get_all_interactions(), positions_to_phys(pos, size),
                              CouplingGraph(size), n_rounds, cross_time_factor=beta)


def _dem_signature(circuit):
    """DEM 的语义签名：{(探测器集, 观测集): 合成概率} 的无序映射"""
    from foldq.simulation.bposd import dem_matrices
    dem = circuit.detector_error_model()
    H, L, priors = dem_matrices(dem)
    cols = {}
    Hd, Ld = H.toarray(), L.toarray()
    for j in range(H.shape[1]):
        key = (tuple(np.nonzero(Hd[:, j])[0]), tuple(np.nonzero(Ld[:, j])[0]))
        q = cols.get(key, 0.0)
        cols[key] = q * (1 - priors[j]) + priors[j] * (1 - q)
    return {k: round(v, 10) for k, v in cols.items()}


@pytest.mark.parametrize('make', [lambda: SurfaceCode(3), lambda: load_hal_code('D4_3')])
def test_timed_beta1_no_idle_matches_untimed_dem(make):
    """黄金校验：β=1、p_idle=0 时 timed 版与旧版的 DEM 语义完全一致"""
    from foldq.simulation.memory_circuit import (route_rounds, build_memory_circuit,
                                                 build_memory_circuit_timed)
    code = make()
    old_rounds = _routed(code, 'spring', n_rounds=3)
    timed_rounds = _routed_timed(code, 'spring', n_rounds=3, beta=1.0)
    c_old = build_memory_circuit(code, old_rounds, 1e-3, 1.5, 5e-4)
    c_new = build_memory_circuit_timed(code, timed_rounds, 1e-3, 1.5, 5e-4, p_idle=0.0)
    assert _dem_signature(c_old) == _dem_signature(c_new)


def test_timed_noiseless_zero_detectors_with_beta_and_idle():
    """调度展开（β=2、含测量时长）在无门噪声、p_idle=0 时不引入任何确定性错误"""
    from foldq.simulation.memory_circuit import build_memory_circuit_timed
    code = load_hal_code('D4_3')
    rounds = _routed_timed(code, 'spring', n_rounds=3, beta=2.0)
    c = build_memory_circuit_timed(code, rounds, 0.0, 2.0, 0.0, p_idle=0.0, t_meas=20)
    det, obs = c.compile_detector_sampler().sample(50, separate_observables=True)
    assert not det.any() and not obs.any()


def test_timed_idle_noise_adds_error_mechanisms():
    """p_idle>0 时空闲错误生效：同签名机制的概率上升（总错误权重变大）"""
    from foldq.simulation.memory_circuit import build_memory_circuit_timed

    def err_weight(c):
        return sum(i.args_copy()[0] for i in c.detector_error_model().flattened()
                   if i.type == 'error')

    code = load_hal_code('D4_3')
    rounds = _routed_timed(code, 'spring', n_rounds=3, beta=1.5)
    c0 = build_memory_circuit_timed(code, rounds, 1e-3, 1.5, 5e-4, p_idle=0.0)
    c1 = build_memory_circuit_timed(code, rounds, 1e-3, 1.5, 5e-4, p_idle=1e-4)
    assert err_weight(c1) > err_weight(c0) * 1.05
