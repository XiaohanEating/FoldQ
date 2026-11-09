"""
纠错码有效性测试：任何实验使用的码都必须先通过这里

运行：python -m pytest foldq/tests -q
"""

import pytest

from foldq.core import SurfaceCode
from foldq.codes import css_check, min_logical_weight, load_hal_codes


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_code_is_valid_css(d):
    code = SurfaceCode(d)
    Hx, Hz = code.parity_check_matrices()
    commute, k = css_check(Hx, Hz)
    assert commute
    assert k == 1
    assert len(code.stabilizers) == d * d - 1
    assert code.num_physical == 2 * d * d - 1
    ok, _ = code.verify_commutation()
    assert ok


@pytest.mark.parametrize("d", [3, 5])
def test_surface_code_distance(d):
    Hx, Hz = SurfaceCode(d).parity_check_matrices()
    assert min_logical_weight(Hx, Hz, max_weight=d) == d


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_code_native_layout_is_nearest_neighbour(d):
    code = SurfaceCode(d)
    pos = code.native_positions()
    size = code.native_grid_size()
    assert len(set(pos.values())) == code.num_physical
    assert all(0 <= r < size and 0 <= c < size for r, c in pos.values())
    for a, q in code.get_all_interactions():
        (r1, c1), (r2, c2) = pos[a], pos[q]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_hal_codes_are_valid_css():
    for name, code in load_hal_codes().items():
        Hx, Hz = code.parity_check_matrices()
        commute, k = css_check(Hx, Hz)
        assert commute, name
        assert k >= 1, name


def test_generalized_toric_codes_match_hal_parameters():
    """GT 码按 HAL 论文参数构造：k 与 CSV 一致，稳定子权重为 6"""
    from foldq.codes.hal_all_codes import parse_gt_params_csv, load_generalized_toric_code
    for p in [q for q in parse_gt_params_csv() if q['n'] <= 60]:
        code = load_generalized_toric_code(p)
        Hx, Hz = code.parity_check_matrices()
        commute, k = css_check(Hx, Hz)
        assert commute and k == p['k'] == code.num_logical
        assert Hx.shape[1] == p['n']
        assert set(Hx.sum(axis=1)) == {6} and set(Hz.sum(axis=1)) == {6}
