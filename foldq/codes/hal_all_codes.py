"""
HAL 全量 qLDPC 码加载器

支持的码类型：
1. Generalized Toric Codes（广义环面码）：从 gt_params.csv 读取参数
2. qTanner codes（qLDPC 基础码）：D4_3, D6_3, D8_3, D8_5, D10_5
3. Tile codes（待实现）
4. Radial codes（待实现）
"""

import csv
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import sympy

from foldq.codes.hal_loader import load_hal_code, HAL_PCM_DIR
from foldq.core.quantum_code import QuantumCode, PauliOperator, Stabilizer

HAL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'HAL'))
# 如果 HAL 在当前工作目录旁边，使用绝对路径
if not os.path.exists(os.path.join(HAL_ROOT, 'hal')):
    HAL_ROOT = 'C:/work/FoldQ/HAL'


def parse_gt_params_csv() -> List[Dict]:
    """解析 gt_params.csv，返回所有 generalized toric code 的参数"""
    csv_path = os.path.join(HAL_ROOT, 'hal', 'codes', 'gt_params.csv')

    codes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 解析 [[n, k, d]] 格式
            nkd_str = row['[[n, k, d]]']
            match = re.match(r'\[\[(\d+),\s*(\d+),\s*(.+)\]\]', nkd_str)
            if not match:
                continue

            n, k, d_str = match.groups()
            n, k = int(n), int(k)

            # 码距可能是 "≤ 22" 格式
            if '≤' in d_str:
                d = int(d_str.split('≤')[1].strip())
                d_exact = False
            else:
                d = int(d_str)
                d_exact = True

            # 解析晶格向量 a1, a2（格式如 "(0, 3)"）
            a1 = eval(row['a1'])  # tuple
            a2 = eval(row['a2'])  # tuple

            # 解析多项式指数
            f_x = int(row['f_x_exp'])
            f_y = int(row['f_y_exp'])
            g_x = int(row['g_x_exp'])
            g_y = int(row['g_y_exp'])

            codes.append({
                'n': n,
                'k': k,
                'd': d,
                'd_exact': d_exact,
                'a1': a1,
                'a2': a2,
                'f_x': f_x,
                'f_y': f_y,
                'g_x': g_x,
                'g_y': g_y,
                'name': f"GT_{n}_{k}_{d}",
            })

    # CSV 中有同参数 [[n,k,d]] 的不同码，后出现的加 _v2、_v3 区分
    seen = {}
    for p in codes:
        seen[p['name']] = seen.get(p['name'], 0) + 1
        if seen[p['name']] > 1:
            p['name'] = f"{p['name']}_v{seen[p['name']]}"
    return codes


def load_code_by_name(name: str) -> QuantumCode:
    """按名字加载单个 HAL 码（qTanner 或 GT），供并行工作进程使用"""
    if name.startswith('GT_'):
        for p in parse_gt_params_csv():
            if p['name'] == name:
                return load_generalized_toric_code(p)
        raise KeyError(name)
    return load_hal_code(name)


def load_generalized_toric_code(params: Dict) -> QuantumCode:
    """
    从参数字典加载 generalized toric code

    需要 HAL 的 GeneralizedToricCode 类
    """
    try:
        from hal.codes.generalized_toric_codes import GeneralizedToricCode
    except ImportError:
        raise ImportError("需要安装 HAL 包才能加载 generalized toric codes")

    # 用 HAL 自带的论文参数构造：f = 1 + x + x^a y^b，g = 1 + y + x^c y^d，
    # 环面阶数取 alpha = a1[1]、beta = a2[0]（旧版只用单项式且阶数取错，得到 k=0 的错误码）
    hal_code = GeneralizedToricCode.from_paper_parameters(
        a=params['f_x'], b=params['f_y'], c=params['g_x'], d=params['g_y'],
        lattice_vectors=(params['a1'], params['a2']),
    )

    # 转换为 FoldQ 的 QuantumCode 格式
    Hx = np.array(hal_code.matrix_x if hasattr(hal_code, 'matrix_x') else hal_code.Hx) % 2
    Hz = np.array(hal_code.matrix_z if hasattr(hal_code, 'matrix_z') else hal_code.Hz) % 2

    from foldq.codes.hal_loader import code_from_pcm
    code = code_from_pcm(Hx.astype(np.uint8), Hz.astype(np.uint8))
    if code.num_logical != params['k']:
        raise ValueError(f"{params['name']}: 构造得到 k={code.num_logical}，与 CSV 的 k={params['k']} 不符")
    code.name = params['name']
    code.distance = params['d']

    return code


def load_all_qtanner_codes() -> List[QuantumCode]:
    """加载所有 qTanner 码（D4_3, D6_3, D8_3, D8_5, D10_5）"""
    names = ['D4_3', 'D6_3', 'D8_3', 'D8_5', 'D10_5']
    return [load_hal_code(name) for name in names]


def load_all_gt_codes(max_n: int = None) -> List[QuantumCode]:
    """
    加载所有 generalized toric codes

    max_n: 如果指定，只加载 n ≤ max_n 的码（控制规模）
    """
    params_list = parse_gt_params_csv()

    if max_n is not None:
        params_list = [p for p in params_list if p['n'] <= max_n]

    codes = []
    for params in params_list:
        try:
            code = load_generalized_toric_code(params)
            codes.append(code)
        except Exception as e:
            print(f"警告：加载 {params['name']} 失败：{e}")

    return codes


def load_all_hal_codes(max_n: int = None) -> List[QuantumCode]:
    """
    加载所有 HAL qLDPC 码

    max_n: 如果指定，只加载 n ≤ max_n 的码（控制规模）
    """
    codes = []

    # qTanner codes（小码，稳定）
    codes.extend(load_all_qtanner_codes())

    # Generalized toric codes（大量，可选择性加载）
    codes.extend(load_all_gt_codes(max_n=max_n))

    return codes


def test_loader():
    """测试加载器"""
    print("测试 qTanner codes...")
    qtanner = load_all_qtanner_codes()
    for code in qtanner:
        print(f"  {code.name}: n={code.num_physical}, k={code.num_logical}, d={code.distance}")

    print("\n测试 GT codes (n ≤ 50)...")
    gt_codes = load_all_gt_codes(max_n=50)
    print(f"  加载了 {len(gt_codes)} 个 GT 码")
    for code in gt_codes[:5]:
        print(f"  {code.name}: n={code.num_physical}, k={code.num_logical}, d={code.distance}")

    print(f"\n总共 {len(qtanner) + len(gt_codes)} 个码")


if __name__ == '__main__':
    test_loader()
