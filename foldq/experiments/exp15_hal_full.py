"""
实验15：HAL 全量 qLDPC 码 SWAP 减少（完整对比）

目标：在所有 HAL qLDPC 码上验证跨层连接选择算法的优越性

协议：
- 码：所有 HAL qLDPC 码（qTanner 5个 + GT 码）
- 布局：spring（1 个种子，后续可扩展）
- 跨层连接预算：k = 10% 物理比特数
- 跨层连接选择算法：
  1. random（10 次取平均±标准差）
  2. longest（旧方法，按欧氏距离从长到短选）
  3. marginal（新方法，按边际 SWAP 减少量贪心选）
- 路由器：
  1. 贪心路由器（route_round）
  2. qiskit SABRE（3 次取平均±标准差）
- 基线：平面网格（无跨层连接）
- 指标：SWAP 数、SWAP 减少百分比、深度

输出格式：
每个码生成一行结果，包含所有配置的 SWAP 和深度
"""

import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np

from foldq.codes.hal_all_codes import load_all_hal_codes
from foldq.optimization.layouts import spring_layout, default_grid_size
from foldq.optimization.crosslayer_select import (
    candidate_pairs, select_random, select_longest, select_marginal
)
from foldq.routing import CouplingGraph, positions_to_phys, route_round

LAYOUT_SEED = 0
CROSS_BUDGET_PCT = 0.10  # 10%
RANDOM_TRIALS = 10
SABRE_TRIALS = 3
MAX_CODE_SIZE = None  # None = 全部，可设置如 100 限制规模
TEST_MODE = False  # 测试模式：只跑前 3 个码

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp15')
SOURCES = [
    'foldq/routing/router.py',
    'foldq/optimization/crosslayer_select.py',
    'foldq/codes/hal_all_codes.py',
    'foldq/experiments/exp15_hal_full.py'
]


def source_hashes():
    out = {}
    for rel in SOURCES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()[:12]
    return out


def budget_k(n_phys: int, pct: float = CROSS_BUDGET_PCT) -> int:
    return max(1, round(pct * n_phys))


def route_with_greedy(inter, phys, graph) -> Dict:
    """贪心路由器"""
    return route_round(inter, phys, graph)


def route_with_sabre(inter, phys, graph, n_trials: int = SABRE_TRIALS) -> Dict:
    """SABRE 路由器（多次取平均）"""
    try:
        from qiskit.transpiler import CouplingMap
        from qiskit import QuantumCircuit
        from qiskit.transpiler.passes import SabreSwap
        from qiskit.converters import circuit_to_dag
    except ImportError:
        return {'swaps': -1, 'depth': -1, 'error': 'qiskit not installed'}

    # 构造 qiskit 的 CouplingMap
    # CouplingGraph 的边 = planar_edges + cross_edges
    edges = graph.planar_edges + graph.cross_edges
    if not edges:
        return {'swaps': 0, 'depth': 0}

    coupling_map = CouplingMap(edges)

    # 构造交互电路（每个交互一个 CX），比特按布局 phys 放到物理位置上，
    # 使 SABRE 与贪心路由从同一个初始布局出发（旧版用逻辑编号，等于平凡布局，已修正）
    qc = QuantumCircuit(graph.n)
    for a, b in inter:
        qc.cx(phys[a], phys[b])

    # 多次运行 SABRE
    results = []
    for seed in range(n_trials):
        dag = circuit_to_dag(qc)
        sabre = SabreSwap(coupling_map, 'decay', seed=seed, trials=20)
        dag_routed = sabre.run(dag)

        # 统计 SWAP 数
        swaps = sum(1 for node in dag_routed.op_nodes() if node.name == 'swap')
        depth = dag_routed.depth()
        results.append({'swaps': swaps, 'depth': depth})

    # 取平均和标准差
    swaps_avg = np.mean([r['swaps'] for r in results])
    swaps_std = np.std([r['swaps'] for r in results])
    depth_avg = np.mean([r['depth'] for r in results])
    depth_std = np.std([r['depth'] for r in results])

    return {
        'swaps': swaps_avg,
        'swaps_std': swaps_std,
        'depth': depth_avg,
        'depth_std': depth_std,
    }


def run_code(code) -> Dict:
    """
    对一个码跑完整的对比实验

    返回：
    {
        'code_name': str,
        'n': int,
        'k': int,
        'd': int,
        'cross_budget': int,
        'baseline_greedy': {swaps, depth},
        'baseline_sabre': {swaps, swaps_std, depth, depth_std},
        'random_greedy': {swaps_avg, swaps_std, depth_avg, depth_std},
        'longest_greedy': {swaps, depth},
        'marginal_greedy': {swaps, depth},
        'marginal_sabre': {swaps, swaps_std, depth, depth_std},
    }
    """
    print(f"\n{code.name} 开始...")
    print(f"  码参数: n={code.num_physical}, k={code.num_logical}, d={code.distance}")

    inter = code.get_all_interactions()
    size = default_grid_size(code)
    pos = spring_layout(code, size, LAYOUT_SEED)
    phys = positions_to_phys(pos, size)

    g0 = CouplingGraph(size)
    k = budget_k(code.num_physical)
    cands = candidate_pairs(inter, phys, g0.dist)

    result = {
        'code_name': code.name,
        'n': code.num_physical,
        'k': code.num_logical,
        'd': code.distance,
        'cross_budget': k,
        'n_candidates': len(cands),
    }

    # 1. 基线（平面网格）- 贪心路由器
    print(f"  1/6: 基线 + 贪心...")
    t = time.time()
    r = route_with_greedy(inter, phys, g0)
    result['baseline_greedy'] = {'swaps': r['swaps'], 'depth': r['depth']}
    print(f"    SWAP={r['swaps']}, 深度={r['depth']}, 用时 {time.time()-t:.2f}s")

    # 2. 基线（平面网格）- SABRE
    print(f"  2/6: 基线 + SABRE...")
    t = time.time()
    r = route_with_sabre(inter, phys, g0, SABRE_TRIALS)
    result['baseline_sabre'] = r
    if 'error' not in r:
        print(f"    SWAP={r['swaps']:.1f}±{r['swaps_std']:.1f}, 深度={r['depth']:.1f}±{r['depth_std']:.1f}, 用时 {time.time()-t:.2f}s")
    else:
        print(f"    错误: {r['error']}")

    # 3. random + 贪心（10 次）
    print(f"  3/6: random + 贪心 (x{RANDOM_TRIALS})...")
    t = time.time()
    random_results = []
    rng = np.random.default_rng()
    for trial in range(RANDOM_TRIALS):
        cross = select_random(cands, g0.dist, k, rng) if cands else []
        g1 = CouplingGraph(size, cross)
        r = route_with_greedy(inter, phys, g1)
        random_results.append({'swaps': r['swaps'], 'depth': r['depth']})

    result['random_greedy'] = {
        'swaps_avg': np.mean([r['swaps'] for r in random_results]),
        'swaps_std': np.std([r['swaps'] for r in random_results]),
        'depth_avg': np.mean([r['depth'] for r in random_results]),
        'depth_std': np.std([r['depth'] for r in random_results]),
    }
    print(f"    SWAP={result['random_greedy']['swaps_avg']:.1f}±{result['random_greedy']['swaps_std']:.1f}, "
          f"深度={result['random_greedy']['depth_avg']:.1f}±{result['random_greedy']['depth_std']:.1f}, "
          f"用时 {time.time()-t:.2f}s")

    # 4. longest + 贪心
    print(f"  4/6: longest + 贪心...")
    t = time.time()
    cross = select_longest(cands, g0.dist, k) if cands else []
    g1 = CouplingGraph(size, cross)
    r = route_with_greedy(inter, phys, g1)
    result['longest_greedy'] = {'swaps': r['swaps'], 'depth': r['depth']}
    print(f"    SWAP={r['swaps']}, 深度={r['depth']}, 用时 {time.time()-t:.2f}s")

    # 5. marginal + 贪心
    print(f"  5/6: marginal + 贪心...")
    t = time.time()
    cross = select_marginal(cands, g0.dist, inter, phys, k) if cands else []
    g1 = CouplingGraph(size, cross)
    r = route_with_greedy(inter, phys, g1)
    result['marginal_greedy'] = {'swaps': r['swaps'], 'depth': r['depth']}
    print(f"    SWAP={r['swaps']}, 深度={r['depth']}, 用时 {time.time()-t:.2f}s")

    # 6. marginal + SABRE
    print(f"  6/6: marginal + SABRE...")
    t = time.time()
    cross = select_marginal(cands, g0.dist, inter, phys, k) if cands else []
    g1 = CouplingGraph(size, cross)
    r = route_with_sabre(inter, phys, g1, SABRE_TRIALS)
    result['marginal_sabre'] = r
    if 'error' not in r:
        print(f"    SWAP={r['swaps']:.1f}±{r['swaps_std']:.1f}, 深度={r['depth']:.1f}±{r['depth_std']:.1f}, 用时 {time.time()-t:.2f}s")
    else:
        print(f"    错误: {r['error']}")

    return result


def summarize(results: List[Dict]) -> str:
    """生成 markdown 汇总"""
    lines = ["# 实验15：HAL 全量 qLDPC 码 SWAP 减少\n"]
    lines.append(f"码数量: {len(results)}\n")
    lines.append(f"跨层连接预算: {CROSS_BUDGET_PCT*100:.0f}% 物理比特数\n")
    lines.append("路由器: 贪心路由器、qiskit SABRE\n")
    lines.append("跨层连接选择: random (10次均值)、longest、marginal\n")

    lines.append("\n## 结果表格（贪心路由器）\n")
    lines.append("| 码 | n | k | d | 基线 SWAP | random SWAP | longest SWAP | marginal SWAP | marginal 优势 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in results:
        name = r['code_name']
        n, k_log, d = r['n'], r['k'], r['d']
        base = r['baseline_greedy']['swaps']
        rand = r['random_greedy']['swaps_avg']
        longest = r['longest_greedy']['swaps']
        marg = r['marginal_greedy']['swaps']

        if base > 0:
            marg_gain = (base - marg) / base * 100
        else:
            marg_gain = 0

        lines.append(
            f"| {name} | {n} | {k_log} | {d} | {base} | {rand:.0f} | {longest} | {marg} | {marg_gain:+.1f}% |"
        )

    lines.append("\n## 结果表格（SABRE 路由器）\n")
    lines.append("| 码 | 基线 SWAP | marginal SWAP | marginal 优势 |")
    lines.append("|---|---|---|---|")

    for r in results:
        name = r['code_name']
        if 'error' in r['baseline_sabre']:
            continue
        base = r['baseline_sabre']['swaps']
        marg = r['marginal_sabre']['swaps']

        if base > 0:
            marg_gain = (base - marg) / base * 100
        else:
            marg_gain = 0

        lines.append(
            f"| {name} | {base:.1f} | {marg:.1f} | {marg_gain:+.1f}% |"
        )

    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # 加载所有码
    print("加载 HAL 全量 qLDPC 码...")
    codes = load_all_hal_codes(max_n=MAX_CODE_SIZE)

    if TEST_MODE:
        print(f"测试模式：只跑前 3 个码")
        codes = codes[:3]

    print(f"共加载 {len(codes)} 个码\n")

    # 逐个跑实验
    results = []
    for i, code in enumerate(codes):
        print(f"进度: {i+1}/{len(codes)}")
        try:
            r = run_code(code)
            results.append(r)
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 保存结果
    output = {
        'source_hashes': source_hashes(),
        'config': {
            'layout_seed': LAYOUT_SEED,
            'cross_budget_pct': CROSS_BUDGET_PCT,
            'random_trials': RANDOM_TRIALS,
            'sabre_trials': SABRE_TRIALS,
            'max_code_size': MAX_CODE_SIZE,
        },
        'results': results,
    }

    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    summary_md = summarize(results)
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary_md)

    elapsed = time.time() - t0
    print(f"\n总用时 {elapsed:.1f}s，结果写入 {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    main()
