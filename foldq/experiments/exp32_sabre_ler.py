"""
实验32（复核 P1-2）：SABRE 路由电路上的逻辑错误率对照

问题：全部 LER 实验（exp25/28b/29/30/31b）建立在自研贪心路由器构建的电路上；
SABRE（SOTA 路由器）的增益保持只在 SWAP 层验证过（exp12/15）。本实验在 SABRE
路由的电路族上直接复测 LER，确认收益不是贪心路由器的伪影。

协议（与 exp28b 完全一致，仅电路的操作序列来自 SABRE）：
- 4 码（D4_3/D6_3/D8_3/GT_98_6_12），spring 种子 0，工作点 k=10%、L6 候选池
- p* 沿用 exp25 标定；α=1.5；3 轮；每条件 3000 shots；CP 区间 + 双比例检验
- SABRE：qiskit SabreSwap（seed 0，trials 20），耦合图双向边（无 H 桥），
  初始布局 = 当前轮起始布局（与贪心版同一物理布局）
- 对照组：贪心电路结果直接引用 exp28b

用法：
  python exp32_sabre_ler.py --workers 2
  python exp32_sabre_ler.py --summarize-only
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.stats import norm

CODES = ['D4_3', 'D6_3', 'D8_3', 'GT_98_6_12']
K_PCT, L_MAX, ALPHA = 0.10, 6, 1.5
PM_RATIO = 0.5
N_ROUNDS, SHOTS = 3, 3000
BP_MAX_ITER, OSD_ORDER = 100, 4

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp32')
EXP25_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp25', 'tasks.jsonl')
EXP28B_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp28b', 'tasks.jsonl')
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp32_sabre_ler.py']
_cache = {}


def _seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _setup(name):
    if name not in _cache:
        from foldq.codes.hal_all_codes import load_code_by_name
        from foldq.optimization.layouts import spring_layout, default_grid_size
        from foldq.optimization.crosslayer_select import candidate_pairs, select_marginal
        from foldq.routing import CouplingGraph, positions_to_phys
        code = load_code_by_name(name)
        inter = code.get_all_interactions()
        size = default_grid_size(code)
        pos = spring_layout(code, size, 0)
        phys = positions_to_phys(pos, size)
        g0 = CouplingGraph(size)
        cands = candidate_pairs(inter, phys, g0.dist, size=size, max_len=L_MAX)
        cross = select_marginal(cands, g0.dist, inter, phys,
                                max(1, round(K_PCT * code.num_physical))) if cands else []
        stype = {s.ancilla_qubit: s.stabilizer_type for s in code.stabilizers}
        _cache[name] = {'code': code, 'inter': inter, 'phys': phys, 'size': size,
                        'g0': g0, 'g': CouplingGraph(size, cross), 'stype': stype}
    return _cache[name]


def sabre_route_round(inter, phys, graph, stype, seed=0):
    """SABRE 路由一轮。返回 (ops, final_phys)，ops 与 route_round(return_ops) 同格式：
    ('CX'|'SWAP', q1, q2, is_cross)，CX 的 q1 = 承载辅助逻辑比特的位置、
    q2 = 承载数据逻辑比特的位置（构建器按 stype[辅助] 定门方向）。

    SABRE 输入电路的 CX 方向按构建器规则给出（Z 型数据控制、X 型辅助控制），
    双向耦合图使 SABRE 不插 H 桥、保留输入方向。"""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import SabreSwap
    from qiskit.converters import circuit_to_dag
    edges = set()
    for a, b in graph.planar_edges + graph.cross_edges:
        edges.add((a, b))
        edges.add((b, a))
    cmap = CouplingMap(list(edges))
    qc = QuantumCircuit(graph.n)
    for anc, data in inter:
        ctrl, tgt = (phys[data], phys[anc]) if stype[anc] == 'Z' else (phys[anc], phys[data])
        qc.cx(ctrl, tgt)
    dag = circuit_to_dag(qc)
    routed = SabreSwap(cmap, 'decay', seed=seed, trials=20).run(dag)

    ops = []
    phys_of = dict(phys)
    logical_at = {p: q for q, p in phys_of.items()}
    cross_set = graph.cross_set
    for inst in routed.op_nodes():
        qs = tuple(routed.find_bit(q).index for q in inst.qargs)
        if inst.name == 'swap':
            a, b = qs
            ops.append(('SWAP', a, b, (min(a, b), max(a, b)) in cross_set))
            u, v = logical_at.get(a), logical_at.get(b)
            logical_at[a], logical_at[b] = v, u
            if u is not None:
                phys_of[u] = b
            if v is not None:
                phys_of[v] = a
        elif inst.name == 'cx':
            a, b = qs
            la, lb = logical_at.get(a), logical_at.get(b)
            # 转回复用构建器的 (辅助位置, 数据位置) 顺序
            q_anc, q_data = (a, b) if stype.get(la) else (b, a)
            ops.append(('CX', q_anc, q_data, (min(a, b), max(a, b)) in cross_set))
        else:
            raise ValueError(f'SABRE 输出含未预期指令 {inst.name}（应为双向耦合图无 H 桥）')
    return ops, phys_of


def task_ler(args):
    name, cond = args
    from foldq.simulation.memory_circuit import build_memory_circuit
    from foldq.simulation.bposd import logical_error_rate
    t0 = time.time()
    st = _setup(name)
    p = _pstar(name)
    graph = st['g'] if cond == 'sabre_foldq' else st['g0']
    rounds = []
    phys = dict(st['phys'])
    for _ in range(N_ROUNDS):
        ops, final = sabre_route_round(st['inter'], phys, graph, st['stype'])
        rounds.append((ops, phys))
        phys = final
    c = build_memory_circuit(st['code'], rounds, p, ALPHA, p * PM_RATIO)
    r = logical_error_rate(c, SHOTS, seed=_seed(name, 'e32', cond),
                           max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    r.update({'code': name, 'cond': cond, 'p': p, 'router': 'sabre',
              'seconds': time.time() - t0})
    return r


_pstars = {}


def _pstar(name):
    if not _pstars:
        with open(EXP25_TASKS, encoding='utf-8') as f:
            for line in f:
                if line.strip() and json.loads(line).get('stage') == 'cal':
                    _pstars[json.loads(line)['code']] = json.loads(line)['p_star']
    return _pstars[name]


def _two_prop_p(k1, n1, k2, n2):
    pbar = (k1 + k2) / (n1 + n2)
    if pbar in (0, 1):
        return 1.0
    z = (k1 / n1 - k2 / n2) / math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    return float(2 * (1 - norm.cdf(abs(z))))


def summarize(records, out_dir):
    ler = {}
    for r in records:
        ler.setdefault(r['code'], {})[r['cond']] = r
    e28 = {}
    with open(EXP28B_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                e28.setdefault(r['code'], {})[r['cond']] = r
    L = ['# 实验32：SABRE 路由电路上的 LER 对照（复核 P1-2）', '',
         '协议同 exp28b（p*、α=1.5、3000 shots），操作序列来自 qiskit SabreSwap',
         '（seed 0、trials 20、双向耦合图）；对照列为贪心电路的 exp28b 结果。', '',
         '| 码 | SABRE基线 LER | SABRE+FoldQ LER | SABRE改善 | p 值 | 贪心改善(exp28b) | 差 |',
         '|---|---|---|---|---|---|---|']
    for c in CODES:
        b, f2 = ler.get(c, {}).get('sabre_base'), ler.get(c, {}).get('sabre_foldq')
        if not (b and f2):
            continue
        imp = 1 - f2['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
        pv = _two_prop_p(b['failures'], b['shots'], f2['failures'], f2['shots'])
        g = e28.get(c, {})
        greedy = (1 - g['foldq_l6']['ler'] / g['base']['ler']
                  if 'base' in g and 'foldq_l6' in g else None)
        mark = '*' if pv < 0.05 else ''
        gtxt = f"{greedy*100:+.0f}%" if greedy is not None else '—'
        dtxt = f"{(imp-greedy)*100:+.0f}pp" if greedy is not None else '—'
        L.append(f"| {c} | {b['ler']:.4f} | {f2['ler']:.4f} | {imp*100:+.0f}%{mark} | "
                 f"{pv:.1e} | {gtxt} | {dtxt} |")
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(out_dir, 'summary.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'tasks.jsonl')
    if a.summarize_only:
        summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)
        return
    meta = {'codes': CODES, 'k_pct': K_PCT, 'l_max': L_MAX, 'alpha': ALPHA,
            'sabre': 'qiskit SabreSwap decay seed=0 trials=20 双向耦合图',
            'source_sha256_12': {s: hashlib.sha256(open(
                os.path.join(os.path.dirname(__file__), '..', '..', s), 'rb')
                .read()).hexdigest()[:12] for s in SOURCES},
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = {(json.loads(l)['code'], json.loads(l)['cond']) for l in f if l.strip()}
    tasks = [(c, cond) for c in sorted(CODES)
             for cond in ('sabre_base', 'sabre_foldq') if (c, cond) not in done]
    print(f'exp32：{len(tasks)} 任务，{a.workers} 进程', flush=True)
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        t0 = time.time()
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f"  [{i}/{len(tasks)}] {rec['code']:12s} {rec['cond']:12s} "
                  f"ler={rec['ler']:.4f} {rec['seconds']:.0f}s 累计 {time.time()-t0:.0f}s",
                  flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
