"""
实验28b：推荐工作点（k=10%，L_max=6 候选池）的逻辑错误率复测

问题：exp28 证明 L6 候选池保留平均全部 SWAP 增益（大码约九成）且 100% 布通；
本实验确认同一工作点上的逻辑错误率收益仍在（BP-OSD，与 exp25 同协议）。

协议（与 exp25 完全一致，仅跨层边集合不同）：
- 24 码（n≤200），spring 种子 0；基线 = 平面方格；
  FoldQ = 平面 + marginal 从 L≤6 候选池选出的 k=10% 边
- p* 复用 exp25 标定值；alpha=1.5；3 轮存储实验；每条件 3000 shots
- Clopper-Pearson 95% CI；双比例检验 + Holm 校正（24 检验族）

L6 候选池下 exp28 实测布通率 100/100，故布通边集 = 选出边集
（若某码 n_routed < n_cross 应剔除未布通的边——记录在 meta 中校验）。

用法：
  python exp28b_ler_workpoint.py --workers 3
  python exp28b_ler_workpoint.py --summarize-only
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

K_PCT, L_MAX, ALPHA = 0.10, 6, 1.5
PM_RATIO = 0.5
N_ROUNDS = 3
SHOTS = 3000
BP_MAX_ITER, OSD_ORDER = 100, 4

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp28b')
EXP25_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp25', 'tasks.jsonl')
EXP28_PARTIAL = os.path.join(os.path.dirname(__file__), 'results', 'exp28', 'results_partial.jsonl')
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/codes/gf2.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp28b_ler_workpoint.py']

_cache = {}


def _seed(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _setup(name):
    """码 + spring 布局 + L6 候选池 marginal 边（k=10%），与 exp28 同源"""
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
        _cache[name] = {'code': code, 'inter': inter, 'phys': phys, 'size': size,
                        'g0': g0, 'g': CouplingGraph(size, cross), 'n_cross': len(cross)}
    return _cache[name]


def task_ler(args):
    name, cond, p, alpha, k_pct = args
    from foldq.simulation.memory_circuit import route_rounds, build_memory_circuit
    from foldq.simulation.bposd import logical_error_rate
    t0 = time.time()
    st = _setup(name)
    graph = st['g0'] if k_pct == 0 else st['g']
    rounds = route_rounds(st['inter'], st['phys'], graph, N_ROUNDS)
    c = build_memory_circuit(st['code'], rounds, p, alpha, p * PM_RATIO)
    r = logical_error_rate(c, SHOTS, seed=_seed(name, 'e28b', cond),
                           max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    r.update({'code': name, 'cond': cond, 'p': p, 'seconds': time.time() - t0})
    return r


def _two_prop_p(k1, n1, k2, n2):
    pbar = (k1 + k2) / (n1 + n2)
    if pbar in (0, 1):
        return 1.0
    z = (k1 / n1 - k2 / n2) / math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    return float(2 * (1 - norm.cdf(abs(z))))


def _holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, mx = [0.0] * m, 0.0
    for r, i in enumerate(order):
        mx = max(mx, (m - r) * pvals[i])
        adj[i] = min(1.0, mx)
    return adj


def summarize(records, out_dir):
    ler = {}
    for r in records:
        ler.setdefault(r['code'], {})[r['cond']] = r
    cal = {}
    with open(EXP25_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip() and json.loads(line).get('stage') == 'cal':
                cal[json.loads(line)['code']] = json.loads(line)
    names = sorted((c for c in ler if 'base' in ler[c] and 'foldq_l6' in ler[c]),
                   key=lambda c: cal[c]['num_physical'])
    raw = [(c, _two_prop_p(ler[c]['base']['failures'], ler[c]['base']['shots'],
                           ler[c]['foldq_l6']['failures'], ler[c]['foldq_l6']['shots']))
           for c in names]
    holm = _holm([p for _, p in raw])
    e25 = {}
    with open(EXP25_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('stage') == 'ler':
                    e25.setdefault(r['code'], {})[r['cond']] = r
    L = ['# 实验28b：推荐工作点（k=10%，L_max=6）逻辑错误率复测', '',
         f'与 exp25 同协议（p* 沿用其标定，alpha={ALPHA}，{SHOTS} shots，{N_ROUNDS} 轮），'
         '差异仅在跨层边来自 L≤6 候选池。Holm 校正（24 检验族）。', '',
         '| 码 | n | 基线 LER [95% CI] | FoldQ-L6 LER [95% CI] | 改善 | p_holm | '
         'exp25(不限长)改善 | 差 |', '|---|---|---|---|---|---|---|---|']
    n_pos = 0
    for (c, p), h in zip(raw, holm):
        b, f2 = ler[c]['base'], ler[c]['foldq_l6']
        imp = 1 - f2['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
        eb, ef = e25.get(c, {}).get('base'), e25.get(c, {}).get('alpha_1.5')
        e25_imp = (1 - ef['ler'] / eb['ler']) if eb and ef and eb['ler'] > 0 else None
        mark = '*' if h < 0.05 else ''
        n_pos += 1 if (h < 0.05 and imp > 0) else 0
        e25_txt = f"{e25_imp*100:+.0f}%" if e25_imp is not None else '—'
        d_txt = f"{(imp - e25_imp) * 100:+.0f}pp" if e25_imp is not None else '—'
        L.append(f"| {c} | {cal[c]['num_physical']} | "
                 f"{b['ler']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | "
                 f"{f2['ler']:.4f} [{f2['ci95'][0]:.4f}, {f2['ci95'][1]:.4f}] | "
                 f"{imp*100:+.0f}%{mark} | {h:.1e} | {e25_txt} | {d_txt} |")
    L += ['', f'Holm 校正后显著改善 {n_pos} / {len(raw)}。']
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(out_dir, 'summary.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'tasks.jsonl')

    if a.summarize_only:
        summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)
        return

    from foldq.codes.hal_all_codes import load_all_hal_codes
    cal = {}
    with open(EXP25_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip() and json.loads(line).get('stage') == 'cal':
                cal[json.loads(line)['code']] = json.loads(line)
    # exp28 校验：L6/k10% 下布通率应为 100%
    routed_ok = {}
    with open(EXP28_PARTIAL, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['k_pct'] == K_PCT and r['l_max'] == f'L{L_MAX}' and 'error' not in r:
                routed_ok[r['code']] = (r['n_routed'] == r['n_cross'])
    bad = [c for c, okk in routed_ok.items() if not okk]
    if bad:
        print(f'警告：{bad} 在 L6/k10% 下布通率 <100%，结果应剔除未布通边（当前直接用选出边）')

    done = {(json.loads(l)['code'], json.loads(l)['cond'])
            for l in open(path, encoding='utf-8') if l.strip()} if os.path.exists(path) else set()
    codes = sorted((c for c in load_all_hal_codes() if c.num_physical <= 200),
                   key=lambda c: -c.num_physical)
    tasks = []
    for c in codes:
        p = cal[c.name]['p_star']
        for cond, alpha, k_pct in (('base', 1.0, 0), ('foldq_l6', ALPHA, K_PCT)):
            if (c.name, cond) not in done:
                tasks.append((c.name, cond, p, alpha, k_pct))
    print(f'LER 复测：{len(tasks)} 个任务，{a.workers} 进程', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f"  [{i}/{len(tasks)}] {rec['code']:14s} {rec['cond']:10s} "
                  f"ler={rec['ler']:.4f}  {rec['seconds']:.0f}s  累计 {time.time()-t0:.0f}s",
                  flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
