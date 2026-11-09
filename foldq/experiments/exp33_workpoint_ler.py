"""
实验33：布线可行工作点的逻辑错误率（k=2%/5%，真实布通边集）

本实验用独立 3D 几何验收通过的边集构建耦合图，复测 k=2%/5% 的 LER。
改善率使用 exp28b 的匹配平面基线；exp28b 的 k=10% 理想边集仅作另一设计点
的参照，其与当前工作点的差异同时包含预算变化和布线影响。

协议同 exp28b（无时间片电路、p* 用 exp25 标定、α=1.5、3000 shots、
CP 区间 + 双比例检验 + Holm）。每码边集在任务内现算：
L6 池 → marginal 选 k·N 条 → HAL 布线（修复后）→ 独立验收 → 布通边。

用法：
  python exp33_workpoint_ler.py --workers 3
  python exp33_workpoint_ler.py --summarize-only
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool

HAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'HAL'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HAL_ROOT)
sys.path.insert(0, os.path.join(HAL_ROOT, 'foldq', 'experiments'))

import numpy as np
from scipy.stats import norm

from exp27_hal_routing import _patch_hal, hal_route  # noqa: E402

K_PCTS = [0.02, 0.05]
L_MAX, ALPHA = 6, 1.5
PM_RATIO = 0.5
N_ROUNDS, SHOTS = 3, 3000
BP_MAX_ITER, OSD_ORDER = 100, 4

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp33')
EXP25_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp25', 'tasks.jsonl')
EXP28B_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp28b', 'tasks.jsonl')
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp33_workpoint_ler.py',
           'HAL/foldq/experiments/exp27_hal_routing.py',
           'HAL/hal/routing/straightline.py', 'HAL/hal/routing/astar.py',
           'HAL/hal/routing/routing_algo.py', 'HAL/hal/hal.py']
_cache = {}
_pstars = {}


def _seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _routed_edges(name, k_pct):
    """L6 池 marginal 选边 → HAL 布线 → 独立验收 → 返回布通边与验收明细"""
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
                            max(1, round(k_pct * code.num_physical))) if cands else []
    edge_res, bench, n_tiers, collisions = hal_route(name, size, pos, phys, cross,
                                                     None, 10, 10)
    ok = {k for k, v in edge_res.items() if v['routed']}
    routed = [e for e in cross if tuple(sorted(e)) in ok]
    col_detail = [{'e1': list(c['e1']), 'e2': list(c['e2']), 'tier': c['tier'],
                   'seg1': c.get('seg1'), 'seg2': c.get('seg2')} for c in collisions]
    return code, inter, size, phys, g0, routed, len(cross), col_detail


def task_ler(args):
    name, k_pct = args
    _patch_hal()   # spawn worker 不执行 main()，HAL 补丁必须在任务内幂等重打
    from foldq.simulation.memory_circuit import route_rounds, build_memory_circuit
    from foldq.simulation.bposd import logical_error_rate
    from foldq.routing import CouplingGraph
    t0 = time.time()
    p = _pstar(name)
    code, inter, size, phys, g0, routed, n_cross, col_detail = _routed_edges(name, k_pct)
    g = CouplingGraph(size, routed)
    rounds = route_rounds(inter, phys, g, N_ROUNDS)
    c = build_memory_circuit(code, rounds, p, ALPHA, p * PM_RATIO)
    r = logical_error_rate(c, SHOTS, seed=_seed(name, 'e33', k_pct),
                           max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    r.update({'code': name, 'cond': f'k{k_pct:g}_routed', 'k_pct': k_pct,
              'n_cross': n_cross, 'n_routed': len(routed), 'n_collisions': len(col_detail),
              'p': p, 'seconds': time.time() - t0,
              # 来源固定（复核 5.3）：接受边集与冲突明细随记录落盘
              'routed_edges': [list(e) for e in routed],
              'collision_detail': col_detail})
    return r


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
    return float(2 * norm.sf(abs(z)))


def _holm(pv):
    m = len(pv)
    order = sorted(range(m), key=lambda i: pv[i])
    adj, mx = [0.0] * m, 0.0
    for r, i in enumerate(order):
        mx = max(mx, (m - r) * pv[i])
        adj[i] = min(1.0, mx)
    return adj


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
    raw = []
    for c in ler:
        b = e28.get(c, {}).get('base')
        for kp in K_PCTS:
            f = ler[c].get(f'k{kp:g}_routed')
            if b and f:
                raw.append((c, kp, _two_prop_p(b['failures'], b['shots'],
                                               f['failures'], f['shots'])))
    holm = _holm([x[2] for x in raw])
    hmap = {(c, kp): h for (c, kp, _), h in zip(raw, holm)}
    L = ['# 实验33：布线可行工作点（k=2%/5%，真实布通边集）的 LER', '',
         '边集 = L6 池 marginal 选边后经 HAL 布线 + 独立验收通过的子集；'
         '协议同 exp28b；基线复用 exp28b 的平面基线。', '',
         '| 码 | n | k=2% 布通(边/碰撞) | 改善 | k=5% 布通 | 改善 | exp28b 理想(k=10%) |',
         '|---|---|---|---|---|---|---|']
    for c in sorted(ler, key=lambda x: -ler[x][list(ler[x])[0]]['p'] if ler[x] else 0):
        b = e28.get(c, {}).get('base')
        row = {'code': c}
        cells = []
        for kp in K_PCTS:
            f = ler[c].get(f'k{kp:g}_routed')
            if not f or not b:
                cells.append(('—', '—'))
                continue
            imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
            mark = '*' if hmap.get((c, kp), 1) < 0.05 else ''
            cells.append((f"{f['n_routed']}/{f['n_cross']}（{f['n_collisions']}碰）",
                          f"{imp*100:+.0f}%{mark}"))
        e28f = e28.get(c, {}).get('foldq_l6')
        e28imp = f"{(1 - e28f['ler'] / b['ler']) * 100:+.0f}%" if e28f and b and b['ler'] > 0 else '—'
        L.append(f"| {c} | {e28b_n(c)} | {cells[0][0]} | {cells[0][1]} | "
                 f"{cells[1][0]} | {cells[1][1]} | {e28imp} |")
    sig2 = sum(1 for (c, kp), h in hmap.items() if kp == 0.02 and h < 0.05)
    sig5 = sum(1 for (c, kp), h in hmap.items() if kp == 0.05 and h < 0.05)
    L += ['', f'Holm 后显著改善：k=2% {sig2}/{sum(1 for _, kp in hmap if kp == 0.02)}，'
          f'k=5% {sig5}/{sum(1 for _, kp in hmap if kp == 0.05)}。']
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(out_dir, 'summary.md')}")


def _e28b_n_cache():
    pass


_ns = {}


def e28b_n(code):
    if not _ns:
        with open(EXP25_TASKS, encoding='utf-8') as f:
            for line in f:
                if line.strip() and json.loads(line).get('stage') == 'cal':
                    _ns[json.loads(line)['code']] = json.loads(line)['num_physical']
    return _ns.get(code, '?')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    _patch_hal()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'tasks.jsonl')
    if a.summarize_only:
        summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)
        return
    from foldq.codes.hal_all_codes import load_all_hal_codes
    meta = {'k_pcts': K_PCTS, 'l_max': L_MAX, 'alpha': ALPHA,
            'routing': 'HAL + independent collision acceptance (P0 fix)',
            'source_sha256_12': {s: hashlib.sha256(open(
                os.path.join(os.path.dirname(__file__), '..', '..', s), 'rb')
                .read()).hexdigest()[:12] for s in SOURCES},
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = {(json.loads(l)['code'], json.loads(l)['k_pct'])
                    for l in f if l.strip()}
    codes = [c.name for c in sorted(load_all_hal_codes(), key=lambda c: -c.num_physical)
             if c.num_physical <= 200]
    tasks = [(c, kp) for c in codes for kp in K_PCTS if (c, kp) not in done]
    print(f'exp33：{len(tasks)} 任务，{a.workers} 进程', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False, default=int) + '\n')
            out.flush()
            print(f"  [{i}/{len(tasks)}] {rec['code']:12s} k={rec['k_pct']:.0%} "
                  f"布通{rec['n_routed']}/{rec['n_cross']} ler={rec['ler']:.4f} "
                  f"累计 {time.time()-t0:.0f}s", flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
