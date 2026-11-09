"""
实验29b（复核 5.4）：空闲噪声公式修正后的 exp29 关键结论复验

两项修正：
1. 合并公式：1-(1-p)^t → (3/4)[1-(1-4p/3)^t]（组合通道，memory_circuit 已改）
2. 消融匹配分母：无空闲 FoldQ 的改善必须除以【无空闲平面基线】，
   新增 idle0_base 条件（旧 exp29 的 idle0 行除以了有空闲基线，结论不成立）

复验条件（每码 5 个）：base、a1.5_b1.5（工作点）、a3_b3（最坏角）、
idle0_base、idle0_a1.5_b1.5（匹配消融对）。8 码 × 5 = 40 条件。

用法：
  python exp29b_formula_recheck.py --workers 3
  python exp29b_formula_recheck.py --summarize-only
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

CODES = ['GT_12_4_2', 'GT_24_4_4', 'D4_3', 'D6_3', 'D8_3',
         'GT_78_4_10', 'GT_90_8_10', 'GT_98_6_12']
K_PCT, L_MAX = 0.10, 6
PM_RATIO, PIDLE_RATIO, T_MEAS = 0.5, 0.1, 20
N_ROUNDS, SHOTS = 3, 3000
BP_MAX_ITER, OSD_ORDER = 100, 4

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp29b')
EXP29_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp29', 'tasks.jsonl')
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp29b_formula_recheck.py']
_cache = {}
_pstars = {}


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
        _cache[name] = {'code': code, 'inter': inter, 'phys': phys, 'size': size,
                        'g0': g0, 'g': CouplingGraph(size, cross)}
    return _cache[name]


def _pstar29(name):
    if not _pstars:
        with open(EXP29_TASKS, encoding='utf-8') as f:
            for line in f:
                if line.strip() and json.loads(line).get('stage') == 'cal':
                    _pstars[json.loads(line)['code']] = json.loads(line)['p_star29']
    return _pstars[name]


def task_ler(args):
    name, cond, alpha, beta, use_cross, p_idle_ratio = args
    from foldq.simulation.memory_circuit import (route_rounds_timed,
                                                 build_memory_circuit_timed)
    from foldq.simulation.bposd import logical_error_rate
    t0 = time.time()
    st = _setup(name)
    p = _pstar29(name)
    graph = st['g'] if use_cross else st['g0']
    rounds = route_rounds_timed(st['inter'], st['phys'], graph, N_ROUNDS,
                                 cross_time_factor=beta)
    c = build_memory_circuit_timed(st['code'], rounds, p, alpha, p * PM_RATIO,
                                   p_idle=p_idle_ratio * p, t_meas=T_MEAS)
    r = logical_error_rate(c, SHOTS, seed=_seed(name, 'e29b2', cond),
                           max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    r.update({'code': name, 'cond': cond, 'p': p, 'alpha': alpha, 'beta': beta,
              'use_cross': use_cross, 'p_idle_ratio': p_idle_ratio,
              'seconds': time.time() - t0})
    return r


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
    conds = [('base', 'idle0_base', 'a1.5_b1.5', 'a3_b3', 'idle0_a1.5_b1.5')]
    rows = []
    for c in CODES:
        if c not in ler or 'base' not in ler[c]:
            continue
        b = ler[c]['base']
        for cond in ('a1.5_b1.5', 'a3_b3'):
            f = ler[c].get(cond)
            if f:
                rows.append((c, cond, _two_prop_p(b['failures'], b['shots'],
                                                  f['failures'], f['shots'])))
        # 匹配消融：无空闲 FoldQ vs 无空闲基线
        b0 = ler[c].get('idle0_base')
        f0 = ler[c].get('idle0_a1.5_b1.5')
        if b0 and f0:
            rows.append((c, 'idle0_pair', _two_prop_p(b0['failures'], b0['shots'],
                                                      f0['failures'], f0['shots'])))
    holm = _holm([x[2] for x in rows])
    hmap = {(c, cond): h for (c, cond, _), h in zip(rows, holm)}
    L = ['# 实验29b：空闲噪声公式修正后的关键结论复验', '',
         '公式：组合通道 (3/4)[1-(1-4p/3)^t]；消融使用匹配的无空闲分母；'
         'p* 沿用 exp29 标定；3000 shots。', '',
         '| 码 | 工作点 α1.5β1.5 | 最坏角 α3β3 | 无空闲基线 LER | 无空闲 FoldQ | '
         '无空闲改善(匹配分母) | 有/无空闲改善对比 |', '|---|---|---|---|---|---|---|']
    for c in CODES:
        if c not in ler or 'base' not in ler[c]:
            continue
        b, f, f3 = ler[c]['base'], ler[c].get('a1.5_b1.5'), ler[c].get('a3_b3')
        b0, f0 = ler[c].get('idle0_base'), ler[c].get('idle0_a1.5_b1.5')
        def imp(x, y):
            return f"{(1 - y['ler'] / x['ler']) * 100:+.0f}%" if x and y and x['ler'] > 0 else '—'
        wp = (imp(b, f) + ('*' if hmap.get((c, 'a1.5_b1.5'), 1) < 0.05 else '')) if f else '—'
        wc = (imp(b, f3) + ('*' if hmap.get((c, 'a3_b3'), 1) < 0.05 else '')) if f3 else '—'
        id0 = (imp(b0, f0) + ('*' if hmap.get((c, 'idle0_pair'), 1) < 0.05 else '')) if b0 and f0 else '—'
        cmp_txt = '—'
        if f and b0 and f0 and b['ler'] > 0 and b0['ler'] > 0:
            wi = 1 - f['ler'] / b['ler']
            w0 = 1 - f0['ler'] / b0['ler']
            cmp_txt = f"有{wi*100:+.0f}% / 无{w0*100:+.0f}%"
        L.append(f"| {c} | {wp} | {wc} | "
                 f"{b0['ler']:.4f} | {f0['ler']:.4f} | {id0} | {cmp_txt} |")
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
    meta = {'codes': CODES, 'fix': 'composite-channel idle merge + matched idle0 denominator',
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
    conds = [('base', 1.0, 1.0, False, PIDLE_RATIO),
             ('a1.5_b1.5', 1.5, 1.5, True, PIDLE_RATIO),
             ('a3_b3', 3.0, 3.0, True, PIDLE_RATIO),
             ('idle0_base', 1.0, 1.0, False, 0.0),
             ('idle0_a1.5_b1.5', 1.5, 1.5, True, 0.0)]
    tasks = [(c, cond, al, be, uc, pr) for c in sorted(CODES, reverse=True)
             for cond, al, be, uc, pr in conds if (c, cond) not in done]
    print(f'exp29b：{len(tasks)} 任务，{a.workers} 进程', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            if i % 5 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {rec['code']:12s} {rec['cond']:16s} "
                      f"ler={rec['ler']:.4f} 累计 {time.time()-t0:.0f}s", flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
