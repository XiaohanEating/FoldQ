"""
实验29：门时长与空闲噪声下的净收益边界（α × β 扫描，带时间片的电路）

问题：跨层门更慢（β 倍时长）时，SWAP 减少带来的周期缩短是否仍胜过等待错误
的累积？净收益边界 P_L(FoldQ) < P_L(平面) 在 (α, β) 平面的哪里？

协议：
- 8 个代表码（小中大），工作点 k=10%、L6 候选池（与 exp28b 一致），p* 复用 exp25
- 电路：build_memory_circuit_timed——ASAP 调度 + 空闲 DEPOLARIZE1(p_idle=0.1·p*)
  + 测量时长 t_meas=20 半 tick；跨层门时长经 route_round(cross_time_factor=β)
- 网格：α ∈ {1.0, 1.5, 2.0, 2.5, 3.0} × β ∈ {1.0, 1.5, 2.0, 2.5, 3.0}（25 格/码）
- **标定**：p 在带空闲噪声的平面基线电路上重新二分（目标 LER∈[3%,8%]）——
  exp25 的 p* 是无空闲电路标定的，直接沿用会把比较推进解码饱和区
  （冒烟实测：D4_3 用 exp25 的 p* 时基线 LER 58%）
- 基线：平面无跨层门，α/β 不影响基线 → 每码 1 条（p_idle 同样开——
  基线每轮 SWAP 多、周期长，等待错误更多，这是 FoldQ 的隐藏收益来源）
- 消融：α=1.5 行补 p_idle=0 的 5 格（分离 β 调度效应与空闲噪声效应）
- 每条件 3000 shots；Clopper-Pearson CI；双比例检验 + Holm

用法：
  python exp29_alpha_beta_timed.py --workers 3
  python exp29_alpha_beta_timed.py --summarize-only
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
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
K_PCT, L_MAX = 0.10, 6
PM_RATIO = 0.5
PIDLE_RATIO = 0.1
T_MEAS = 20
N_ROUNDS = 3
SHOTS = 3000
BP_MAX_ITER, OSD_ORDER = 100, 4

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp29')
EXP25_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp25', 'tasks.jsonl')
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp29_alpha_beta_timed.py']

_cache = {}


def _seed(*parts) -> int:
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


def task_cal(args):
    """带空闲噪声的平面基线电路上二分 p，使 LER 落入 [3%, 8%]"""
    name, = args
    import math
    t0 = time.time()
    st = _setup(name)
    lo, hi = math.log(1e-6), math.log(3e-3)
    band, target, hist = [0.03, 0.08], 0.05, []
    for it in range(8):
        p = math.exp((lo + hi) / 2)
        r = task_ler((name, f'cal{it}', p, 1.0, 1.0, False, PIDLE_RATIO))
        hist.append({'p': p, 'ler': r['ler']})
        if band[0] <= r['ler'] <= band[1]:
            break
        if r['ler'] < target:
            lo = math.log(p)
        else:
            hi = math.log(p)
    best = min(hist, key=lambda h: abs(h['ler'] - target))
    return {'code': name, 'stage': 'cal', 'p_star29': best['p'],
            'cal_ler': best['ler'], 'in_band': band[0] <= best['ler'] <= band[1],
            'history': hist, 'seconds': time.time() - t0}


def task_ler(args):
    name, cond, p, alpha, beta, use_cross, p_idle_ratio = args
    from foldq.simulation.memory_circuit import (route_rounds_timed,
                                                 build_memory_circuit_timed)
    from foldq.simulation.bposd import logical_error_rate
    t0 = time.time()
    st = _setup(name)
    graph = st['g'] if use_cross else st['g0']
    rounds = route_rounds_timed(st['inter'], st['phys'], graph, N_ROUNDS,
                                 cross_time_factor=beta)
    c = build_memory_circuit_timed(st['code'], rounds, p, alpha, p * PM_RATIO,
                                   p_idle=p_idle_ratio * p, t_meas=T_MEAS)
    r = logical_error_rate(c, SHOTS, seed=_seed(name, 'e29', cond),
                           max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    # 每轮深度（tick）：最大门结束时刻 / 2（半 tick → CX 时长单位）
    depth = max((ev[4] + ev[5]) for s, _, _ in rounds for ev in s) / 2 if rounds else 0
    r.update({'code': name, 'cond': cond, 'p': p, 'alpha': alpha, 'beta': beta,
              'use_cross': use_cross, 'p_idle_ratio': p_idle_ratio,
              'round_depth': depth, 'seconds': time.time() - t0})
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
    ler, cal29 = {}, {}
    for r in records:
        if r.get('stage') == 'cal':
            cal29[r['code']] = r
        elif 'cond' in r:
            ler.setdefault(r['code'], {})[r['cond']] = r
    L = ['# 实验29：α × β 扫描（带时间片电路，空闲噪声 p_idle=0.1p，测量 20 半tick）', '',
         f'工作点 k={K_PCT:.0%}、L{L_MAX}；p* 为带空闲噪声的基线电路上重新标定'
         '（exp25 的 p* 沿用会把比较推进饱和区）；每条件 '
         f'{SHOTS} shots。改善 = 1 − LER(FoldQ@α,β)/LER(平面)，'
         '平面基线每码一条（α/β 不影响基线）。', '']
    raw = []
    for code, conds in ler.items():
        b = conds.get('base')
        if not b:
            continue
        for a in ALPHAS:
            for be in BETAS:
                f = conds.get(f'a{a:g}_b{be:g}')
                if f:
                    raw.append((code, a, be, _two_prop_p(b['failures'], b['shots'],
                                                         f['failures'], f['shots'])))
    holm = _holm([x[3] for x in raw])
    hmap = {(c, a, b): h for (c, a, b, _), h in zip(raw, holm)}
    for code in CODES:
        if code not in ler or 'base' not in ler[code]:
            continue
        b = ler[code]['base']
        L += [f'## {code}（基线 LER {b["ler"]:.4f}，每轮深度 {b.get("round_depth", 0):.0f} 半tick）',
              '', '| α\\β | ' + ' | '.join(f'β={be:g}' for be in BETAS) + ' |',
              '|---|' + '---|' * len(BETAS)]
        for a in ALPHAS:
            row = []
            for be in BETAS:
                f = ler[code].get(f'a{a:g}_b{be:g}')
                if not f:
                    row.append('—')
                    continue
                imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
                mark = '*' if hmap.get((code, a, be), 1) < 0.05 else ''
                row.append(f"{imp*100:+.0f}%{mark}（{f['ler']:.4f}）")
            L.append(f'| α={a:g} | ' + ' | '.join(row) + ' |')
        # 消融行
        ab = [ler[code].get(f'idle0_a1.5_b{be:g}') for be in BETAS]
        if any(ab):
            row = []
            for f in ab:
                if not f:
                    row.append('—')
                    continue
                imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
                row.append(f"{imp*100:+.0f}%（{f['ler']:.4f}）")
            L.append(f'| α=1.5, 无空闲 | ' + ' | '.join(row) + ' |')
        L.append('')
    sig = sum(1 for h in holm if h < 0.05)
    worse_raw = sum(1 for c, a, b_, p in raw if ler[c][f'a{a:g}_b{b_:g}']['ler'] > ler[c]['base']['ler'])
    L += [f'Holm 校正后显著改善 {sig} / {len(raw)}；LER 绝对值高于基线（改善为负）的格数 {worse_raw}。']
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

    # 标定先行（断点续跑：已完成标定的码跳过）
    all_recs = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
    cal_done = {r['code']: r for r in all_recs if r.get('stage') == 'cal'}
    e25n = {}
    with open(EXP25_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip() and json.loads(line).get('stage') == 'cal':
                e25n[json.loads(line)['code']] = json.loads(line)['num_physical']
    cal_tasks = [(n,) for n in sorted(CODES, key=lambda c: -e25n.get(c, 0))
                 if n not in cal_done]
    if cal_tasks:
        print(f'标定：{len(cal_tasks)} 码（带空闲噪声的基线电路）', flush=True)
        with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
            for rec in pool.imap_unordered(task_cal, cal_tasks):
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                print(f"  {rec['code']:12s} p*29={rec['p_star29']:.2e} "
                      f"ler={rec['cal_ler']:.3f} {rec['seconds']:.0f}s", flush=True)
        with open(path, encoding='utf-8') as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
        cal_done = {r['code']: r for r in all_recs if r.get('stage') == 'cal'}

    done = {(r['code'], r['cond']) for r in all_recs
            if r.get('stage') != 'cal' and 'cond' in r}
    tasks = []
    for name in sorted(CODES, key=lambda c: -cal_done[c]['p_star29']):
        p = cal_done[name]['p_star29']
        if (name, 'base') not in done:
            tasks.append((name, 'base', p, 1.0, 1.0, False, PIDLE_RATIO))
        for al in ALPHAS:
            for be in BETAS:
                cond = f'a{al:g}_b{be:g}'
                if (name, cond) not in done:
                    tasks.append((name, cond, p, al, be, True, PIDLE_RATIO))
        for be in BETAS:  # 消融：α=1.5、无空闲
            cond = f'idle0_a1.5_b{be:g}'
            if (name, cond) not in done:
                tasks.append((name, cond, p, 1.5, be, True, 0.0))
    print(f'exp29：{len(tasks)} 个任务，{a.workers} 进程', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {rec['code']:12s} {rec['cond']:14s} "
                      f"ler={rec['ler']:.4f}  累计 {time.time()-t0:.0f}s", flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
