"""
实验25：逻辑错误率（按码标定噪声，BP-OSD 完整解码）

替代已作废的 exp22–24（见 foldq/archive_invalid/README.md）。

协议：
- 码：HAL qLDPC 码中 num_physical ≤ 200 的 24 个（3 个 qTanner + 21 个 GT）
- 布局：spring，种子 0；基线 = 平面方格；FoldQ = 平面方格 + marginal 选出的跨层耦合器
- 电路：3 轮 Z 基存储实验，每轮从当前布局重新路由（simulation/memory_circuit.py）
- 噪声：CNOT 后 DEPOLARIZE2，平面 p、跨层 alpha·p；测量翻转 p/2
- 解码：BP（最小和，100 次迭代）+ OSD-CS 4 阶；任一逻辑观测量判错计一次失败。
  只声明 Z 型探测器（X、Z 分开解码），X 型稳定子照常测量、其噪声全部计入
- 标定：对每个码，在 log p 上二分，找使基线逻辑错误率落在 [3%, 8%] 的 p*（每次 300 次采样）
- 比较：在 p* 下，每个条件 3000 次采样
    基线（alpha 不影响基线，因为基线没有跨层门）
    alpha 扫描：k=10%，alpha ∈ {0.8, 1.0, 1.2, 1.5, 2.0, 2.5}
    k 扫描：alpha=1.5，k ∈ {1%, 2%, 5%, 10%, 15%, 20%}（k=10% 与 alpha=1.5 共用）
- 统计：Clopper-Pearson 95% 置信区间；基线与 FoldQ 用双比例 z 检验，p<0.05 视为显著

任务按"码 × 条件"拆分，多进程并行；每完成一个任务追加写一行到 tasks.jsonl，
重启时跳过已完成的任务。并行只改变调度，不改变任何任务的种子与结果。

用法：
  python exp25_ler_calibrated.py --workers 4          # 全量
  python exp25_ler_calibrated.py --smoke              # 2 个小码、少量采样，检查流程
  python exp25_ler_calibrated.py --summarize-only     # 只从 tasks.jsonl 重新生成汇总
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.stats import norm

CONFIG = {
    'max_n_physical': 200,
    'layout_seed': 0,
    'n_rounds': 3,
    'k_main': 0.10,
    'alpha_main': 1.5,
    'alphas': [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
    'k_pcts': [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    'pm_ratio': 0.5,
    'target_ler': 0.05,
    'cal_band': [0.03, 0.08],
    'cal_shots': 300,
    'cal_max_iter': 10,
    'p_lo': 1e-5,
    'p_hi': 1e-2,
    'shots': 3000,
    'bp_max_iter': 100,
    'osd_order': 4,
    'x_detectors': False,
}
SMOKE = {'cal_shots': 100, 'shots': 300, 'cal_max_iter': 6}

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/codes/gf2.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/simulation/memory_circuit.py',
           'foldq/simulation/bposd.py', 'foldq/experiments/exp25_ler_calibrated.py']

_cfg = dict(CONFIG)
_cache = {}


def _seed(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _setup(name):
    """工作进程内按码缓存：码、交互、布局、平面图、候选边、各预算的耦合图"""
    if name not in _cache:
        from foldq.codes.hal_all_codes import load_code_by_name
        from foldq.optimization.layouts import spring_layout, default_grid_size
        from foldq.optimization.crosslayer_select import candidate_pairs
        from foldq.routing import CouplingGraph, positions_to_phys
        code = load_code_by_name(name)
        inter = code.get_all_interactions()
        size = default_grid_size(code)
        phys = positions_to_phys(spring_layout(code, size, _cfg['layout_seed']), size)
        g0 = CouplingGraph(size)
        _cache[name] = {'code': code, 'inter': inter, 'size': size, 'phys': phys, 'g0': g0,
                        'cands': candidate_pairs(inter, phys, g0.dist), 'graphs': {}}
    return _cache[name]


def _graph(st, k_pct):
    if k_pct == 0:
        return st['g0'], 0
    if k_pct not in st['graphs']:
        from foldq.optimization.crosslayer_select import select_marginal
        from foldq.routing import CouplingGraph
        code = st['code']
        k = max(1, round(k_pct * code.num_physical))
        cross = select_marginal(st['cands'], st['g0'].dist, st['inter'], st['phys'], k) if st['cands'] else []
        st['graphs'][k_pct] = (CouplingGraph(st['size'], cross), len(cross))
    return st['graphs'][k_pct]


def _ler(st, k_pct, p, alpha, shots, seed):
    from foldq.simulation.memory_circuit import route_rounds, build_memory_circuit
    from foldq.simulation.bposd import logical_error_rate
    g, n_cross = _graph(st, k_pct)
    rounds = route_rounds(st['inter'], st['phys'], g, _cfg['n_rounds'])
    c = build_memory_circuit(st['code'], rounds, p, alpha, p * _cfg['pm_ratio'],
                             x_detectors=_cfg['x_detectors'])
    r = logical_error_rate(c, shots, seed=seed, max_iter=_cfg['bp_max_iter'], osd_order=_cfg['osd_order'])
    r['swaps_total'] = sum(1 for ops, _ in rounds for op in ops if op[0] == 'SWAP')
    r['n_cross'] = n_cross
    return r


def task_calibrate(args):
    name, cfg = args
    _cfg.update(cfg)
    t0 = time.time()
    st = _setup(name)
    lo, hi = math.log(_cfg['p_lo']), math.log(_cfg['p_hi'])
    band, target, hist = _cfg['cal_band'], _cfg['target_ler'], []
    for it in range(_cfg['cal_max_iter']):
        p = math.exp((lo + hi) / 2)
        r = _ler(st, 0, p, 1.0, _cfg['cal_shots'], _seed(name, 'cal', it))
        hist.append({'p': p, 'ler': r['ler'], 'failures': r['failures']})
        if band[0] <= r['ler'] <= band[1]:
            break
        if r['ler'] < target:
            lo = math.log(p)
        else:
            hi = math.log(p)
    best = min(hist, key=lambda h: abs(h['ler'] - target))
    return {'stage': 'cal', 'code': name, 'num_physical': st['code'].num_physical,
            'k_logical': st['code'].num_logical, 'p_star': best['p'], 'cal_ler': best['ler'],
            'in_band': band[0] <= best['ler'] <= band[1], 'history': hist,
            'seconds': time.time() - t0}


def task_ler(args):
    name, cond, k_pct, alpha, p, cfg = args
    _cfg.update(cfg)
    t0 = time.time()
    st = _setup(name)
    r = _ler(st, k_pct, p, alpha, _cfg['shots'], _seed(name, cond))
    r.update({'stage': 'ler', 'code': name, 'cond': cond, 'k_pct': k_pct, 'alpha': alpha,
              'p': p, 'seconds': time.time() - t0})
    return r


def _conditions():
    conds = [('base', 0, 1.0)]
    conds += [(f'alpha_{a}', _cfg['k_main'], a) for a in _cfg['alphas']]
    conds += [(f'k_{k}', k, _cfg['alpha_main']) for k in _cfg['k_pcts'] if k != _cfg['k_main']]
    return conds


def _load_done(path):
    done = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = [json.loads(line) for line in f if line.strip()]
    return done


def _run_pool(func, tasks, path, workers, label):
    if not tasks:
        return
    print(f"{label}: {len(tasks)} 个任务，{workers} 个进程", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(func, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            tag = rec.get('cond', 'cal')
            val = rec.get('p_star', rec.get('ler'))
            print(f"  [{i}/{len(tasks)}] {rec['code']:14s} {tag:10s} {val:.3e}  {rec['seconds']:.0f}s"
                  f"  累计 {time.time()-t0:.0f}s", flush=True)


def _two_prop_p(k1, n1, k2, n2):
    pbar = (k1 + k2) / (n1 + n2)
    if pbar in (0, 1):
        return 1.0
    z = (k1 / n1 - k2 / n2) / math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    return float(2 * (1 - norm.cdf(abs(z))))


def summarize(records, out_dir):
    cal = {r['code']: r for r in records if r['stage'] == 'cal'}
    ler = {}
    for r in records:
        if r['stage'] == 'ler':
            ler.setdefault(r['code'], {})[r['cond']] = r
    names = sorted(cal, key=lambda n: cal[n]['num_physical'])

    def cell(code, cond):
        b, f = ler[code].get('base'), ler[code].get(cond)
        if not b or not f:
            return '—', None
        imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
        pv = _two_prop_p(b['failures'], b['shots'], f['failures'], f['shots'])
        mark = '*' if pv < 0.05 else ''
        return f"{imp*100:+.0f}%{mark}", (imp, pv)

    L = ['# 实验25：逻辑错误率（按码标定噪声，BP-OSD 解码）', '',
         f"码数：{len(names)}；每码先标定 p*（基线 LER 目标 {CONFIG['target_ler']:.0%}），"
         f"再在 p* 下比较，每个条件 {_cfg['shots']} 次采样，{_cfg['n_rounds']} 轮。"
         '改善 = 1 − LER(FoldQ)/LER(基线)，* 表示双比例检验 p<0.05。', '',
         '## 主比较（alpha=1.5，k=10%）', '',
         '| 码 | n | k | p* | 基线 LER [95% CI] | FoldQ LER [95% CI] | 改善 | p 值 | 每轮 SWAP 基线→FoldQ |',
         '|---|---|---|---|---|---|---|---|---|']
    main = f"alpha_{CONFIG['alpha_main']}"
    stats = {'better': 0, 'worse': 0, 'ns': 0}
    for n in names:
        if n not in ler or 'base' not in ler[n] or main not in ler[n]:
            continue
        b, f = ler[n]['base'], ler[n][main]
        txt, res = cell(n, main)
        imp, pv = res
        stats['better' if pv < 0.05 and imp > 0 else 'worse' if pv < 0.05 else 'ns'] += 1
        R = _cfg['n_rounds']
        L.append(f"| {n} | {cal[n]['num_physical']} | {cal[n]['k_logical']} | {cal[n]['p_star']:.2e} | "
                 f"{b['ler']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | "
                 f"{f['ler']:.4f} [{f['ci95'][0]:.4f}, {f['ci95'][1]:.4f}] | {txt} | {pv:.1e} | "
                 f"{b['swaps_total']/R:.0f}→{f['swaps_total']/R:.0f} |")
    L += ['', f"显著更好 {stats['better']}，显著更差 {stats['worse']}，不显著 {stats['ns']}。", '']

    for title, conds, lab in (
            ('alpha 扫描（k=10%）', [f'alpha_{a}' for a in _cfg['alphas']], [f'α={a}' for a in _cfg['alphas']]),
            ('k 扫描（alpha=1.5）', [f'k_{k}' if k != _cfg['k_main'] else main for k in _cfg['k_pcts']],
             [f'k={k:.0%}' for k in _cfg['k_pcts']])):
        L += [f'## {title}', '', '| 码 | n | ' + ' | '.join(lab) + ' |', '|---|---|' + '---|' * len(lab)]
        for n in names:
            if n in ler:
                L.append(f"| {n} | {cal[n]['num_physical']} | " + ' | '.join(cell(n, c)[0] for c in conds) + ' |')
        L.append('')

    L += ['## 标定', '', '| 码 | p* | 标定 LER | 落在目标区间 | 二分次数 |', '|---|---|---|---|---|']
    for n in names:
        c = cal[n]
        L.append(f"| {n} | {c['p_star']:.2e} | {c['cal_ler']:.3f} | {'是' if c['in_band'] else '否'} | {len(c['history'])} |")
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        _cfg.update(SMOKE)
    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'exp25_smoke' if a.smoke else 'exp25')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'tasks.jsonl')

    if not a.summarize_only:
        from foldq.codes.hal_all_codes import load_all_hal_codes
        codes = sorted((c for c in load_all_hal_codes() if c.num_physical <= _cfg['max_n_physical']),
                       key=lambda c: -c.num_physical)  # 大码先跑，便于负载均衡
        if a.smoke:
            codes = codes[-2:]
        names = [c.name for c in codes]

        import stim, ldpc, scipy
        meta = {'config': _cfg, 'smoke': a.smoke, 'workers': a.workers, 'codes': names,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'stim': stim.__version__, 'ldpc': getattr(ldpc, '__version__', '?'),
                'source_sha256_12': {s: hashlib.sha256(open(os.path.join(ROOT, s), 'rb').read()).hexdigest()[:12]
                                     for s in SOURCES},
                'started': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(os.path.join(out_dir, f"meta_{time.strftime('%Y%m%d_%H%M%S')}.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

        done = _load_done(path)
        have_cal = {r['code'] for r in done if r['stage'] == 'cal'}
        _run_pool(task_calibrate, [(n, _cfg) for n in names if n not in have_cal], path, a.workers, '标定')

        done = _load_done(path)
        cal = {r['code']: r for r in done if r['stage'] == 'cal'}
        have = {(r['code'], r['cond']) for r in done if r['stage'] == 'ler'}
        tasks = [(n, cond, k, al, cal[n]['p_star'], _cfg) for n in names for cond, k, al in _conditions()
                 if (n, cond) not in have]
        _run_pool(task_ler, tasks, path, a.workers, '比较')

    summarize(_load_done(path), out_dir)
    print(f"汇总写入 {os.path.join(out_dir, 'summary.md')}")


if __name__ == '__main__':
    main()
