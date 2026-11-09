"""
实验15b（复核 5.3）：random 基线的种子化重跑

exp15 的 random 选择未设种子（np.random.default_rng() 无参），无法逐次重放。
本实验按原范围（105 码 × 每码 10 次随机选边）固定种子重跑，保存每次选边
与 SWAP 指标；与 exp15 存档的 random 均值对照。

种子规则：rng = default_rng(sha256(码名)[:8])，逐次选边用同一 rng 顺序推进。

用法：
  python exp15b_random_seeded.py --workers 3
  python exp15b_random_seeded.py --summarize-only
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp15b')
EXP15_RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'exp15', 'results.json')
K_PCT = 0.10
LAYOUT_SEED = 0
N_TRIALS = 10
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/optimization/crosslayer_select.py',
           'foldq/optimization/layouts.py', 'foldq/routing/router.py',
           'foldq/experiments/exp15b_random_seeded.py']
_cache = {}


def _setup(name):
    if name not in _cache:
        from foldq.codes.hal_all_codes import load_code_by_name
        from foldq.optimization.layouts import spring_layout, default_grid_size
        from foldq.optimization.crosslayer_select import candidate_pairs
        from foldq.routing import CouplingGraph, positions_to_phys
        code = load_code_by_name(name)
        inter = code.get_all_interactions()
        size = default_grid_size(code)
        pos = spring_layout(code, size, LAYOUT_SEED)
        phys = positions_to_phys(pos, size)
        g0 = CouplingGraph(size)
        cands = candidate_pairs(inter, phys, g0.dist)
        _cache[name] = {'code': code, 'inter': inter, 'size': size, 'phys': phys,
                        'g0': g0, 'cands': cands}
    return _cache[name]


def task_one(args):
    name, = args
    from foldq.optimization.crosslayer_select import select_random
    from foldq.routing import CouplingGraph
    from foldq.routing.router import route_round
    t0 = time.time()
    st = _setup(name)
    rng_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    k = max(1, round(K_PCT * st['code'].num_physical))
    trials = []
    for t in range(N_TRIALS):
        cross = select_random(st['cands'], st['g0'].dist, k, rng) if st['cands'] else []
        g = CouplingGraph(st['size'], cross)
        p, tot = dict(st['phys']), 0
        for _ in range(3):
            r = route_round(st['inter'], p, g, return_ops=True)
            tot += r['swaps']
            p = r['final_phys']
        trials.append({'trial': t, 'n_cross': len(cross),
                       'cross': [list(e) for e in cross], 'swaps_per_round': tot / 3})
    sw = [t['swaps_per_round'] for t in trials]
    return {'code': name, 'n': st['code'].num_physical,
            'rng_seed': rng_seed, 'trials': trials,
            'swaps_mean': float(np.mean(sw)), 'swaps_std': float(np.std(sw)),
            'seconds': time.time() - t0}


def summarize(records, out_dir):
    e15 = json.load(open(EXP15_RESULTS, encoding='utf-8'))['results']
    old = {r['code_name']: r for r in e15}
    L = ['# 实验15b：random 基线种子化重跑（105 码 × 10 次，逐次选边已存档）', '',
         '种子 = sha256(码名) 前 8 位；布局/预算/路由与 exp15 同配置。', '',
         '| 码 | n | 本次均值±std | exp15 存档均值 | 差 |', '|---|---|---|---|---|']
    diffs = []
    for r in sorted(records, key=lambda x: x['n']):
        o = old.get(r['code'])
        if not o:
            continue
        om = o['random_greedy'].get('swaps_avg', o['random_greedy'].get('swaps'))
        d = (r['swaps_mean'] - om) / om * 100 if om else float('nan')
        diffs.append(abs(d))
        L.append(f"| {r['code']} | {r['n']} | {r['swaps_mean']:.0f}±{r['swaps_std']:.0f} | "
                 f"{om:.0f} | {d:+.1f}% |")
    import statistics as st
    L += ['', f'与存档均值差的绝对值：均值 {st.mean(diffs):.1f}%、最大 {max(diffs):.1f}%（随机基线的重采样波动）。']
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
    meta = {'n_trials': N_TRIALS, 'k_pct': K_PCT, 'layout_seed': LAYOUT_SEED,
            'seed_rule': 'sha256(code_name)[:8] as int, sequential draws',
            'source_sha256_12': {s: hashlib.sha256(open(
                os.path.join(os.path.dirname(__file__), '..', '..', s), 'rb')
                .read()).hexdigest()[:12] for s in SOURCES},
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = {json.loads(l)['code'] for l in f if l.strip()}
    codes = [c.name for c in sorted(load_all_hal_codes(), key=lambda c: -c.num_physical)
             if c.name not in done]
    print(f'exp15b：{len(codes)} 码，{a.workers} 进程', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_one, [(c,) for c in codes]), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            if i % 10 == 0 or i == len(codes):
                print(f"  [{i}/{len(codes)}] {rec['code']:14s} mean={rec['swaps_mean']:.0f} "
                      f"累计 {time.time()-t0:.0f}s", flush=True)
    summarize([json.loads(l) for l in open(path, encoding='utf-8') if l.strip()], OUT_DIR)


if __name__ == '__main__':
    main()
