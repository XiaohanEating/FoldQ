"""
实验26：exp25 证据链的压力测试（独立复核）

三部分：
  --part audit  6 个代表码的单故障扫描：确定性 Pauli 注入（p=1.0），
                预期探测器/观测量签名从 GF(2) 校验矩阵独立推导，不经 Stim 的 DEM。
                注入类别：
                  A  数据比特 X/Z/Y，第 1 轮路由前（错误随 SWAP 移动是语义的一部分）
                  B  数据比特 X/Z/Y，最终数据测量前（按路由后的最终布局定位）
                  C  逻辑 X 算符支撑集（不可检测错误，只应翻观测量）
                  D  辅助比特 X/Z，最后一轮测量前
  --part fixp   24 码 × 固定物理噪声 p ∈ {1e-4, 3e-4, 1e-3}：基线 vs FoldQ 的绝对 LER，
                检验 exp25 的规模优势不是按码标定 p* 的产物
  --part xdet   24 码主比较 + x_detectors=True：验证只声明 Z 型探测器的口径不影响结论

布局/选择器/路由/电路与 exp25 完全同配置（spring 种子 0，marginal k=10%，alpha=1.5）。
p* 直接读 exp25 的标定结果，不重新标定。

用法：
  python exp26_ler_audit.py --part audit
  python exp26_ler_audit.py --part fixp  --workers 4
  python exp26_ler_audit.py --part xdet  --workers 4
  python exp26_ler_audit.py --part summarize
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

AUDIT_CODES = ['GT_12_4_2', 'GT_24_4_4', 'D4_3', 'D6_3', 'D8_3', 'GT_98_6_12']
FIXED_PS = [1e-4, 3e-4, 1e-3]
K_PCT, ALPHA = 0.10, 1.5
PM_RATIO = 0.5
N_ROUNDS = 3
SHOTS = 3000
BP_MAX_ITER, OSD_ORDER = 100, 4
X_DETECTORS = False  # fixp/主口径与 exp25 一致；xdet 部分显式置 True

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCES = ['foldq/codes/hal_all_codes.py', 'foldq/codes/gf2.py', 'foldq/routing/router.py',
           'foldq/optimization/crosslayer_select.py', 'foldq/optimization/layouts.py',
           'foldq/simulation/memory_circuit.py', 'foldq/simulation/bposd.py',
           'foldq/experiments/exp26_ler_audit.py']
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'exp26')
EXP25_TASKS = os.path.join(os.path.dirname(__file__), 'results', 'exp25', 'tasks.jsonl')

_cfg = {}
_cache = {}


def _seed(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _setup(name):
    """与 exp25 相同的码/布局/耦合图构建（含跨层边）"""
    if name not in _cache:
        from foldq.codes.hal_all_codes import load_code_by_name
        from foldq.optimization.layouts import spring_layout, default_grid_size
        from foldq.optimization.crosslayer_select import candidate_pairs, select_marginal
        from foldq.routing import CouplingGraph, positions_to_phys
        code = load_code_by_name(name)
        inter = code.get_all_interactions()
        size = default_grid_size(code)
        phys = positions_to_phys(spring_layout(code, size, 0), size)
        g0 = CouplingGraph(size)
        k = max(1, round(K_PCT * code.num_physical))
        cands = candidate_pairs(inter, phys, g0.dist)
        cross = select_marginal(cands, g0.dist, inter, phys, k) if cands else []
        _cache[name] = {'code': code, 'inter': inter, 'size': size, 'phys': phys,
                        'g0': g0, 'g': CouplingGraph(size, cross), 'n_cross': len(cross)}
    return _cache[name]


def _ler(st, p, alpha, shots, seed, x_detectors, k_pct):
    from foldq.simulation.memory_circuit import route_rounds, build_memory_circuit
    from foldq.simulation.bposd import logical_error_rate
    graph = st['g0'] if k_pct == 0 else st['g']   # 基线必须路由在纯平面图上
    rounds = route_rounds(st['inter'], st['phys'], graph, N_ROUNDS)
    c = build_memory_circuit(st['code'], rounds, p, alpha, p * PM_RATIO, x_detectors=x_detectors)
    r = logical_error_rate(c, shots, seed=seed, max_iter=BP_MAX_ITER, osd_order=OSD_ORDER)
    r['swaps_total'] = sum(1 for ops, _ in rounds for op in ops if op[0] == 'SWAP')
    r['n_cross'] = 0 if k_pct == 0 else st['n_cross']
    return r


def _exp25_pstars():
    cal = {}
    with open(EXP25_TASKS, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('stage') == 'cal':
                cal[r['code']] = r
    return cal


# ---------------------------------------------------------------- audit ----

def _inject_prefix(base: 'stim.Circuit', pos: int, pauli: str):
    import stim
    c = stim.Circuit()
    c.append(f'{pauli}_ERROR', [pos], 1.0)
    c += base
    return c


def _inject_before_final_data_measure(base: 'stim.Circuit', pos: int, pauli: str):
    """在最后一条数据测量（n targets 的 M）之前插入错误，其余指令原样复制"""
    import stim
    last_m = max(i for i, inst in enumerate(base) if inst.name == 'M')
    out = stim.Circuit()
    for i, inst in enumerate(base):
        if i == last_m:
            out.append(f'{pauli}_ERROR', [pos], 1.0)
        out.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
    return out


def _inject_before_last_ancilla_measure(base: 'stim.Circuit', pos: int, pauli: str,
                                        n_anc: int):
    """在最后一轮辅助测量（倒数第 n_anc 个 target 的那条 M）之前插入错误"""
    import stim
    m_idx = [i for i, inst in enumerate(base) if inst.name == 'M']
    target_m = m_idx[-2]  # [-1] 是最终数据测量
    out = stim.Circuit()
    for i, inst in enumerate(base):
        if i == target_m:
            out.append(f'{pauli}_ERROR', [pos], 1.0)
        out.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
    return out


def audit_one(name: str) -> dict:
    """对单个码执行 A–D 四类单故障扫描，返回逐类通过/失配明细"""
    import stim
    from foldq.codes.gf2 import logical_basis
    from foldq.routing.router import route_round
    from foldq.simulation.memory_circuit import build_memory_circuit

    st = _setup(name)
    code = st['code']
    n, A = code.num_data, code.num_physical - code.num_data
    ancillas = list(range(n, code.num_physical))
    stype = {s.ancilla_qubit: s.stabilizer_type for s in code.stabilizers}
    Hx, Hz = code.parity_check_matrices()
    Lz = logical_basis(Hx, Hz)
    Lx = logical_basis(Hz, Hx)
    z_rows = [s for s in code.stabilizers if s.stabilizer_type == 'Z']

    # 逐轮路由并记录每轮操作/起始布局，最后一轮后再路由一次拿 final_phys（贪心确定性，重放一致）
    rounds, phys = [], dict(st['phys'])
    for _ in range(N_ROUNDS):
        r = route_round(st['inter'], phys, st['g'], return_ops=True)
        rounds.append((r['operations'], phys))
        phys = r['final_phys']

    base = build_memory_circuit(code, rounds, 0.0, ALPHA, 0.0, x_detectors=False)

    # 探测器索引（与 memory_circuit 的声明顺序一致）：每轮 Z 型辅助按逻辑编号序，终末按 z_rows 序
    z_ancs = [a for a in ancillas if stype[a] == 'Z']
    zA = len(z_ancs)
    round_col = {a: i for i, a in enumerate(z_ancs)}
    term_row = {s.ancilla_qubit: t for t, s in enumerate(z_rows)}
    n_det = N_ROUNDS * zA + zA
    n_obs = Lz.shape[0]

    def sampler(c):
        det, obs = c.compile_detector_sampler().sample(1, separate_observables=True)
        return det[0].astype(np.uint8), obs[0].astype(np.uint8)

    def expect_A(q, pauli):
        e = np.zeros(n_det, dtype=np.uint8)
        if pauli in ('X', 'Y'):  # Z 分量对 Z 基存储无探测器签名
            for a in z_ancs:
                e[round_col[a]] = Hz[term_row[a], q]
        # 观测：X/Y 的 X 分量翻 Lz 列；纯 Z 与 Z 型逻辑算符对易，不翻
        obs = (Lz[:, q].astype(np.uint8).copy() if pauli in ('X', 'Y')
               else np.zeros(n_obs, dtype=np.uint8))
        return e, obs

    def expect_B(q, pauli):
        e = np.zeros(n_det, dtype=np.uint8)
        if pauli in ('X', 'Y'):
            for t in range(zA):
                e[N_ROUNDS * zA + t] = Hz[t, q]
        return e, (Lz[:, q].astype(np.uint8).copy() if pauli in ('X', 'Y')
                   else np.zeros(n_obs, dtype=np.uint8))

    def check(circuit, exp_det, exp_obs, cat, detail, fails):
        det, obs = sampler(circuit)
        ok = np.array_equal(det, exp_det) and np.array_equal(obs, exp_obs)
        if not ok:
            fails.append({'cat': cat, 'detail': detail,
                          'det_match': bool(np.array_equal(det, exp_det)),
                          'obs_match': bool(np.array_equal(obs, exp_obs)),
                          'got_det_nonzero': np.nonzero(det)[0][:8].tolist(),
                          'exp_det_nonzero': np.nonzero(exp_det)[0][:8].tolist(),
                          'got_obs': obs.tolist(), 'exp_obs': exp_obs.tolist()})
        return ok

    counts = {}
    fails = []
    phys0 = rounds[0][1]          # 第 1 轮起始：逻辑比特 -> 物理位置
    final_phys = phys             # 全部轮结束后的布局

    for q in range(n):                                   # A：第 1 轮路由前
        for pauli in ('X', 'Z', 'Y'):
            e_d, e_o = expect_A(q, pauli)
            ok = check(_inject_prefix(base, phys0[q], pauli), e_d, e_o,
                       'A', f'q{q}/{pauli}', fails)
            counts[f'A/{pauli}'] = counts.get(f'A/{pauli}', [0, 0])
            counts[f'A/{pauli}'][0 if ok else 1] += 1
    for q in range(n):                                   # B：最终数据测量前
        for pauli in ('X', 'Z', 'Y'):
            e_d, e_o = expect_B(q, pauli)
            ok = check(_inject_before_final_data_measure(base, final_phys[q], pauli),
                       e_d, e_o, 'B', f'q{q}/{pauli}', fails)
            counts[f'B/{pauli}'] = counts.get(f'B/{pauli}', [0, 0])
            counts[f'B/{pauli}'][0 if ok else 1] += 1
    for m, lx in enumerate(Lx):                          # C：逻辑 X 算符
        pre = stim.Circuit()
        for q in np.nonzero(lx)[0]:
            pre.append('X_ERROR', [phys0[q]], 1.0)
        pre += base
        e_o = (Lz.astype(np.uint8) @ lx.astype(np.uint8)) % 2
        ok = check(pre, np.zeros(n_det, dtype=np.uint8), e_o, 'C', f'Lx{m}', fails)
        counts['C'] = counts.get('C', [0, 0])
        counts['C'][0 if ok else 1] += 1
    for a in ancillas:                                   # D：最后一轮辅助测量前
        pos_last = None
        at = {p: q_ for q_, p in rounds[-1][1].items()}
        # 最后轮测量发生在该轮路由回放之后：重放该轮操作得到测量时位置
        pos = dict(rounds[-1][1])
        for op, q1, q2, _ in rounds[-1][0]:
            if op == 'SWAP':
                u, v = at.get(q1), at.get(q2)
                at[q1], at[q2] = v, u
                if u is not None:
                    pos[u] = q2
                if v is not None:
                    pos[v] = q1
        pos_last = pos[a]
        e = np.zeros(n_det, dtype=np.uint8)
        if stype[a] == 'Z':                              # X 错误：最后轮探测器 + 终末
            e[(N_ROUNDS - 1) * zA + round_col[a]] = 1
            e[N_ROUNDS * zA + term_row[a]] = 1
        for pauli in ('X', 'Z'):
            e_o = np.zeros(n_obs, dtype=np.uint8)
            exp = e if pauli == 'X' else np.zeros(n_det, dtype=np.uint8)
            ok = check(_inject_before_last_ancilla_measure(base, pos_last, pauli, A),
                       exp, e_o, 'D', f'a{a}({stype[a]})/{pauli}', fails)
            counts[f'D/{pauli}'] = counts.get(f'D/{pauli}', [0, 0])
            counts[f'D/{pauli}'][0 if ok else 1] += 1

    total_ok = sum(v[0] for v in counts.values())
    total_bad = sum(v[1] for v in counts.values())
    return {'code': name, 'n_data': n, 'n_anc': A, 'zA': zA, 'k': Lz.shape[0],
            'total': [total_ok, total_bad], 'counts': counts, 'fails': fails[:20]}


# ------------------------------------------------------------ fixp/xdet ----

def task_ler(args):
    name, cond, p, alpha, x_det, shots, seed, k_pct = args
    t0 = time.time()
    st = _setup(name)
    r = _ler(st, p, alpha, shots, seed, x_det, k_pct)
    r.update({'stage': 'ler', 'code': name, 'cond': cond, 'p': p, 'alpha': alpha,
              'k_pct': k_pct, 'x_detectors': x_det, 'seconds': time.time() - t0})
    return r


def _run_pool(tasks, path, workers, label, shots):
    if not tasks:
        return
    print(f"{label}: {len(tasks)} 个任务，{workers} 进程，每任务 {shots} shots", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(path, 'a', encoding='utf-8') as out:
        for i, rec in enumerate(pool.imap_unordered(task_ler, tasks), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f"  [{i}/{len(tasks)}] {rec['code']:14s} {rec['cond']:14s} "
                  f"ler={rec['ler']:.4f}  {rec['seconds']:.0f}s  累计 {time.time()-t0:.0f}s",
                  flush=True)


def _load_done(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            out = [json.loads(l) for l in f if l.strip()]
    return out


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


def run_fixp(workers):
    from foldq.codes.hal_all_codes import load_all_hal_codes
    out = os.path.join(OUT_DIR, 'fixed_p')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'tasks.jsonl')
    codes = sorted((c for c in load_all_hal_codes() if c.num_physical <= 200),
                   key=lambda c: -c.num_physical)
    done = {(r['code'], r['cond']) for r in _load_done(path) if r['stage'] == 'ler'}
    tasks = []
    for c in codes:
        for p in FIXED_PS:
            for cond, alpha, k_pct in (('base', 1.0, 0), ('foldq', ALPHA, K_PCT)):
                if (c.name, f'p{p:g}_{cond}') not in done:
                    tasks.append((c.name, f'p{p:g}_{cond}', p, alpha,
                                  X_DETECTORS, SHOTS, _seed(c.name, 'fixp', f'{p:g}', cond),
                                  k_pct))
    _run_pool(tasks, path, workers, '固定噪声对照', SHOTS)
    summarize_fixp(_load_done(path), out)


def run_xdet(workers):
    from foldq.codes.hal_all_codes import load_all_hal_codes
    out = os.path.join(OUT_DIR, 'x_detectors')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'tasks.jsonl')
    cal = _exp25_pstars()
    codes = sorted((c for c in load_all_hal_codes() if c.num_physical <= 200),
                   key=lambda c: -c.num_physical)
    done = {(r['code'], r['cond']) for r in _load_done(path) if r['stage'] == 'ler'}
    tasks = []
    for c in codes:
        p = cal[c.name]['p_star']
        for cond, alpha, k_pct in (('base_xdet', 1.0, 0), ('foldq_xdet', ALPHA, K_PCT)):
            if (c.name, cond) not in done:
                st_seed = _seed(c.name, 'xdet', cond)
                tasks.append((c.name, cond, p, alpha, True, SHOTS, st_seed, k_pct))
    _run_pool(tasks, path, workers, 'x_detectors=True 对照', SHOTS)
    summarize_xdet(_load_done(path), out, cal)


# -------------------------------------------------------------- summary ----

def summarize_fixp(records, out):
    ler = {}
    for r in records:
        if r['stage'] == 'ler':
            ler.setdefault(r['code'], {})[r['cond']] = r
    names = sorted(ler, key=lambda nm: next(iter(ler[nm].values())).get('n', 0)
                   or len(ler[nm]))
    L = ['# exp26 固定物理噪声对照（绝对 LER，无按码标定）', '',
         f'24 码 × p ∈ {FIXED_PS}，基线（平面）vs FoldQ（k=10%，α={ALPHA}），'
         f'每条件 {SHOTS} shots，{N_ROUNDS} 轮存储实验。p 值经 Holm 校正（全部 72 个检验族）。', '',
         '| 码 | n | ' + ' | '.join(f'p={p:g} 基线→FoldQ（改善，p_holm）' for p in FIXED_PS) + ' |',
         '|---|---|' + '---|' * len(FIXED_PS)]
    # n 从 tasks 记录补
    ns = {}
    for r in records:
        if r['stage'] == 'ler':
            ns[r['code']] = ns.get(r['code'])
    from foldq.codes.hal_all_codes import load_all_hal_codes
    ns = {c.name: c.num_physical for c in load_all_hal_codes() if c.num_physical <= 200}
    # Holm 校正族：所有 (code, p) 的基线 vs FoldQ 检验
    raw = []
    for nm in names:
        for p in FIXED_PS:
            b, f = ler[nm].get(f'p{p:g}_base'), ler[nm].get(f'p{p:g}_foldq')
            if b and f:
                raw.append((nm, p, _two_prop_p(b['failures'], b['shots'],
                                               f['failures'], f['shots'])))
    holm = _holm([x[2] for x in raw])
    hmap = {(nm, p): h for (nm, p, _), h in zip(raw, holm)}
    for nm in names:
        cells = []
        for p in FIXED_PS:
            b, f = ler[nm].get(f'p{p:g}_base'), ler[nm].get(f'p{p:g}_foldq')
            if not b or not f:
                cells.append('—')
                continue
            imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
            h = hmap[(nm, p)]
            mark = '*' if h < 0.05 else ''
            cells.append(f"{b['ler']:.4f}→{f['ler']:.4f}（{imp*100:+.0f}%{mark}，{h:.1e}）")
        L.append(f"| {nm} | {ns.get(nm, '?')} | " + ' | '.join(cells) + ' |')
    # 只统计改善方向
    n_pos = sum(1 for (nm, p, _), h in zip(raw, holm)
                if h < 0.05 and ler[nm][f'p{p:g}_foldq']['ler'] < ler[nm][f'p{p:g}_base']['ler'])
    n_neg = sum(1 for (nm, p, _), h in zip(raw, holm)
                if h < 0.05 and ler[nm][f'p{p:g}_foldq']['ler'] > ler[nm][f'p{p:g}_base']['ler'])
    L += ['', f'Holm 校正后：显著改善 {n_pos}，显著变差 {n_neg}，共 {len(raw)} 个检验。']
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(out, 'summary.md')}")


def summarize_xdet(records, out, cal):
    ler = {}
    for r in records:
        if r['stage'] == 'ler':
            ler.setdefault(r['code'], {})[r['cond']] = r
    names = sorted(cal, key=lambda nm: cal[nm]['num_physical'])
    raw = [(nm, _two_prop_p(ler[nm]['base_xdet']['failures'], ler[nm]['base_xdet']['shots'],
                            ler[nm]['foldq_xdet']['failures'], ler[nm]['foldq_xdet']['shots']))
           for nm in names if 'base_xdet' in ler.get(nm, {}) and 'foldq_xdet' in ler.get(nm, {})]
    holm = _holm([p for _, p in raw])
    L = ['# exp26 x_detectors=True 对照（与 exp25 主比较同配置，仅探测器口径不同）', '',
         f'p* 沿用 exp25 标定；每条件 {SHOTS} shots；Holm 校正。', '',
         '| 码 | n | 基线 LER | FoldQ LER | 改善 | p_holm | 与 exp25 改善之差 |', '|---|---|---|---|---|---|---|']
    exp25_main = {}
    try:
        for line in open(EXP25_TASKS, encoding='utf-8'):
            r = json.loads(line)
            if r.get('stage') == 'ler':
                exp25_main.setdefault(r['code'], {})[r['cond']] = r
    except OSError:
        pass
    n_pos = 0
    for (nm, p), h in zip(raw, holm):
        b, f = ler[nm]['base_xdet'], ler[nm]['foldq_xdet']
        imp = 1 - f['ler'] / b['ler'] if b['ler'] > 0 else float('nan')
        e25 = ''
        eb, ef = exp25_main.get(nm, {}).get('base'), exp25_main.get(nm, {}).get('alpha_1.5')
        if eb and ef and eb['ler'] > 0:
            e25 = f"{(imp - (1 - ef['ler'] / eb['ler'])) * 100:+.0f}pp"
        mark = '*' if h < 0.05 else ''
        n_pos += 1 if (h < 0.05 and imp > 0) else 0
        L.append(f"| {nm} | {cal[nm]['num_physical']} | {b['ler']:.4f} | {f['ler']:.4f} | "
                 f"{imp*100:+.0f}%{mark} | {h:.1e} | {e25} |")
    L += ['', f'Holm 校正后显著改善 {n_pos} / {len(raw)}。']
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(out, 'summary.md')}")


def run_audit():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for name in AUDIT_CODES:
        t0 = time.time()
        r = audit_one(name)
        r['seconds'] = time.time() - t0
        print(f"  {name:12s} 注入 {sum(r['total'])} 处：通过 {r['total'][0]}，失配 {r['total'][1]}"
              f"  ({r['seconds']:.0f}s)", flush=True)
        results.append(r)
    with open(os.path.join(OUT_DIR, 'fault_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    L = ['# exp26 单故障扫描（独立预期推导，不经 DEM）', '',
         '每码注入：数据比特 X/Z/Y × 2 个时点 + 逻辑 X 算符 + 辅助比特 X/Z。'
         '预期签名从 GF(2) 校验矩阵推导，与 Stim 采样逐位比对。', '',
         '| 码 | n | 注入处 | 通过 | 失配 |', '|---|---|---|---|---|']
    for r in results:
        L.append(f"| {r['code']} | {r['n_data']}+{r['n_anc']} | {sum(r['total'])} | "
                 f"{r['total'][0]} | {r['total'][1]} |")
    bad = sum(r['total'][1] for r in results)
    L += ['', f'总失配 {bad}。' + ('全部通过。' if bad == 0 else '失配明细见 fault_audit.json。')]
    with open(os.path.join(OUT_DIR, 'fault_audit_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(OUT_DIR, 'fault_audit_summary.md')}")


def main():
    global AUDIT_CODES
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['audit', 'fixp', 'xdet', 'summarize'], required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--audit-codes', nargs='*', default=None,
                    help='覆盖 audit 的码列表（冒烟用）')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    import stim, ldpc, scipy
    meta = {'config': {'audit_codes': a.audit_codes or AUDIT_CODES, 'fixed_ps': FIXED_PS,
                       'k_pct': K_PCT, 'alpha': ALPHA, 'shots': SHOTS, 'n_rounds': N_ROUNDS,
                       'pm_ratio': PM_RATIO, 'x_detectors_main': X_DETECTORS},
            'part': a.part,
            'python': platform.python_version(),
            'stim': stim.__version__, 'ldpc': getattr(ldpc, '__version__', '?'),
            'source_sha256_12': {s: hashlib.sha256(open(os.path.join(ROOT, s), 'rb')
                                                   .read()).hexdigest()[:12] for s in SOURCES},
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(OUT_DIR, f"meta_{a.part}_{time.strftime('%Y%m%d_%H%M%S')}.json"),
              'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    if a.audit_codes:
        AUDIT_CODES = a.audit_codes
    if a.part == 'audit':
        run_audit()
    elif a.part == 'fixp':
        run_fixp(a.workers)
    elif a.part == 'xdet':
        run_xdet(a.workers)
    elif a.part == 'summarize':
        summarize_fixp(_load_done(os.path.join(OUT_DIR, 'fixed_p', 'tasks.jsonl')),
                       os.path.join(OUT_DIR, 'fixed_p'))
        summarize_xdet(_load_done(os.path.join(OUT_DIR, 'x_detectors', 'tasks.jsonl')),
                       os.path.join(OUT_DIR, 'x_detectors'), _exp25_pstars())


if __name__ == '__main__':
    main()
