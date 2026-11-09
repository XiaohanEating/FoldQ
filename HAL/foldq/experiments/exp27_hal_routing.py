"""
实验27：HAL 真实布线验证——marginal 选出的跨层耦合器在多层约束下能否布通、成本几何

问题：FoldQ 的跨层边（marginal，k=10%）交给 HAL 的多层布线器（A*，带层数/长度/
TSV/层切换约束），报告：可布线成功率、所需布线层数、实际路径长度、TSV/层切换、
以及"只保留布通边"后的路由 SWAP 收益保留比例。

对接方式（区别于 HAL 论文的原生用法）：HAL 原生布的是码连接图（Tanner 图）；
这里 tier 0 = 平面网格邻接边（天然平面耦合器），marginal 跨层边作为待布线边
注入，HAL place(custom_pos) 自动把非相邻边分入高层 tier 并布线。

扫描（单轴，其余默认）：
  L 轴  max_coupler_length ∈ {不限, 12, 10, 8, 6, 4}   （实际布线路径长度上限，格）
  T 轴  max_tsvs_per_coupler ∈ {不限, 2, 1}
  B 轴  max_bump_transitions_per_coupler ∈ {10, 4, 2}

运行：hal 环境，F:/FoldQ/HAL 下
  python foldq/experiments/exp27_hal_routing.py            # 6 码 × 10 配置
  python foldq/experiments/exp27_hal_routing.py --full     # 24 码 × 关键配置
结果写 F:/FoldQ/foldq/experiments/results/exp27/
"""
import argparse
import hashlib
import json
import os
import sys
import time
import threading

import psutil

HAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FOLDQ_ROOT = os.path.abspath(os.path.join(HAL_ROOT, '..'))
sys.path.insert(0, HAL_ROOT)
sys.path.insert(0, FOLDQ_ROOT)

import networkx as nx

K_PCT = 0.10
LAYOUT_SEED = 0
AUDIT6 = ['GT_12_4_2', 'GT_24_4_4', 'D4_3', 'D6_3', 'D8_3', 'GT_98_6_12']
# GT_98 的 HAL 布线单进程吃 50GB（2026-11-05 实测峰值 52.47GB），移出默认循环，
# 需单独带内存监控调参后再跑
DEFAULT_CODES = AUDIT6[:5]
MEM_LIMIT_GB = 20.0
OUT_DIR = os.path.join(FOLDQ_ROOT, 'foldq', 'experiments', 'results', 'exp27')
TMP_DIR = os.path.join(HAL_ROOT, 'foldq', 'tmp')


class MemGuard(threading.Thread):
    """后台巡检本进程 RSS，超阈值抛异常终止当前码（制度：2026-11-05 内存事故后）"""

    def __init__(self, limit_gb=MEM_LIMIT_GB):
        super().__init__(daemon=True)
        self.limit = limit_gb * (1 << 30)
        self.proc = psutil.Process(os.getpid())
        self.peak = 0
        self.abort = False

    def run(self):
        import time as _t
        while not self.abort:
            try:
                rss = self.proc.memory_info().rss
            except psutil.NoSuchProcess:
                return
            self.peak = max(self.peak, rss)
            if rss > self.limit:
                raise_for_all(rss)
            _t.sleep(1.0)   # 忙等会抢 GIL 拖慢主线程，1 秒采样足够

    def stop(self):
        self.abort = True


def raise_for_all(rss):
    print(f"!! 内存巡检触发：RSS {rss/(1<<30):.1f}GB 超过 {MEM_LIMIT_GB}GB 上限", flush=True)
    os.kill(os.getpid(), 9)

# max_tsvs_per_coupler 同时是层数护栏：布不通的边会被 HAL 无限提升到新 tier
# （实测一次创建 180 万层、内存爆到 20GB+），必须给有限值。8 层对正常布线足够宽。
CONFIGS = ([('default', None, 10, 10)]
           + [(f'L{v}', v, 8, 10) for v in (12, 10, 8, 6, 4)]
           + [('T2', None, 2, 10), ('T1', None, 1, 10)]
           + [('B4', None, 8, 4), ('B2', None, 8, 2)])


def _patch_hal():
    """HAL 对空 tier（所有边布进 base 层时高层无边）会在两处崩：
    1) route() 末尾无条件 self.save() → Tier.avg_route_length 除零；
    2) benchmark() 对空 tier 求聚合属性，同样除零/空序列。
    exp27 不落盘，且空 tier 的聚合值应为 0，做两个 monkey-patch。"""
    from hal.hal import HardwareAwareLayout
    from hal.tier import Tier
    HardwareAwareLayout.save = lambda self, *a, **k: None

    _has_edges = lambda self: bool(self.G) and self.G.number_of_edges() > 0
    _orig_arl = Tier.avg_route_length.fget
    Tier.avg_route_length = property(
        lambda self: _orig_arl(self) if _has_edges(self) else 0.0)
    Tier.max_face_switches = property(
        lambda self: (max((len(v) for v in self.switch_edges.values()), default=0)
                      if self.switch_edges else getattr(self, 'cached_max_face_switches', 0)))
    Tier.avg_face_switches = property(
        lambda self: (sum(len(v) for v in self.switch_edges.values()) / len(self.switch_edges)
                      if self.switch_edges else getattr(self, 'cached_avg_face_switches', 0.0)))


def foldq_setup(name):
    """FoldQ 侧：码 + spring 布局 + marginal 跨层边（与 exp25/26 同源协议）"""
    from foldq.codes.hal_all_codes import load_code_by_name
    from foldq.optimization.layouts import spring_layout, default_grid_size
    from foldq.optimization.crosslayer_select import candidate_pairs, select_marginal
    from foldq.routing import CouplingGraph, positions_to_phys
    code = load_code_by_name(name)
    inter = code.get_all_interactions()
    size = default_grid_size(code)
    pos_logical = spring_layout(code, size, LAYOUT_SEED)
    phys = positions_to_phys(pos_logical, size)
    g0 = CouplingGraph(size)
    k = max(1, round(K_PCT * code.num_physical))
    cands = candidate_pairs(inter, phys, g0.dist)
    cross = select_marginal(cands, g0.dist, inter, phys, k) if cands else []
    return code, inter, size, pos_logical, phys, g0, cross


def _seg_intersect(a1, a2, b1, b2):
    """两线段是否相交或接触（含端点触碰与共线重叠）。坐标 (x, y, z)。"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def on_seg(p, q, r):   # 共线前提下 p 是否在 qr 上
        return (min(q[0], r[0]) <= p[0] <= max(q[0], r[0])
                and min(q[1], r[1]) <= p[1] <= max(q[1], r[1]))

    d1, d2 = cross(b1, b2, a1), cross(b1, b2, a2)
    d3, d4 = cross(a1, a2, b1), cross(a1, a2, b2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if d1 == 0 and on_seg(a1, b1, b2):
        return True
    if d2 == 0 and on_seg(a2, b1, b2):
        return True
    if d3 == 0 and on_seg(b1, a1, a2):
        return True
    if d4 == 0 and on_seg(b2, a1, a2):
        return True
    return False


def _v3(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _cross3(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _dot3(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _seg3d_intersect(a1, a2, b1, b2):
    """两 3D 线段是否相交或接触（含端点、共线重叠）。整数坐标精确运算。

    统一处理同面段、垂直 via 与斜跨面连接段；判定与端点存储方向无关。
    """
    d1, d2, r = _v3(a1, a2), _v3(b1, b2), _v3(a1, b1)
    # 零长度段按点在线段上判断；两段均为点时闭包围盒退化为点。
    if not any(d1):
        return (not any(_cross3(_v3(b1, a1), d2))
                and all(min(b1[i], b2[i]) <= a1[i] <= max(b1[i], b2[i])
                        for i in range(3)))
    if not any(d2):
        return (not any(_cross3(r, d1))
                and all(min(a1[i], a2[i]) <= b1[i] <= max(a1[i], a2[i])
                        for i in range(3)))
    n = _cross3(d1, d2)
    nn = _dot3(n, n)
    if nn != 0:                       # 不平行：可能交于一点
        if _dot3(r, n) != 0:          # 异面
            return False
        t_num, s_num = _dot3(_cross3(r, d2), n), _dot3(_cross3(r, d1), n)
        # t = t_num/nn, s = s_num/nn；闭区间 [0,1]（含端点接触）
        for num in (t_num, s_num):
            if num < 0 or num > nn:
                return False
        return True
    # 平行：先判共线（r × d1 == 0），再判 1D 区间重叠
    if any(c != 0 for c in _cross3(r, d1)):
        return False
    # 共线后任取非零轴，使用两段各自的两个端点比较闭区间。
    axis = next(i for i in range(3) if d1[i] != 0)
    lo = max(min(a1[axis], a2[axis]), min(b1[axis], b2[axis]))
    hi = min(max(a1[axis], a2[axis]), max(b1[axis], b2[axis]))
    return lo <= hi


def _check_path_conflicts(hal):
    """独立几何验收（2026-11-07 修订二，3D 版，方向不变）：

    任意两条不同边的全部路径线段（同面段、垂直 via、斜跨面连接段统一为 3D
    线段）不得相交、接触或共线重叠；无端点豁免；同一边自身的子段不互检。
    判定只用端点坐标，与存储方向无关。
    """
    conflicts = []
    for t, tier in enumerate(hal.tiers):
        sw_map = getattr(tier, 'switch_edges', {}) or {}
        edges = []
        for orig, subedges in sw_map.items():
            segs = []
            if not subedges:
                r = tier.G.edges[orig].get('route') if orig in tier.G.edges else None
                if r and len(r) >= 2:
                    segs = list(zip(r[:-1], r[1:]))
            else:
                for se in subedges:
                    r = tier.G.edges[se].get('route') if se in tier.G.edges else None
                    if r and len(r) >= 2:
                        segs += list(zip(r[:-1], r[1:]))
            if segs:
                edges.append((tuple(orig[:2]), segs))
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                (e1, s1), (e2, s2) = edges[i], edges[j]
                for a1, a2 in s1:
                    for b1, b2 in s2:
                        if _seg3d_intersect(a1, a2, b1, b2):
                            conflicts.append({'tier': t, 'e1': e1, 'e2': e2,
                                              'kind': 'seg3d',
                                              'seg1': [list(a1), list(a2)],
                                              'seg2': [list(b1), list(b2)]})
                            break
                    else:
                        continue
                    break
        # 垂直 via 的端点占用（同格点两面的点）已由 3D 段端点覆盖：
        # via 段 [(x,y,0),(x,y,1)] 与穿过 (x,y,0) 的同面段在端点处 3D 相交
    return conflicts


def hal_route(name, size, pos_logical, phys, cross, max_len, max_tsvs, max_bumps):
    """HAL 侧：构图、place(custom_pos)、route，返回 per-edge 布线结果与聚合指标"""
    from hal.hal import HardwareAwareLayout
    from hal.settings import Settings

    # 统一硬件图（复核 5.1）：节点 = 完整方格的格点（与路由器同一编号系，绕行空间一致）；
    # 边 = 仅 marginal 跨层边。网格相邻边是平面耦合器，天然无需布线层走线，
    # 不得作为待布边传入（否则被 HAL 挤进布线层，语义错误——GT_98 曾因此把
    # 网格边推上 tier 1 产生 31 处假冲突）。
    xy = {u: (u % size, u // size) for u in range(size * size)}
    G = nx.Graph()
    G.add_nodes_from(range(size * size))
    cross_nodes = [tuple(sorted((a, b))) for a, b in cross]
    for a, b in cross_nodes:
        G.add_edge(a, b)

    settings = Settings(save=False, verbose=False, grid_size=size + 2)
    hal = HardwareAwareLayout(f'foldq_{name}', directory_path=TMP_DIR,
                              tanner_graph=G, settings=settings)
    hal.place(custom_pos=xy)
    hal.route(max_coupler_length=max_len, max_tsvs_per_coupler=max_tsvs,
              max_bump_transitions_per_coupler=max_bumps)

    # 提取每条 marginal 边的布线结果。tier>0 的图含 ('switch', i) 虚拟节点，
    # 原始边可能被拆成多段；tier.switch_edges 给出 原始边 -> 子边列表 的映射
    # （tier 0 直连边子边列表为空，route 挂在原始边上）。
    cross_set = set(cross_nodes)
    edge_res = {}
    for t, tier in enumerate(hal.tiers):
        sw_map = getattr(tier, 'switch_edges', {}) or {}
        for orig, subedges in sw_map.items():
            key = tuple(sorted(orig[:2]))
            if key not in cross_set:
                continue
            if not subedges:  # 直连：route 在原始边属性上
                route = tier.G.edges[orig].get('route')
                edge_res[key] = {'tier': t, 'routed': route is not None,
                                 'path_len': len(route) if route else 0, 'switches': 0}
            else:             # 拆段：路径长为各子边 route 之和；缺 route 的子边按未布通
                segs = [tier.G.edges[se].get('route') for se in subedges
                        if se in tier.G.edges]
                if all(r is not None for r in segs):
                    edge_res[key] = {'tier': t, 'routed': True,
                                     'path_len': sum(len(r) for r in segs),
                                     'switches': len(subedges)}
                else:
                    edge_res[key] = {'tier': t, 'routed': False,
                                     'path_len': 0, 'switches': len(subedges)}
        # 从未进入任何 tier 布线的边（约束下直接被拒）记为未布通
        for key in cross_set - set(edge_res):
            edge_res[key] = {'tier': None, 'routed': False, 'path_len': 0, 'switches': 0}
    try:
        bench = hal.benchmark().iloc[0].to_dict()
    except Exception:   # 布不通的配置下 benchmark 可能因边缺 route 属性而崩，核心指标已在 edge_res
        bench = {}
    collisions = _check_path_conflicts(hal)
    # 独立验收（复核 P0）：相交边一律不计入布通——HAL 内部报告的"布通"
    # 不作为无冲突证据，以本检查为准
    bad = set()
    for c in collisions:
        bad.add(c['e1'])
        bad.add(c['e2'])
    for k in edge_res:
        if k in bad:
            edge_res[k]['routed'] = False
            edge_res[k]['collision'] = True
    return edge_res, bench, len(hal.tiers), collisions


def routed_swap(code, inter, phys, size, cross, routed_cross):
    """FoldQ 侧：只用布通的边建耦合图，重路由得每轮 SWAP；同时给纯平面与理想全通参照"""
    from foldq.routing import CouplingGraph
    from foldq.routing.router import route_round
    def swaps(graph):
        p = dict(phys)
        total = 0
        for _ in range(3):
            r = route_round(inter, p, graph, return_ops=True)
            total += r['swaps']
            p = r['final_phys']
        return total / 3
    base = swaps(CouplingGraph(size))
    foldq_ideal = swaps(CouplingGraph(size, list(cross)))
    foldq_routed = swaps(CouplingGraph(size, routed_cross))
    return base, foldq_ideal, foldq_routed


def run(codes):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    partial = os.path.join(OUT_DIR, 'results_partial.jsonl')
    done = set()
    results = []
    if os.path.exists(partial):
        with open(partial, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done.add((r['code'], r['config']))
    guard = MemGuard()
    guard.start()
    out = open(partial, 'a', encoding='utf-8')
    for name in codes:
        code, inter, size, pos_logical, phys, g0, cross = foldq_setup(name)
        print(f"{name}: n={code.num_physical}, cross={len(cross)}", flush=True)
        for cname, max_len, max_tsvs, max_bumps in CONFIGS:
            if (name, cname) in done:
                continue
            t0 = time.time()
            try:
                edge_res, bench, n_tiers, collisions = hal_route(
                    name, size, pos_logical, phys, cross, max_len, max_tsvs, max_bumps)
                # 统一格点编号系：edge_res 的 key 与 cross 同为排序的格点对
                ok_keys = {k for k, v in edge_res.items() if v['routed']}
                routed = [c for c in cross if tuple(sorted(c)) in ok_keys]
                base, ideal, kept = routed_swap(code, inter, phys, size, cross, routed)
                rec = {'code': name, 'n': code.num_physical, 'config': cname,
                       'max_len': max_len, 'max_tsvs': max_tsvs, 'max_bumps': max_bumps,
                       'n_cross': len(cross), 'n_routed': len(ok_keys),
                       'routed_frac': len(ok_keys) / len(cross) if cross else 1.0,
                       'n_tiers': n_tiers,
                       'avg_path_len': (sum(v['path_len'] for v in edge_res.values()
                                            if v['path_len']) / max(1, len(ok_keys))),
                       'max_path_len': max((v['path_len'] for v in edge_res.values()
                                            if v['path_len']), default=0),
                       'total_switches': sum(v['switches'] for v in edge_res.values()),
                       'n_collisions': len(collisions),
                       'bench': {k2: (float(v2) if not isinstance(v2, str) else v2)
                                 for k2, v2 in bench.items()},
                       'swap_base': base, 'swap_ideal': ideal, 'swap_kept': kept,
                       'gain_ideal': 1 - ideal / base if base else 0,
                       'gain_kept': 1 - kept / base if base else 0,
                       'gain_retained': (1 - kept / base) / (1 - ideal / base)
                                        if base and ideal < base else None,
                       'seconds': time.time() - t0}
            except Exception as exc:  # 布线失败本身是实验结果
                rec = {'code': name, 'config': cname, 'error': repr(exc)[:200],
                       'seconds': time.time() - t0}
            results.append(rec)
            done.add((name, cname))
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            if 'error' in rec:
                print(f"  {cname:8s} ERROR {rec['error'][:80]}", flush=True)
            else:
                print(f"  {cname:8s} 布通 {rec['n_routed']}/{rec['n_cross']}"
                      f" tiers={rec['n_tiers']} 路径均长 {rec['avg_path_len']:.1f}"
                      f" SWAP增益 理想 {rec['gain_ideal']*100:.0f}% → 保留 {rec['gain_kept']*100:.0f}%"
                      f"  {rec['seconds']:.0f}s", flush=True)
    guard.stop()
    out.close()
    print(f"内存巡检峰值 {guard.peak/(1<<30):.1f}GB", flush=True)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    summarize(results)
    return results


def summarize(results):
    L = ['# 实验27：HAL 多层布线验证（marginal 跨层边）', '',
         f'6 码 × {len(CONFIGS)} 配置（单轴扫描），k={K_PCT:.0%}，布局 spring 种子 {LAYOUT_SEED}。',
         '布通率 = marginal 边中 HAL 布线成功的比例；SWAP 增益保留 = 布通边重路由的收益'
         '占理想（全部边）收益的比例。', '',
         '## 默认配置（无长度/TSV/层切换约束）', '',
         '| 码 | n | 边数 | 布通 | tiers | 路径均长 | 路径最长 | SWAP 基线 | 理想增益 | 保留增益 | 保留比例 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    byc = {}
    for r in results:
        byc.setdefault(r['code'], {})[r['config']] = r
    for code, cfgs in byc.items():
        r = cfgs.get('default')
        if not r or 'error' in r:
            continue
        L.append(f"| {code} | {r['n']} | {r['n_cross']} | {r['n_routed']} | {r['n_tiers']} | "
                 f"{r['avg_path_len']:.1f} | {r['max_path_len']} | {r['swap_base']:.0f} | "
                 f"{r['gain_ideal']*100:.0f}% | {r['gain_kept']*100:.0f}% | "
                 f"{(r['gain_retained'] or 0)*100:.0f}% |")
    L += ['', '## 单轴扫描（每行一个约束轴，其余默认）', '',
          '| 码 | 配置 | 布通 | tiers | 路径最长 | 保留增益 | 保留比例 |', '|---|---|---|---|---|---|---|']
    for code, cfgs in byc.items():
        for cname, *_ in CONFIGS:
            if cname == 'default':
                continue
            r = cfgs.get(cname)
            if not r or 'error' in r:
                L.append(f"| {code} | {cname} | ERROR | | | | |")
                continue
            L.append(f"| {code} | {cname} | {r['n_routed']}/{r['n_cross']} | {r['n_tiers']} | "
                     f"{r['max_path_len']} | {r['gain_kept']*100:.0f}% | "
                     f"{(r['gain_retained'] or 0)*100:.0f}% |")
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"写入 {os.path.join(OUT_DIR, 'summary.md')}")


def main():
    _patch_hal()
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='24 码（默认 audit 前 5 码）')
    ap.add_argument('--codes', nargs='*', default=None, help='显式指定码列表')
    a = ap.parse_args()
    codes = DEFAULT_CODES
    if a.codes:
        codes = a.codes
    elif a.full:
        from foldq.codes.hal_all_codes import load_all_hal_codes
        codes = [c.name for c in sorted(load_all_hal_codes(), key=lambda c: c.num_physical)
                 if c.num_physical <= 200]
    src = os.path.abspath(__file__)
    meta = {'codes': codes, 'configs': CONFIGS, 'k_pct': K_PCT, 'layout_seed': LAYOUT_SEED,
            'source_sha256_12': hashlib.sha256(open(src, 'rb').read()).hexdigest()[:12],
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    run(codes)


if __name__ == '__main__':
    main()
