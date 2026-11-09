"""Export source-linked numerical tables; no new sampling."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived'
SOURCES = {}


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding='utf8')
    OUT.mkdir(parents=True, exist_ok=True)
    original = read('foldq/experiments/results/exp15/results.json')['results']
    random = {r['code']: r for r in read('foldq/experiments/results/exp15b/single_round_check.json')}
    assert len(original) == len(random) == 105
    selection = []
    for r in sorted(original, key=lambda r: (r['n'], r['code_name'])):
        code = r['code_name']
        base = r['baseline_greedy']['swaps']
        marg = r['marginal_greedy']['swaps']
        rand = random[code]['single_round_mean']
        selection.append(dict(code=code, N=r['n'], k=0.1, baseline_swaps=base,
                              marginal_swaps=marg, random_single_round_mean=rand,
                              longest_swaps=r['longest_greedy']['swaps'],
                              marginal_gain=1-marg/base, random_gain=1-rand/base,
                              longest_gain=1-r['longest_greedy']['swaps']/base,
                              marginal_vs_random_difference_over_base=(rand-marg)/base,
                              marginal_better_than_random_mean=marg < rand))
    write_csv('selection_105.csv', selection)
    depth = []
    for r in sorted(original, key=lambda r: (r['n'], r['code_name'])):
        for compiler, definition in [('greedy', 'ASAP_CX1_SWAP3'), ('sabre', 'DAG_operations_SWAP_unexpanded')]:
            b, f = r['baseline_' + compiler], r['marginal_' + compiler]
            assert b['swaps'] > 0 and b['depth'] > 0
            depth.append(dict(code=r['code_name'], N=r['n'], k=.1, compiler=compiler,
                              depth_definition=definition, baseline_swaps=b['swaps'], folded_swaps=f['swaps'],
                              swap_gain=1-f['swaps']/b['swaps'], baseline_depth=b['depth'],
                              folded_depth=f['depth'], depth_gain=1-f['depth']/b['depth']))
    write_csv('architecture_depth_105.csv', depth)
    audit = 'data/audits/'
    scan = read(audit + 'scan_replay.json')
    summary = read(audit + 'scan_summary.json')
    geometry = {(r['code'], r['k_pct']): r for r in scan}
    assert len(geometry) == 96
    budget = [dict(k=r['k'], selected=r['selected'], accepted=r['routed'],
                   edge_weighted_route_rate=r['routed']/r['selected'],
                   fully_routed_codes=r['full_codes'], total_codes=24,
                   mean_per_code_swap_gain=r['mean_gain_kept']) for r in summary['k_aggregates']]
    write_csv('budget_aggregate.csv', budget)
    write_csv('budget_per_code.csv', [dict(code=r['code'], N=r['n_initial'], M=r['grid_sites'],
                                          k=r['k_pct'], budget=max(1, round(r['k_pct']*r['n_initial'])),
                                          selected=r['selected'], accepted=r['accepted'],
                                          baseline_swaps=r['baseline_swaps'], accepted_swaps=r['accepted_swaps'],
                                          gain=r['gain_kept']) for r in scan])
    comparison = read(audit + 'statistics.json')['exp33']
    statistics = {(r['key'][0], r['key'][1]): r for r in comparison}
    physical = read('foldq/experiments/results/exp33/tasks.jsonl')
    base = {r['code']: r for r in read('foldq/experiments/results/exp28b/tasks.jsonl') if r['cond'] == 'base'}
    rows = []
    for r in sorted(physical, key=lambda r: (geometry[(r['code'], r['k_pct'])]['n_initial'], r['code'], r['k_pct'])):
        key = r['code'], r['k_pct']
        g, stat, b = geometry[key], statistics[key], base[r['code']]
        assert {tuple(sorted(e)) for e in r['routed_edges']} == {tuple(sorted(e)) for e in g['accepted_edges']}
        rows.append(dict(code=r['code'], N=g['n_initial'], M=g['grid_sites'], k=r['k_pct'],
                         budget=max(1, round(r['k_pct']*g['n_initial'])), selected=r['n_cross'], accepted=r['n_routed'],
                         baseline_failures=b['failures'], baseline_shots=b['shots'], baseline_ler=b['ler'],
                         baseline_ci_low=b['ci95'][0], baseline_ci_high=b['ci95'][1],
                         failures=r['failures'], shots=r['shots'], ler=r['ler'],
                         ci_low=r['ci95'][0], ci_high=r['ci95'][1],
                         ler_improvement=stat['improvement'], p_holm_48=stat['p_holm'],
                         significant_improvement=stat['p_holm'] < .05 and stat['improvement'] > 0))
    assert len(rows) == 48
    write_csv('physical_ler_all48.csv', rows)
    case_keys = [(r['code'], .05) for r in rows if r['k'] == .05 and r['significant_improvement']]
    case_keys += [('GT_96_4_12', .02), ('GT_96_4_12', .05), ('GT_98_6_12', .05)]
    assert len(case_keys) == len(set(case_keys)) == 9
    bykey = {(r['code'], r['k']): r for r in rows}
    write_csv('workpoint_cases.csv', [bykey[k] for k in case_keys])
    main_keys = [('GT_48_4_8', .05), ('GT_72_8_8', .05), ('GT_96_4_12', .02), ('GT_98_6_12', .05)]
    write_csv('main_workpoints4.csv', [bykey[k] for k in main_keys])
    ideal = {r['code']: r for r in read('foldq/experiments/results/exp28b/tasks.jsonl') if r['cond'] == 'foldq_l6'}
    ideal_audit = read('data/audits/ler_comparisons.json')['comparisons']['exp28b:base:foldq_l6']
    ideal_stats = {r['code']: r for r in ideal_audit['rows']}
    main_ler = []
    for code in sorted(base, key=lambda code: (geometry[(code, .05)]['n_initial'], code)):
        b, f, s = base[code], ideal[code], ideal_stats[code]
        assert (b['failures'], b['shots'], f['failures'], f['shots']) == (s['base_failures'], s['base_shots'], s['treatment_failures'], s['treatment_shots'])
        g = geometry[(code, .1)]
        main_ler.append(dict(code=code, N=g['n_initial'], M=g['grid_sites'], graph='selected', k=.1, L=6,
                             baseline_failures=b['failures'], baseline_shots=b['shots'], baseline_ler=b['ler'],
                             baseline_ci_low=b['ci95'][0], baseline_ci_high=b['ci95'][1],
                             failures=f['failures'], shots=f['shots'], ler=f['ler'], ci_low=f['ci95'][0], ci_high=f['ci95'][1],
                             ler_improvement=s['improvement'], p_holm=s['p_holm'], holm_family='selected_L6_24',
                             significant_improvement=s['p_holm'] < .05 and s['improvement'] > 0))
        for k in [.02, .05]:
            r = bykey[(code, k)]
            main_ler.append(dict(code=code, N=r['N'], M=r['M'], graph='accepted', k=k, L=6,
                                 **{name: r[name] for name in ['baseline_failures', 'baseline_shots', 'baseline_ler',
                                 'baseline_ci_low', 'baseline_ci_high', 'failures', 'shots', 'ler', 'ci_low', 'ci_high', 'ler_improvement']},
                                 p_holm=r['p_holm_48'], holm_family='accepted_both_budgets_48',
                                 significant_improvement=r['significant_improvement']))
    assert len(main_ler) == 72 and len({r['code'] for r in main_ler}) == 24
    assert sum(r['significant_improvement'] for r in main_ler if r['graph'] == 'selected') == 21
    write_csv('main_ler72.csv', main_ler)
    facts = dict(selection_codes=105, marginal_mean_gain=sum(r['marginal_gain'] for r in selection)/105,
                 random_mean_gain=sum(r['random_gain'] for r in selection)/105,
                 marginal_wins=sum(r['marginal_better_than_random_mean'] for r in selection),
                 longest_mean_gain=sum(r['longest_gain'] for r in selection)/105,
                 architecture={compiler: dict(codes=105,
                     mean_swap_gain=sum(r['swap_gain'] for r in depth if r['compiler']==compiler)/105,
                     mean_depth_gain=sum(r['depth_gain'] for r in depth if r['compiler']==compiler)/105,
                     negative_depth_codes=[r['code'] for r in depth if r['compiler']==compiler and r['depth_gain']<0],
                     depth_definition=next(r['depth_definition'] for r in depth if r['compiler']==compiler))
                     for compiler in ['greedy', 'sabre']},
                 physical_comparisons=48, geometry_conditions=96,
                 significant_by_budget={str(k): sum(r['significant_improvement'] for r in rows if r['k']==k) for k in [.02,.05]})
    assert facts['marginal_wins'] == 103
    assert facts['significant_by_budget'] == {'0.02': 4, '0.05': 6}
    (OUT/'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2)+chr(10), encoding='utf8')
    manifest = {'sources_sha256': SOURCES, 'export_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in OUT.glob('*.csv')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+chr(10), encoding='utf8')
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
