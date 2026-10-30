"""比较同一固定持出盘面，报告配对指标差及按发牌聚类区间。"""
import argparse
import json
from pathlib import Path
from collections import Counter
from .benchmark import cluster_interval


def compare_audits(before, after):
    for key in ('source_sha256', 'selected_indexes', 'prompt_style', 'prompt_version', 'context_limit'):
        if key not in before or before[key] != after.get(key):
            raise ValueError(f'持出比较条件不一致：{key}')
    if len(before['records']) != len(after['records']) or not before['records']:
        raise ValueError('持出盘面数量不匹配或为空')
    differences = []
    for left, right in zip(before['records'], after['records']):
        for key in ('game_id', 'deal_seed', 'decision_id', 'state_sha256', 'target', 'legal_count'):
            if key not in left or left[key] != right.get(key):
                raise ValueError(f'盘面或标签不一致：{key}')
        differences.append({'game_id': left['game_id'], 'deal_seed': left['deal_seed'],
            'agreement_delta': right['agreement'] - left['agreement'],
            'kl_delta': right['kl'] - left['kl'],
            'menu_stable_delta': right['menu_stable'] - left['menu_stable']})
    result = {'samples': len(differences), 'deal_clusters': len({r['deal_seed'] for r in differences}),
              'before_adapter_sha256': before['adapter_sha256'], 'after_adapter_sha256': after['adapter_sha256'],
              'direction': 'agreement及menu_stable正值为改善；kl负值为改善',
              'before_chosen_counts': dict(Counter(r['chosen'] for r in before['records'])),
              'after_chosen_counts': dict(Counter(r['chosen'] for r in after['records']))}
    for key in ('agreement_delta', 'kl_delta', 'menu_stable_delta'):
        result[key] = sum(row[key] for row in differences) / len(differences)
        result[key + '_ci95_by_deal'] = cluster_interval(differences, key)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_audits(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
