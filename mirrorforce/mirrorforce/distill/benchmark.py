"""固定的分层盘面抽样与按对局分组的统计工具。"""
import json
import random
from collections import defaultdict
from .data import digest


def sample_states(rows, limit=200, seed=42, per_game=20, per_turn=3):
    """先去重，再轮流抽取阶段/菜单类型；限制单局和单回合贡献。"""
    if min(limit, per_game, per_turn) <= 0:
        raise ValueError('抽样限额必须为正数')
    rng, groups, seen = random.Random(seed), defaultdict(list), set()
    for index, row in enumerate(rows):
        view = row.get('view') or row.get('observation', {})
        key = digest({'observation': row.get('observation', view),
                      'legal_actions': row['legal_actions'], 'cards': row.get('card_dictionary', {})})
        if key in seen:
            continue
        seen.add(key)
        group = (row.get('first_player'), str(view.get('phase')), str(view.get('prompt_msg')))
        groups[group].append(index)
    buckets = list(groups.values())
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
    games, turns, selected = defaultdict(int), defaultdict(int), []
    while buckets and len(selected) < limit:
        remaining = []
        for bucket in buckets:
            while bucket:
                index = bucket.pop()
                row = rows[index]
                game = row['game_id']
                turn = (game, (row.get('view') or row.get('observation', {})).get('turn'))
                if games[game] >= per_game or turns[turn] >= per_turn:
                    continue
                games[game] += 1
                turns[turn] += 1
                selected.append(index)
                break
            if bucket:
                remaining.append(bucket)
            if len(selected) >= limit:
                break
        buckets = remaining
    return selected


def cluster_interval(records, field, seed=42):
    """按完整对局重采样，不能将同局盘面当作独立样本。"""
    groups = defaultdict(list)
    for row in records:
        groups[row.get('deal_seed', row['game_id'])].append(row[field])
    values = list(groups.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(2000):
        sampled = [v for group in rng.choices(values, k=len(values)) for v in group]
        samples.append(sum(sampled) / len(sampled))
    samples.sort()
    return [samples[49], samples[1949]]


def main(argv=None):
    """在训练前生成固定盘面文件及来源清单。"""
    import argparse
    import hashlib
    from pathlib import Path
    from .data import read_rows
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-game', type=int, default=20)
    p.add_argument('--per-turn', type=int, default=3)
    a = p.parse_args(argv)
    manifest = Path(str(a.out) + '.manifest.json')
    if a.out.exists() or manifest.exists():
        raise ValueError('固定评测文件已存在，禁止覆盖')
    rows = read_rows(a.data, a.split)
    indexes = sample_states(rows, a.limit, a.seed, a.per_game, a.per_turn)
    if not indexes:
        raise ValueError('没有可用持出盘面')
    payload = ''.join(json.dumps(rows[i], ensure_ascii=False) + '\n' for i in indexes)
    record = {**vars(a), 'selected_indexes': indexes, 'samples': len(indexes),
        'games': sorted({rows[i]['game_id'] for i in indexes}),
        'source_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256(payload.encode()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(payload, encoding='utf-8')
    manifest.write_text(json.dumps(record, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, default=str, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
