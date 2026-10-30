"""持出样本的 KL、动作一致率及菜单重排一致率。"""
import argparse
import copy
import json
import random
from .data import read_rows, probabilities, TOKENS, digest


def permute_menu(row, order):
    """仅做离线鲁棒性评测；返回新菜单及其同步监督，不执行虚拟索引。"""
    n = len(row["legal_actions"])
    if len(order) != n or any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
        raise ValueError("动作置换必须是完整整数排列")
    changed = copy.deepcopy(row)
    changed["legal_actions"] = [{**row["legal_actions"][old], "token": TOKENS[new]}
                                for new, old in enumerate(order)]
    if row.get("view") is not None:
        changed["view"]["actions"] = [row["view"]["actions"][i] for i in order]
        changed["menu_sha256"] = digest(changed["view"]["actions"])
    changed["teacher_logits"] = [row["teacher_logits"][i] for i in order]
    changed["action_mapping"] = [{"token": TOKENS[i], "native_index": i} for i in range(n)]
    if "chosen_index" in row:
        chosen = order.index(row["chosen_index"])
        changed.update(chosen=chosen, chosen_index=chosen, chosen_token=TOKENS[chosen])
    return changed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument('--context-limit', type=int, help='正式持出盘面使用16384；必须在比较两侧保持一致')
    p.add_argument('--prompt-style', choices=['compact', 'named_menu'])
    p.add_argument("--out", help="保存逐样本结果及固定抽样清单，不覆盖已有文件")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-game", type=int, default=20)
    p.add_argument("--per-turn", type=int, default=3)
    a = p.parse_args(argv)
    from .student import Student
    from .train import action_loss
    from .collect import content_identity
    from .stages import source_identity
    student = Student(adapter=a.adapter, device=a.device, prompt_style=a.prompt_style, context_limit=a.context_limit)
    from .benchmark import sample_states, cluster_interval
    from pathlib import Path
    source = read_rows(a.data, a.split)
    indexes = sample_states(source, a.limit, a.seed, a.per_game, a.per_turn)
    rows = [source[i] for i in indexes]
    if not rows:
        raise ValueError("没有持出样本")
    rng, kl, agree, stable = random.Random(42), 0., 0, 0
    records = []
    with student.torch.no_grad():
        for row in rows:
            scores, _ = student.logits([row])
            n = len(row["teacher_logits"])
            chosen = int(scores[0, student.ids[:n]].argmax())
            loss = action_loss(scores, [row], "kd", student.ids).item()
            target = max(range(n), key=row["teacher_logits"].__getitem__)
            agreement = chosen == target
            kl += loss
            agree += agreement
            order = list(range(n))
            rng.shuffle(order)
            if order == list(range(n)):
                order = order[1:] + order[:1]
            changed = permute_menu(row, order)
            reordered_choice = order[student(changed)]
            stable += reordered_choice == chosen
            records.append({'game_id': row['game_id'], 'kl': loss, 'agreement': int(agreement),
                            'deal_seed': row.get('deal_seed', row['game_id']),
                            'decision_id': row.get('decision_id'), 'chosen': chosen, 'target': target,
                            'legal_count': n, 'target_kind': row['legal_actions'][target].get('kind'),
                            'reordered_choice': reordered_choice, 'menu_stable': int(reordered_choice == chosen),
                            'state_sha256': digest({key: row[key] for key in ('observation', 'card_dictionary', 'public_history', 'legal_actions')})})
    result = {"samples": len(rows), "teacher_kl": kl / len(rows),
        "teacher_agreement": agree / len(rows), "menu_permutation_agreement": stable / len(rows),
        "game_count": len({r['game_id'] for r in rows}), 'selection_seed': a.seed,
        'prompt_style': student.prompt_style, 'prompt_version': student.prompt_version,
        'context_limit': student.context_limit,
        'adapter_sha256': content_identity(a.adapter), **source_identity(),
        'selected_indexes': indexes, 'source_sha256': __import__('hashlib').sha256(Path(a.data).read_bytes()).hexdigest(),
        'agreement_ci95_by_game': cluster_interval(records, 'agreement'),
        'kl_ci95_by_game': cluster_interval(records, 'kl'), 'records': records}
    if a.out:
        with Path(a.out).open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
