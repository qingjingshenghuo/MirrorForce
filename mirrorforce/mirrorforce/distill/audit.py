"""持出样本的 KL、动作一致率及菜单重排一致率。"""
import argparse
import copy
import json
import random
from .data import read_rows, probabilities, TOKENS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    from .student import Student
    from .train import action_loss
    student = Student(adapter=a.adapter, device=a.device)
    rows = read_rows(a.data, a.split)[:a.limit]
    if not rows:
        raise ValueError("没有持出样本")
    rng, kl, agree, stable = random.Random(42), 0., 0, 0
    with student.torch.no_grad():
        for row in rows:
            scores, _ = student.logits([row])
            n = len(row["teacher_logits"])
            chosen = int(scores[0, student.ids[:n]].argmax())
            kl += action_loss(scores, [row], "kd", student.ids).item()
            agree += chosen == max(range(n), key=row["teacher_logits"].__getitem__)
            order = list(range(n))
            rng.shuffle(order)
            changed = copy.deepcopy(row)
            changed["view"]["actions"] = [row["view"]["actions"][i] for i in order]
            if "legal_actions" in row:
                changed["legal_actions"] = [{**row["legal_actions"][old], "token": TOKENS[new],
                    "native_index": new} for new, old in enumerate(order)]
            changed["teacher_logits"] = [row["teacher_logits"][i] for i in order]
            stable += order[student(changed)] == chosen
    print(json.dumps({"samples": len(rows), "teacher_kl": kl / len(rows),
        "teacher_agreement": agree / len(rows), "menu_permutation_agreement": stable / len(rows)}, indent=2))


if __name__ == "__main__":
    main()
