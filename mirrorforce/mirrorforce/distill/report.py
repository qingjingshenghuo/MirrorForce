"""汇总实际实验预算与对局结果；缺证据时明确标记未完成。"""
import argparse
import json
from pathlib import Path
from .evaluate import compare, manifest_plan


def build_report(root):
    root = Path(root)
    missing, training, evaluations = [], {}, {}
    def read(path):
        if not path.is_file():
            missing.append(str(path.relative_to(root)))
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    for branch in ("continued-sft", "offline-kd", "online-kd"):
        results = [read(root / f"{branch}-{i}" / "result.json") for i in range(1, 4)]
        if all(item is not None for item in results):
            training[branch] = {"steps": sum(r["steps"] for r in results),
                                "effective_tokens": sum(r["effective_tokens"] for r in results)}
    matched = (len(training) == 3 and all(r["steps"] > 0 and r["effective_tokens"] > 0 for r in training.values())
               and len({(r["steps"], r["effective_tokens"]) for r in training.values()}) == 1)
    manifests, games = {}, {}
    for branch in ("sft", "continued-sft", "offline-kd", "online-kd"):
        directory = root / f"eval-{branch}"
        evaluations[branch] = read(directory / "summary.json")
        manifests[branch] = read(directory / "manifest.json")
        path = directory / "games.jsonl"
        if path.is_file():
            games[branch] = [json.loads(line) for line in path.read_text().splitlines() if line]
        else:
            missing.append(str(path.relative_to(root)))
    comparison, identity_matches = None, False
    if all(manifests[k] is not None and k in games for k in ("offline-kd", "online-kd")):
        left, right = manifests["online-kd"], manifests["offline-kd"]
        keys = ("deck_sha256", "cards_db_sha256", "native_sha256", "teacher_identity", "pairing", "serializer_version", "selection", "context_limit")
        identity_matches = all(k in left and k in right and left[k] == right[k] for k in keys)
        if identity_matches:
            comparison = compare(games["online-kd"], games["offline-kd"], manifest_plan(left), manifest_plan(right))
    evaluations_complete = all(manifests[k] is not None and k in games
        and len(games[k]) == len(manifest_plan(manifests[k])) * 2 for k in manifests)
    return {"schema": "mirrorforce_experiment_summary/v1", "missing_evidence": missing,
            "training": training, "matched_actual_training_budget": matched,
            "evaluation_identity_matches": identity_matches, "evaluations": evaluations,
            "online_vs_offline": comparison, "costs": read(root / "costs.json"),
            "planned_evaluations_observed": evaluations_complete,
            "ready_for_final_review": not missing and matched and identity_matches and evaluations_complete,
            "note": "汇总可供人工逐项验收；不是完整项目或模型能力已经达标的自动声明"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(argv)
    report = build_report(a.root)
    (a.root / "experiment-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_final_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
