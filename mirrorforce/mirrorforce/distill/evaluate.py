"""按发牌对进行 bootstrap；异常局不能当成正常平局。"""
import argparse
import json
import random
from pathlib import Path


def paired_scores(rows):
    groups = {}
    for row in rows:
        if type(row.get("a_first")) is not bool or (not row.get("error") and row.get("winner") not in ("a", "b", "draw")):
            raise ValueError("对局缺少合法先后手或结局")
        group = groups.setdefault(row["deal_seed"], {})
        if row["a_first"] in group:
            raise ValueError("重复的发牌与先后手")
        group[row["a_first"]] = row
    return {seed: sum(1. if r["winner"] == "a" else .5 if r["winner"] == "draw" else 0.
                      for r in pair.values()) / 2
            for seed, pair in groups.items() if len(pair) == 2 and not any(r["error"] for r in pair.values())}


def interval(values, seed=42):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return [samples[49], samples[1949]]


def summarize(rows, planned_games=None):
    valid = [r for r in rows if not r["error"]]
    scores = paired_scores(rows)
    if planned_games is not None and (type(planned_games) is not int or planned_games < len(rows)):
        raise ValueError("计划对局数必须是整数且不能少于已观察局数")
    denominator = len(rows) if planned_games is None else planned_games
    latency = sorted(x for row in rows for x in row.get("latency_ms", []))
    return {"games": len(rows), "failures": len(rows) - len(valid),
            "planned_games": planned_games,
            "missing_games": planned_games - len(rows) if planned_games is not None else None,
            "completion_rate": len(valid) / denominator if denominator else None,
            "completion_denominator": "observed" if planned_games is None else "planned",
            "observed_completion_rate": len(valid) / len(rows) if rows else None,
            "pure_win_rate": sum(r["winner"] == "a" for r in valid) / len(valid) if valid else None,
            "wins": sum(r["winner"] == "a" for r in valid),
            "draws": sum(r["winner"] == "draw" for r in valid),
            "losses": sum(r["winner"] == "b" for r in valid),
            "complete_pairs": len(scores), "paired_score": sum(scores.values()) / len(scores) if scores else None,
            "paired_score_ci95": interval(list(scores.values())),
            "latency_p50_ms": latency[len(latency) // 2] if latency else None,
            "latency_p95_ms": latency[min(len(latency) - 1, int(len(latency) * .95))] if latency else None,
            "teacher_queries": sum(r.get("teacher_queries", 0) for r in rows)}


def manifest_plan(manifest):
    """从锁定的采样计划恢复发牌集合，不能由已完成结果反推计划。"""
    pairs, seed = manifest.get("pairs"), manifest.get("seed")
    if type(pairs) is not int or pairs <= 0 or type(seed) is not int:
        raise ValueError("manifest 必须提供正整数 pairs 和整数 seed")
    return set(range(seed, seed + pairs))


def compare(rows, baseline_rows, planned_seeds=None, baseline_planned_seeds=None):
    left, right = paired_scores(rows), paired_scores(baseline_rows)
    common = sorted(left.keys() & right.keys())
    diffs = [left[k] - right[k] for k in common]
    observed = {r["deal_seed"] for r in rows + baseline_rows}
    all_seeds = observed | set(planned_seeds or ()) | set(baseline_planned_seeds or ())
    excluded = sorted(all_seeds - set(common))
    return {"complete_common_pairs": len(common),
            "score_difference": sum(diffs) / len(diffs) if diffs else None,
            "ci95": interval(diffs), "excluded_pairs": len(excluded),
            "excluded_pair_seeds": excluded, "comparison_pair_universe": len(all_seeds),
            "observed_pair_universe": len(observed)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("games", type=Path)
    p.add_argument("--baseline", type=Path)
    a = p.parse_args(argv)
    read = lambda path: [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    rows = read(a.games)
    def manifest(path, required=False):
        file = path.parent / "manifest.json"
        if not file.exists():
            if required:
                raise ValueError("比较必须提供 manifest，不能仅凭相同种子认定公平")
            return None
        return json.loads(file.read_text(encoding="utf-8"))

    lm = manifest(a.games, required=bool(a.baseline))
    left_plan = manifest_plan(lm) if lm is not None else None
    if left_plan is not None and any(r["deal_seed"] not in left_plan for r in rows):
        raise ValueError("已观察对局的种子不在 manifest 计划内")
    out = summarize(rows, planned_games=len(left_plan) * 2 if left_plan is not None else None)
    out["planned_pair_seeds"] = sorted(left_plan) if left_plan is not None else None
    out["unobserved_pair_seeds"] = sorted(left_plan - {r["deal_seed"] for r in rows}) if left_plan is not None else None
    if a.baseline:
        fields = ("deck_sha256", "cards_db_sha256", "native_sha256", "teacher_identity", "pairing", "serializer_version", "selection", "context_limit")
        rm = manifest(a.baseline, required=True)
        right_plan = manifest_plan(rm)
        for field in fields:
            if field not in lm or field not in rm or lm[field] != rm[field]:
                raise ValueError("评测身份不一致或缺失：" + field)
        baseline_rows = read(a.baseline)
        if any(r["deal_seed"] not in right_plan for r in baseline_rows):
            raise ValueError("基线对局种子不在 manifest 计划内")
        out["comparison"] = compare(rows, baseline_rows, left_plan, right_plan)
        out["baseline_summary"] = summarize(baseline_rows, len(right_plan) * 2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
