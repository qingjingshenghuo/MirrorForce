"""计划覆盖率与失败发牌对不能在统计中消失。"""
import json
import pytest
from mirrorforce.distill.evaluate import summarize, compare, main, manifest_plan


def game(seed, first=True, winner="a", error=""):
    return {"deal_seed": seed, "a_first": first, "winner": winner, "error": error}


def test_planned_and_observed_completion_are_distinct():
    rows = [game(1), game(1, False, error="timeout")]
    out = summarize(rows, planned_games=6)
    assert out["completion_rate"] == 1 / 6
    assert out["observed_completion_rate"] == .5
    assert out["missing_games"] == 4
    assert out["failures"] == 1 and out["complete_pairs"] == 0
    assert summarize(rows)["completion_denominator"] == "observed"


def test_empty_results_still_show_plan():
    result = summarize([], planned_games=100)
    assert result["completion_rate"] == 0 and result["missing_games"] == 100
    assert result["observed_completion_rate"] is None
    assert result["paired_score_ci95"] is None


@pytest.mark.parametrize("planned", [-1, 1, 2.5, True])
def test_bad_plan_refused(planned):
    with pytest.raises(ValueError):
        summarize([game(1), game(1, False)], planned)


def test_shared_failures_and_unobserved_pairs_are_excluded():
    rows = [game(1), game(1, False), game(2, error="failure")]
    baseline = [game(1, winner="b"), game(1, False), game(2, error="failure")]
    out = compare(rows, baseline, {1, 2, 3}, {1, 2, 3})
    assert out["score_difference"] == .5
    assert out["excluded_pairs"] == 2 and out["excluded_pair_seeds"] == [2, 3]
    assert out["complete_common_pairs"] == 1 and out["ci95"] is None


def test_all_failed_without_plan_still_counts_observed_seeds():
    result = compare([game(7, error="timeout")], [game(7, error="timeout")])
    assert result["excluded_pairs"] == 1
    assert result["score_difference"] is None


def write_result(tmp_path, name, rows, **changes):
    root = tmp_path / name
    root.mkdir()
    manifest = {"seed": 1, "pairs": 3, "deck_sha256": "deck", "cards_db_sha256": "db",
        "native_sha256": "native", "teacher_identity": {"id": "teacher"}, "pairing": "paired",
        "serializer_version": "v2", "selection": "greedy", "context_limit": 16384, **changes}
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = root / "games.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_cli_uses_manifest_and_reports_missing(tmp_path, capsys):
    path = write_result(tmp_path, "left", [game(1), game(1, False)])
    baseline = write_result(tmp_path, "right", [game(1), game(1, False), game(2, error="failure")])
    main([str(path), "--baseline", str(baseline)])
    out = json.loads(capsys.readouterr().out)
    assert out["completion_rate"] == 2 / 6
    assert out["missing_games"] == 4 and out["unobserved_pair_seeds"] == [2, 3]
    assert out["comparison"]["excluded_pairs"] == 2
    assert out["baseline_summary"]["missing_games"] == 3


def test_cli_rejects_incomparable_identity(tmp_path):
    left = write_result(tmp_path, "left", [game(1)])
    right = write_result(tmp_path, "right", [game(1)], selection="sample")
    with pytest.raises(ValueError, match="selection"):
        main([str(left), "--baseline", str(right)])


def test_cli_rejects_out_of_plan_seed(tmp_path):
    path = write_result(tmp_path, "left", [game(10)])
    with pytest.raises(ValueError, match="种子"):
        main([str(path)])


def test_manifest_plan_requires_explicit_valid_schedule():
    assert manifest_plan({"seed": 1, "pairs": 2}) == {1, 2}
    with pytest.raises(ValueError):
        manifest_plan({"seed": 1})
