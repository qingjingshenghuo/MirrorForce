"""评测身份、失败覆盖率与统计边界。"""
import json
import pytest
from mirrorforce.distill.evaluate import summarize, main


def test_failure_is_visible_and_never_counted_as_draw():
    rows = [{"deal_seed": 1, "a_first": True, "winner": "a", "error": ""},
            {"deal_seed": 1, "a_first": False, "winner": "draw", "error": ""},
            {"deal_seed": 2, "a_first": True, "error": "context_overflow"}]
    result = summarize(rows)
    assert result["completion_rate"] == 2 / 3
    assert result["pure_win_rate"] == .5
    assert result["paired_score"] == .75
    assert result["failures"] == 1
    assert result["paired_score_ci95"] is None


def test_manifest_required_before_cross_run_comparison(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    for directory in (one, two):
        (directory / "games.jsonl").write_text("")
    with pytest.raises(ValueError, match="manifest"):
        main([str(one / "games.jsonl"), "--baseline", str(two / "games.jsonl")])
    fields = ("deck_sha256", "cards_db_sha256", "native_sha256", "teacher_identity", "pairing", "serializer_version", "selection")
    manifest = {key: "same" for key in fields}
    manifest.update(seed=300000, pairs=1)
    manifest["context_limit"] = 16384
    (one / "manifest.json").write_text(json.dumps(manifest))
    manifest["selection"] = "different"
    (two / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selection"):
        main([str(one / "games.jsonl"), "--baseline", str(two / "games.jsonl")])
