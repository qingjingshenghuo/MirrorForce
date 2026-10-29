"""缺失真实结果时不得生成看似完成的实验结论。"""
import json
from mirrorforce.distill.report import build_report


def test_missing_results_are_explicit(tmp_path):
    result = build_report(tmp_path)
    assert result["missing_evidence"]
    assert not result["ready_for_final_review"]
    assert not result["matched_actual_training_budget"]


def test_equal_limits_do_not_hide_unequal_actual_budgets(tmp_path):
    for branch, tokens in (("continued-sft", 100), ("offline-kd", 100), ("online-kd", 99)):
        for i in range(1, 4):
            path = tmp_path / f"{branch}-{i}"
            path.mkdir()
            (path / "result.json").write_text(json.dumps({"steps": 2, "effective_tokens": tokens}))
    result = build_report(tmp_path)
    assert result["training"]["online-kd"]["effective_tokens"] == 297
    assert not result["matched_actual_training_budget"]
