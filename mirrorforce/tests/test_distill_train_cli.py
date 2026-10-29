"""真实微型 Qwen 优化器更新与训练入口恢复等价。"""
import json
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_distill_standard_lm import tiny_base


def test_train_entry_resume_matches_continuous(monkeypatch, tiny_base, tmp_path):
    from mirrorforce.distill import train, student, data
    rows = [{"legal_actions": [{}, {}], "teacher_logits": [-1., 2.]}]
    def prompt(row, tokenizer, limit=4096):
        return "choose", [3, tokenizer.convert_tokens_to_ids(data.TOKENS[0]), 3]
    monkeypatch.setattr(student, "prompt_text", prompt)
    monkeypatch.setattr(data, "prompt_text", prompt)
    monkeypatch.setattr(train, "read_rows", lambda path: rows)
    source = tmp_path / "train.jsonl"
    source.write_text("fixture-content")
    def execute(out, steps, resume=None):
        args = ["--base", str(tiny_base), "--device", "cpu", "--data", str(source),
                "--out", str(out), "--objective", "sft", "--steps", str(steps),
                "--accumulate", "1", "--save-every", "1", "--token-budget", "100"]
        if resume:
            args += ["--resume", str(resume)]
        train.main(args)
        return json.loads((out / "result.json").read_text())["checkpoint"]
    full = execute(tmp_path / "full", 2)
    part = execute(tmp_path / "partial", 1)
    resumed = execute(tmp_path / "resumed", 2, part)
    a = student.Student(adapter=full, device="cpu")
    b = student.Student(adapter=resumed, device="cpu")
    assert torch.allclose(a.logits(rows)[0], b.logits(rows)[0], atol=1e-6)
    state = torch.load(tmp_path / "resumed" / "step-000002" / "training.pt", weights_only=False)
    assert state["step"] == 2 and state["tokens"] == 6
    # 修改数据内容即使样本数相同，也必须拒绝恢复。
    source.write_text("changed")
    with pytest.raises(ValueError, match="恢复"):
        execute(tmp_path / "bad", 3, part)
