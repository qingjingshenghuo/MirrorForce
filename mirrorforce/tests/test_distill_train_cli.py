"""真实微型 Qwen 优化器更新与训练入口恢复等价。"""
import json
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_distill_standard_lm import tiny_base


@pytest.mark.parametrize('augmented', [False, True])
def test_train_entry_resume_matches_continuous(monkeypatch, tiny_base, tmp_path, augmented):
    from mirrorforce.distill import train, student, data
    rows = [{"legal_actions": [{}, {}], "teacher_logits": [-1., 2.]}]
    def prompt(row, tokenizer, limit=4096):
        return "choose", [3, tokenizer.convert_tokens_to_ids(data.TOKENS[0]), 3]
    monkeypatch.setattr(student, "prompt_text", prompt)
    monkeypatch.setattr(data, "prompt_text", prompt)
    if augmented:
        from test_distill_data_v2 import sample
        rows = [sample()]
        monkeypatch.setattr(data, 'prompt_details', lambda row, tokenizer, limit=4096, prompt_style='compact':
            (*prompt(row, tokenizer, limit), {'history_removed': 0, 'input_tokens': 3}))
    monkeypatch.setattr(train, "read_rows", lambda path: rows)
    source = tmp_path / "train.jsonl"
    source.write_text("fixture-content")
    def execute(out, steps, resume=None):
        args = ["--base", str(tiny_base), "--device", "cpu", "--data", str(source),
                "--out", str(out), "--objective", "sft", "--steps", str(steps),
                "--accumulate", "1", "--save-every", "1", "--token-budget", "100"]
        if resume:
            args += ["--resume", str(resume)]
        if augmented:
            args += ['--permute-menus', '--action-lr', '0.001']
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
    assert state["identity"]["runtime"]["deterministic_algorithms"]
    assert state['identity']['permute_menus'] == augmented
    metrics = [json.loads(line) for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["update_seconds"] > 0 and metrics[0]["input_tokens_per_second"] > 0
    # 修改数据内容即使样本数相同，也必须拒绝恢复。
    source.write_text("changed")
    with pytest.raises(ValueError, match="恢复"):
        execute(tmp_path / "bad", 3, part)


def test_checkpoint_retention_keeps_resumable_final(monkeypatch, tiny_base, tmp_path):
    from mirrorforce.distill import train, student, data
    monkeypatch.setattr(train, 'read_rows', lambda path: [{'legal_actions': [{}, {}], 'teacher_logits': [0., 1.]}])
    def prompt(row, tokenizer, limit=4096):
        return 'choose', [3, 3]
    monkeypatch.setattr(student, 'prompt_text', prompt)
    monkeypatch.setattr(data, 'prompt_text', prompt)
    source = tmp_path / 'data.jsonl'
    source.write_text('保留策略测试')
    out = tmp_path / 'run'
    train.main(['--data', str(source), '--out', str(out), '--base', str(tiny_base),
                '--device', 'cpu', '--objective', 'sft', '--steps', '2', '--accumulate', '1',
                '--save-every', '1', '--keep-last', '1'])
    assert not (out / 'step-000001').exists()
    checkpoint = out / 'step-000002'
    assert torch.load(checkpoint / 'training.pt', weights_only=False)['step'] == 2
    restored = student.Student(adapter=checkpoint, device='cpu')
    assert torch.isfinite(restored.logits([{'legal_actions': [{}, {}]}])[0]).all()
