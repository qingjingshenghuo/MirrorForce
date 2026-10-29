"""显式启用的 CUDA/BF16 训练恢复逐位一致性验证。"""
import json
import os
import pytest

torch = pytest.importorskip("torch")
from test_distill_standard_lm import tiny_base


@pytest.mark.skipif(os.environ.get("MF_DISTILL_CUDA_TEST") != "1" or not torch.cuda.is_available(),
                    reason="需要显式启用真实CUDA验证")
def test_cuda_bf16_resume_matches_continuous(monkeypatch, tiny_base, tmp_path):
    from safetensors.torch import load_file
    from mirrorforce.distill import train, student, data
    rows = [{"legal_actions": [{}, {}], "teacher_logits": [-1., 2.]}]
    def prompt(row, tokenizer, limit=4096):
        return "choose", [3, tokenizer.convert_tokens_to_ids(data.TOKENS[0]), 3]
    monkeypatch.setattr(student, "prompt_text", prompt)
    monkeypatch.setattr(data, "prompt_text", prompt)
    monkeypatch.setattr(train, "read_rows", lambda path: rows)
    source = tmp_path / "data.jsonl"
    source.write_text("真实CUDA测试数据标识")
    def execute(out, resume=None):
        args = ["--base", str(tiny_base), "--device", "cuda", "--data", str(source),
                "--out", str(out), "--objective", "sft", "--steps", "2", "--accumulate", "1",
                "--save-every", "1", "--token-budget", "100"]
        if resume:
            args += ["--resume", str(resume)]
        train.main(args)
        return out / "step-000002"
    full = execute(tmp_path / "full")
    resumed = execute(tmp_path / "resumed", tmp_path / "full/step-000001")
    for name in ["adapter_model.safetensors", "action_rows.safetensors"]:
        a,b = [load_file(str(path / name)) for path in [full, resumed]]
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)
    metrics = [json.loads(x) for x in (resumed.parent / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["cuda_peak_allocated_bytes"] > 0
