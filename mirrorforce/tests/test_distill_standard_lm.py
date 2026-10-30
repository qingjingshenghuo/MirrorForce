"""本地随机微型 Qwen 的真实标准输出、梯度和检查点测试。"""
import json
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


@pytest.fixture(params=[True, False])
def tiny_base(tmp_path, request):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    path = tmp_path / "base"
    tok = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "choose": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    fast.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}choose"
    fast.save_pretrained(path)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=4, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=request.param, eos_token_id=1, pad_token_id=2))
    model.save_pretrained(path)
    return path


def setup(monkeypatch, tiny_base, training=False):
    from mirrorforce.distill import student as module
    monkeypatch.setattr(module, "prompt_text", lambda row, tok, limit=4096: ("choose", [3, tok.convert_tokens_to_ids("<|action_000|>"), 3]))
    return module.Student(str(tiny_base), device="cpu", training=training)


def sample(n=2):
    return {"legal_actions": [{} for _ in range(n)]}


def test_full_vocab_forward_and_constrained_generation(monkeypatch, tiny_base):
    student = setup(monkeypatch, tiny_base)
    scores, tokens = student.logits([sample()])
    assert scores.shape == (1, 196) and tokens == 3
    for n in [1, 2, 192]:
        row = sample(n)
        assert student.generate_action(row) == student(row)
        assert student.select(row, "greedy")["logprob"] == 0.
        result = student.select(row, "sample", seed=7)
        assert 0 <= result["index"] < n and result["logprob"] <= 0
    assert len(set(student.ids)) == 192


def test_gradients_old_rows_and_lora(monkeypatch, tiny_base):
    student = setup(monkeypatch, tiny_base, training=True)
    optimizer = torch.optim.AdamW(student.optimizer_groups())
    before = [p.detach().clone() for p in student.row_parameters]
    lora_before = {k: p.detach().clone() for k, p in student.model.named_parameters() if "lora_" in k}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scores, _ = student.logits([sample()])
        loss = torch.nn.functional.cross_entropy(scores, torch.tensor([student.ids[1]]))
        loss.backward()
        for parameter in student.row_parameters:
            assert torch.count_nonzero(parameter.grad[:4]) == 0
        optimizer.step()
    assert all(torch.equal(old[:4], now[:4]) for old, now in zip(before, student.row_parameters))
    assert any(not torch.equal(old[student.ids], now[student.ids]) for old, now in zip(before, student.row_parameters))
    assert any(not torch.equal(lora_before[k], p) for k, p in student.model.named_parameters() if k in lora_before)


def test_reload_and_export_equivalence(monkeypatch, tiny_base, tmp_path):
    from mirrorforce.distill.student import Student
    from transformers import AutoModelForCausalLM
    student = setup(monkeypatch, tiny_base, training=True)
    opt = torch.optim.AdamW(student.optimizer_groups())
    scores, _ = student.logits([sample()])
    (-scores[0, student.ids[1]]).backward()
    opt.step()
    student.model.eval()
    before = student.logits([sample()])[0].detach()
    path = tmp_path / "adapter"
    student.save(path)
    restored = Student(adapter=path, device="cpu")
    assert restored.ids == student.ids and restored.tied == student.tied
    assert torch.allclose(before, restored.logits([sample()])[0], atol=1e-6)
    merged = tmp_path / "merged"
    restored.export_merged(merged)
    model = AutoModelForCausalLM.from_pretrained(merged).eval()
    inputs, _ = restored._inputs([sample()])
    after = model(**inputs).logits[:, -1].detach()
    assert torch.allclose(before, after, atol=1e-5)


def test_old_format_rejected(tmp_path):
    from mirrorforce.distill.student import Student
    (tmp_path / "student.json").write_text(json.dumps({"format": "old"}))
    with pytest.raises(ValueError, match="旧动作头"):
        Student(adapter=tmp_path, device="cpu")


def test_named_prompt_style_survives_checkpoint(tiny_base, tmp_path):
    from mirrorforce.distill.student import Student
    model = Student(str(tiny_base), device='cpu', prompt_style='named_menu')
    path = tmp_path / 'named'
    model.save(path)
    restored = Student(adapter=path, device='cpu')
    assert restored.prompt_style == 'named_menu' and restored.prompt_version == 'named_action_menu/v1'


def test_padded_model_vocab_does_not_collide_with_new_tokens(monkeypatch, tiny_base, tmp_path):
    from transformers import AutoModelForCausalLM
    from mirrorforce.distill.student import Student
    padded = AutoModelForCausalLM.from_pretrained(tiny_base)
    padded.resize_token_embeddings(256)
    padded.save_pretrained(tiny_base)
    student = setup(monkeypatch, tiny_base, training=True)
    assert student.original_vocab == 256 and student.base_tokenizer_size == 4
    assert student.ids[0] == 4
    scores, _ = student.logits([sample()])
    assert scores.shape[-1] == 256
    before = [p.detach().clone() for p in student.row_parameters]
    opt = torch.optim.AdamW(student.optimizer_groups())
    (-scores[0, student.ids[1]]).backward()
    opt.step()
    for old, parameter in zip(before, student.row_parameters):
        assert torch.equal(old[:4], parameter[:4])
        assert torch.equal(old[196:], parameter[196:])
    student.model.eval()
    expected = student.logits([sample()])[0].detach()
    path = tmp_path / "padded-adapter"
    student.save(path)
    restored = Student(adapter=path, device="cpu")
    assert restored.model.get_input_embeddings().num_embeddings == 256
    assert torch.allclose(expected, restored.logits([sample()])[0], atol=1e-6)


def test_context_limit_is_bounded_and_persisted(tiny_base, tmp_path):
    from mirrorforce.distill.student import Student
    with pytest.raises(ValueError, match="上下文"):
        Student(str(tiny_base), device="cpu", context_limit=513)
    student = Student(str(tiny_base), device="cpu", context_limit=256)
    path = tmp_path / "context-adapter"
    student.save(path)
    assert Student(adapter=path, device="cpu").context_limit == 256


def test_greedy_refuses_nonfinite_legal_logits(monkeypatch, tiny_base):
    student = setup(monkeypatch, tiny_base)
    scores = torch.zeros(1, student.model.config.vocab_size)
    scores[0, student.ids[0]] = float("nan")
    monkeypatch.setattr(student, "logits", lambda rows: (scores, 3))
    with pytest.raises(ValueError, match="非有限"):
        student.select(sample(), selection="greedy")


def test_eos_and_ordinary_tokens_cannot_escape_constraint(monkeypatch, tiny_base):
    student = setup(monkeypatch, tiny_base)

    def force_non_action(module, args, output):
        changed = output.clone()
        changed[..., 1] = 10000.
        changed[..., 3] = 20000.
        return changed

    hook = student.model.get_output_embeddings().register_forward_hook(force_non_action)
    try:
        assert student.generate_action(sample(1)) == 0
        assert student.generate_action(sample(2)) in (0, 1)
    finally:
        hook.remove()
