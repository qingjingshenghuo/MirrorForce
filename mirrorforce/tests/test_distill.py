"""验证信息边界、动作对齐、损失及整局配对，不依赖冠军权重。"""
import copy
import json
import random
from types import SimpleNamespace as NS
import pytest
from mirrorforce.distill.data import public_view, probabilities, ACTION_FIELDS, read_rows, SCHEMA, split_for, prompt_text, build_sample, REQUIRED_METADATA
from mirrorforce.distill.evaluate import summarize, paired_scores
from mirrorforce.distill.train import exact_token_sampler


def menu(n=2):
    return [{"act": 8 if i == 0 else 9, "phase": 0, "finish": False, "code": 456 if i == 0 else 0,
             "effect": 10010, "position": 0, "number": 0, "place": 0, "attribute": 0,
             "spec": "h1" if i == 0 else ""} for i in range(n)]


def observer(secret=123):
    hidden = NS(known=False, code=secret, attack=secret)
    shown = NS(known=True, code=456, position=1, attack=1500, defense=0, level=4, rank=0, status=0)
    board = NS(zones={(0, 2): [shown], (1, 2): [hidden], (1, 1): [shown]},
               materials={}, counters={}, turn_player=0, phase=4, deck_count=[30, 30], extra_count=[15, 15])
    return NS(board=board, seat=0)


def row(seed=100000):
    metadata = {key: "test" for key in REQUIRED_METADATA}
    metadata.update(deal_seed=seed, viewer=0, first_player="self", subdecision_id=0)
    view = public_view(observer(), menu())
    view["card_texts"] = {"456": {"name": "card", "effect_text": "effect"}}
    result = build_sample(view, [], metadata, [1., 2.], 0)
    result.update(completed=True, training_eligible=True, episode_result="games.jsonl#test")
    return result


def test_hidden_identity_and_stats_never_cross_boundary():
    assert public_view(observer(123), menu()) == public_view(observer(987), menu())
    result = public_view(observer(), menu())
    assert result["cards"][1] == {"player": 1, "location": 2, "sequence": 0, "code": 0}
    assert all(c["location"] != 1 for c in result["cards"])


@pytest.mark.parametrize("values", [[], [0.] * 193, [float("nan")], [float("inf")]])
def test_invalid_teacher_refused(values):
    with pytest.raises(ValueError):
        probabilities(values)


def test_probabilities_stable():
    assert probabilities([10000., 10000.]) == [.5, .5]


def test_split_leak_refused(tmp_path):
    a, b = row(), row()
    b["split"] = "test"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b))
    with pytest.raises(ValueError, match="相同发牌|种子"):
        read_rows(path)


def test_exact_budget_sampling():
    draw = exact_token_sampler([3, 5, 7], 16)
    for seed in range(20):
        assert sum([3, 5, 7][i] for i in draw(random.Random(seed))) == 16
    with pytest.raises(ValueError):
        exact_token_sampler([4, 8], 15)


def test_failed_and_incomplete_pairs_not_normal_results():
    rows = [{"deal_seed": 1, "a_first": True, "winner": "a", "error": ""},
            {"deal_seed": 1, "a_first": False, "winner": "b", "error": ""},
            {"deal_seed": 2, "a_first": True, "error": "timeout"}]
    out = summarize(rows)
    assert out["complete_pairs"] == 1 and out["paired_score"] == .5 and out["failures"] == 1
    with pytest.raises(ValueError):
        paired_scores(rows + rows[:1])


def test_context_overflow_never_truncates_menu():
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return json.dumps(messages)
        def encode(self, text, **kw):
            return list(text)
    sample = row()
    sample["public_history"] = [{"event": "old", "details": "x" * 10000}]
    original = copy.deepcopy(sample)
    with pytest.raises(ValueError):
        prompt_text(sample, Tokenizer(), limit=10)
    assert sample == original
    _, ids = prompt_text(sample, Tokenizer(), limit=4096)
    assert len(ids) < 4096


def test_loss_masks_illegal_and_has_correct_gradient():
    torch = pytest.importorskip("torch")
    from mirrorforce.distill.train import action_loss
    scores = torch.tensor([[0., 0., 1000.]], requires_grad=True)
    sample = row()
    loss = action_loss(scores, [sample], "kd", [0, 1])
    loss.backward()
    assert scores.grad[0, 2] == 0
    assert scores.grad[0, 1] < 0 < scores.grad[0, 0]
    same = torch.tensor([[1., 2., -1000.]])
    assert abs(action_loss(same, [sample], "kd", [0, 1]).item()) < 1e-6
    # SFT 必须在完整词表归一化，非法普通词仍贡献分母。
    hard = action_loss(scores, [sample], "sft", [0, 1])
    assert hard.item() > 999


def test_service_pending_binding_and_commit():
    from tools.mf_runtime_policy_service import Service, Session, UniformBackend
    import numpy as np
    class Client:
        def __init__(self):
            self.__dict__.update(observer().__dict__)
            self.position = 0
        def prompt(self):
            return 0, 11, menu()
        def observation(self):
            return {"x": np.array([self.position])}
        def step(self, index):
            self.position = index
            return b"answer"
    backend = UniformBackend()
    service = Service(None, backend, {}, "greedy", 1., {})
    session = Session(Client(), backend, 0)
    service.sessions["test"] = session
    pending = service._score([session])[0]
    result = service.distill_view({"session": "test", "obs_sha256": pending["obs_sha256"]})
    assert len(result["view"]["actions"]) == 2
    with pytest.raises(ValueError):
        service.distill_view({"session": "test", "obs_sha256": "old"})
    assert service.commit({"session": "test", "index": 1})["response"] == b"answer".hex()
    assert session.client.position == 1 and session.pending is None and session.first is False
