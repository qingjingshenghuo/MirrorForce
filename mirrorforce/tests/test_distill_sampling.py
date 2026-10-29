"""采样协议模拟测试：不把模拟教师结果视为真实模型证据。"""
from types import SimpleNamespace as NS
import pytest

from mirrorforce.distill import policy as P
from mirrorforce.distill.collect import content_identity
from mirrorforce.distill.data import TOKENS, digest
from mirrorforce.netduel import agent_wire as W


class Chooser:
    def __init__(self, index=1):
        self.index, self.calls = index, []
    def select(self, row, **kwargs):
        self.calls.append(kwargs)
        return {"index": self.index, "token": TOKENS[self.index], "logprob": -0.5,
                "selection": kwargs["selection"], "temperature": kwargs["temperature"]}


def make(chooser=None):
    policy = P.DistillPolicy(None, {"protocol": W.PROTOCOL}, seed=42, chooser=chooser,
                            texts=NS(enrich=lambda raw: raw), metadata={})
    policy.active, policy.session, policy.seat = True, "session", 0
    policy._client = NS(_lp=[8000, 8000], result=NS(turns=1))
    policy.pending = [(11, b"\x00")]
    policy._make_row = lambda raw, logits, sha, sub: {"subdecision_id": sub, "teacher_logits": logits,
        "legal_actions": [{"token": TOKENS[i], "card_code": i} for i in range(len(logits))]}
    return policy


def test_sampling_and_greedy_modes_propagate():
    chooser = Chooser()
    policy = make(chooser)
    assert policy._choose({}, [1., 2.])["index"] == 1
    assert chooser.calls[-1]["selection"] == "sample"
    policy.selection = "greedy"
    policy._choose({}, [1., 2.])
    assert chooser.calls[-1]["selection"] == "greedy"
    assert type(chooser.calls[-1]["seed"]) is int


def test_single_action_skips_model():
    chooser = Chooser()
    assert make(chooser)._choose({}, [0.])["index"] == 0
    assert not chooser.calls


def test_invalid_student_never_falls_back():
    policy = make(Chooser(3))
    with pytest.raises(ValueError, match="禁止教师接管"):
        policy._choose({}, [1., 2.])


def test_teacher_demonstration_is_deterministic():
    chosen = make()._choose({}, [2., 2., 1.])
    assert chosen["index"] == 0 and chosen["selection"] == "greedy" and chosen["logprob"] == 0


def test_multi_subchoices_commit_the_actual_student_action(monkeypatch):
    policy = make()
    monkeypatch.setattr(P, "parse_select", lambda *args: NS(cards=[]))
    calls = []
    pending = {"msg": 11, "rows": 2, "logits": [0., 1.], "obs_sha256": "current"}
    raw = {"actions": [{"code": 0}, {"code": 1}]}
    def call(request):
        calls.append(request)
        if request["op"] == "prompt":
            return {"pending": pending}
        if request["op"] == "distill_view":
            return {"view": raw, "obs_sha256": "current", "menu_sha256": digest(raw["actions"])}
        if len([r for r in calls if r["op"] == "commit"]) == 1:
            return {"pending": {**pending, "logits": [1., 0.]}}
        return {"response": "0100"}
    policy._call = call
    result = policy._parse(11, b"\x00", None)
    commits = [r for r in calls if r["op"] == "commit"]
    assert [r["index"] for r in commits] == [1, 0]
    assert all(r["obs_sha256"] == "current" and r["menu_sha256"] == digest(raw["actions"]) for r in commits)
    assert [row["subdecision_id"] for row in policy.samples] == [0, 1]
    assert result.auto_response == b"\x01\x00"


def test_stale_view_is_not_executed(monkeypatch):
    policy = make()
    monkeypatch.setattr(P, "parse_select", lambda *args: NS(cards=[]))
    calls = []
    def call(req):
        calls.append(req)
        if req["op"] == "prompt":
            return {"pending": {"msg": 11, "rows": 2, "logits": [0., 1.], "obs_sha256": "new"}}
        return {"obs_sha256": "old", "view": {"actions": [{}, {}]}, "menu_sha256": "wrong"}
    policy._call = call
    with pytest.raises(ValueError, match="过期"):
        policy._parse(11, b"\x00", None)
    assert not any(r["op"] == "commit" for r in calls)
    assert policy.failures


def test_content_identity_uses_file_bytes_not_path(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    for path in (left, right):
        (path / "weights").write_bytes(b"same")
    assert content_identity(left) == content_identity(right)
    (right / "weights").write_bytes(b"changed")
    assert content_identity(left) != content_identity(right)


def test_service_rejects_stale_commit_without_advancing():
    from tools.mf_runtime_policy_service import Service, Session, UniformBackend
    backend = UniformBackend()
    class Client:
        def __init__(self):
            self.stepped = []
        def prompt(self):
            return 0, 11, [{"code": 1}, {"code": 2}]
        def step(self, index):
            self.stepped.append(index)
            return b"done"
    client = Client()
    session = Session(client, backend, 0)
    session.pending = {"rows": 2, "obs_sha256": "current"}
    service = Service(None, backend, {}, "greedy", 1., {})
    service.sessions["session"] = session
    for index in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            service.commit({"session": "session", "index": index})
    with pytest.raises(ValueError, match="过期"):
        service.commit({"session": "session", "index": 1, "obs_sha256": "old", "menu_sha256": "old"})
    assert client.stepped == [] and session.pending is not None
    result = service.commit({"session": "session", "index": 1, "obs_sha256": "current",
                             "menu_sha256": digest(client.prompt()[2])})
    assert result["response"] == b"done".hex() and client.stepped == [1]
