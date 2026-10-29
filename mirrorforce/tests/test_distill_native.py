"""L1：真实原生游戏，uniform测试教师与随机学生；不产生正式训练数据。"""
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import tempfile
from types import SimpleNamespace
import pytest


def test_native_distillation_and_independent_student(tmp_path):
    bundle = Path(os.environ.get("MF_DISTILL_TEST_ASSETS", "/not-configured"))
    native = Path(os.environ.get("MF_DUEL_NATIVE", "/not-configured"))
    if not (bundle / "cards.cdb").is_file() or not native.is_file():
        pytest.skip("需要真实原生模块及测试资源")
    from mirrorforce.distill.data import TOKENS, CardTexts, validate
    from mirrorforce.distill.policy import DistillPolicy, StudentPolicy
    from mirrorforce.netduel.agent_policy import RemotePolicy
    from mirrorforce.netduel.cards import CardPool
    from mirrorforce.worldmodel.engine import load_ydk
    from tools.mf_runtime_local_match import play_game, service_identity
    from tools.mf_runtime_client_parity import host_core
    sock = Path(tempfile.gettempdir()) / ("distill-L1-" + str(os.getpid()) + ".sock")
    announce = next(bundle.glob("announce-tables-*.json"))
    config = {"max_options": 192, "max_steps": 1000, "public_opponent_recipe": True}
    cmd = [sys.executable, "-m", "tools.mf_runtime_policy_service", "--backend", "uniform", "--selection", "sample",
           "--cards-db", str(bundle / "cards.cdb"), "--code-list", str(bundle / "code_list.txt"),
           "--script-root", str(bundle), "--announce-tables", str(announce), "--config", json.dumps(config),
           "--opponent-mode", "mirror", "--socket", str(sock), "--out", str(tmp_path / "service")]
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
        def encode(self, text, **kwargs):
            return [0] * 10
    class RandomStudent:
        tokenizer = Tokenizer()
        def select(self, row, seed=None, **kwargs):
            n = len(row["legal_actions"])
            index = random.Random(seed).randrange(n)
            return {"index": index, "token": TOKENS[index], "logprob": -math.log(n),
                    "selection": "sample", "temperature": 1.}
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise AssertionError((tmp_path / "service.log").read_text())
                try:
                    identity = service_identity(sock, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("测试教师未启动")
                    time.sleep(.1)
            core = host_core(native, bundle / "cards.cdb", bundle, tmp_path)
            recipe = load_ydk(bundle / "decks/SkyStriker.ydk")
            cards, texts = CardPool(bundle / "cards.cdb"), CardTexts(bundle / "cards.cdb")
            from tools.mf_runtime_policy_service import load_native
            local_native, _ = load_native(SimpleNamespace(script_root=bundle, cards_db=bundle / "cards.cdb",
                code_list=bundle / "code_list.txt", backend="uniform", announce_tables=announce, announce_cap=192))
            evidence = []
            for independent in (False, True):
                held = []
                def factory(role, seat):
                    if role == "b":
                        return RemotePolicy("unix:" + str(sock), identity, seed=100001)
                    metadata = {"run_id": "L1-test-only", "game_id": "test-" + str(independent),
                        "deal_seed": 100001, "first_player": "self", "deck_id": "test-deck",
                        "opponent_deck_id": "test-deck", "student_checkpoint_sha": "random-test-not-model",
                        "source_commit": "test", "engine_hash": "test", "cards_db_hash": "test", "tokenizer_hash": "test"}
                    options = dict(chooser=RandomStudent(), texts=texts, metadata=metadata, seed=100001)
                    policy = (StudentPolicy(local_native, identity, **options) if independent else
                              DistillPolicy("unix:" + str(sock), identity, **options))
                    original = policy._make_row
                    def checked_row(raw, logits, sha, sub):
                        try:
                            return original(raw, logits, sha, sub)
                        except Exception:
                            (tmp_path / "failed-public-view.json").write_text(json.dumps(raw, indent=2))
                            raise
                    policy._make_row = checked_row
                    held.append(policy)
                    return policy
                result = play_game(core, cards, recipe, 100001, True, {"a": sock, "b": sock},
                    {"a": identity, "b": identity}, max_seconds=120, policy_factory=factory, preserve_role_deals=True)
                assert not result["error"], result
                assert held[0].decisions
                for row in held[0].samples:
                    row.update(completed=True, training_eligible=True, episode_result="test-only")
                    validate(row)
                if independent:
                    assert held[0].sock is None and held[0].samples == []
                else:
                    assert held[0].samples
                evidence.append({"independent_student": independent, "result": result,
                                 "validated_samples": len(held[0].samples), "actual_llm": False})
                if not independent:
                    remote_decisions = [row["chosen"] for row in held[0].decisions]
                    remote_result = result
                    tokenizer_path = os.environ.get("MF_DISTILL_REAL_TOKENIZER")
                    if tokenizer_path:
                        # 仅用于序列化诊断，严格标注不可用于训练。
                        with (tmp_path / "test-only-samples.jsonl").open("w") as diagnostic:
                            for sample in held[0].samples:
                                diagnostic.write(json.dumps({**sample, "training_eligible": False,
                                    "exclusion_reason": "uniform_teacher_random_student_test_only"}) + "\n")
                        from transformers import AutoTokenizer
                        from mirrorforce.distill.data import prompt_details
                        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                        tok.add_special_tokens({"additional_special_tokens": TOKENS})
                        lengths, overflows = [], 0
                        for sample in held[0].samples:
                            _, token_ids, _ = prompt_details(sample, tok, limit=1000000)
                            lengths.append(len(token_ids))
                            try:
                                prompt_details(sample, tok, limit=4096)
                            except ValueError:
                                overflows += 1
                        lengths.sort()
                        (tmp_path / "real-tokenizer-lengths.json").write_text(json.dumps({
                            "model": tokenizer_path, "samples": len(lengths), "p50": lengths[len(lengths)//2],
                            "p95": lengths[int(len(lengths)*.95)], "p99": lengths[int(len(lengths)*.99)],
                            "max": max(lengths), "context_overflows_after_history_trim": overflows,
                            "teacher": "uniform_test_only", "student": "random_test_only"}, indent=2))
                else:
                    assert [row["chosen"] for row in held[0].decisions] == remote_decisions
                    assert result["winner"] == remote_result["winner"]
            from tools.mf_runtime_local_match import sha256
            (tmp_path / "L1-evidence.json").write_text(json.dumps({
                "schema": "mirrorforce_distill_native_test/v1", "level": "L1",
                "teacher_backend": "uniform", "student_backend": "seeded_random_not_llm",
                "native_path": str(native), "native_sha256": sha256(native),
                "cards_db_sha256": sha256(bundle / "cards.cdb"), "games": evidence,
                "training_data_written": False}, indent=2))
        finally:
            process.terminate()
            process.wait(timeout=10)
            sock.unlink(missing_ok=True)
