"""v2 语义输入、消息历史及严格数据边界回归。"""
import copy
import json
import struct
import sqlite3
from types import SimpleNamespace as NS
import pytest
from mirrorforce.distill.data import (SCHEMA, TOKENS, REQUIRED_METADATA, public_view, semantic_view, semantic_action,
    public_event, build_sample, validate, read_rows, prompt_text, probabilities, split_for)
from mirrorforce.netduel import constants as C


def native_action(**updates):
    value = dict(act=8, phase=0, finish=False, code=456, effect=10010, position=0, number=0, place=0, attribute=0, spec="h1")
    return {**value, **updates}


def observer(secret=123, own=456):
    hidden = NS(known=False, code=secret, attack=secret)
    visible = NS(known=True, code=own, position=1, attack=1500, defense=0, level=4, rank=0, status=0)
    board = NS(zones={(0, 2): [visible], (1, 2): [hidden], (1, 1): [visible]},
               materials={}, counters={}, turn_player=0, phase=4, deck_count=[30, 30], extra_count=[15, 15])
    return NS(board=board, seat=0)


def sample(seed=100000):
    view = public_view(observer(), [native_action(), native_action(act=9, code=0, spec="")])
    view.update(lp=[8000, 6000], turn=1, prompt_msg=11,
                card_texts={"456": {"name": "测试卡", "effect_text": "测试效果"}})
    metadata = {key: "test" for key in REQUIRED_METADATA}
    metadata.update(deal_seed=seed, viewer=0, first_player="self", subdecision_id=0)
    row = build_sample(view, [{"event": "new_turn", "player": "self"}], metadata, [0., 1.], 1)
    row.update(completed=True, training_eligible=True, episode_result="games.jsonl#test")
    return row


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)
    def encode(self, text, **kwargs):
        return list(text)


def test_named_menu_preserves_full_input_and_only_visible_names():
    from mirrorforce.distill.data import prompt_details
    row = sample()
    text, _, _ = prompt_details(row, Tokenizer(), 20000, 'named_menu')
    assert '合法动作摘要' in text and '测试卡' in text and '测试效果' in text
    assert all(token in text for token in TOKENS[:2])
    changed = copy.deepcopy(row)
    changed['teacher_logits'] = [999., -999.]
    changed['teacher_identity'] = '私密教师身份'
    assert text == prompt_details(changed, Tokenizer(), 20000, 'named_menu')[0]
    assert '私密教师身份' not in text


def test_menu_augmentation_keeps_native_mapping_and_supervision():
    from mirrorforce.distill.augmentation import permute_training_menu
    from mirrorforce.distill.data import model_input
    row = sample()
    changed = validate(permute_training_menu(row, [1, 0]))
    assert changed['chosen_index'] == 0
    assert changed['teacher_logits'] == [1., 0.]
    assert changed['action_mapping'][0]['native_index'] == 1
    assert 'augmentation' not in model_input(changed)
    assert row['chosen_index'] == 1 and 'augmentation' not in row
    changed['observation']['lp'] = [1, 2]
    with pytest.raises(ValueError, match='来源盘面'):
        validate(changed)


@pytest.mark.parametrize('order', [[0, 0], [True, 0], [0], [0, '1']])
def test_menu_augmentation_rejects_invalid_bijection(order):
    from mirrorforce.distill.augmentation import permute_training_menu
    with pytest.raises(ValueError):
        permute_training_menu(sample(), order)


def test_hidden_truth_is_not_serialized_but_own_card_is():
    rows = [native_action()]
    assert public_view(observer(123), rows) == public_view(observer(987), rows)
    assert public_view(observer(123, 456), rows) != public_view(observer(123, 457), rows)
    view = semantic_view(public_view(observer(), rows))
    unknown = next(c for c in view["observation"]["cards"] if c["ref"] == "opponent.hand.0")
    assert unknown == {"ref": "opponent.hand.0", "visibility": "unknown"}
    assert view["observation"]["phase"] == "main_phase_1"


def test_same_named_card_positions_and_overlay_distinguished():
    a = semantic_action(native_action(spec="m1"), 0)
    b = semantic_action(native_action(spec="m2"), 1)
    c = semantic_action(native_action(spec="om1b"), 2)
    assert a["source_ref"] != b["source_ref"]
    assert c["source_ref"] == "opponent.monster.0.material.1"
    assert "effect_index" not in a and a["native_description_index"] == 10010


def test_deck_candidates_have_distinct_offered_references():
    left = semantic_action(native_action(act=0, spec="33"), 0, C.MSG_SELECT_CARD)
    right = semantic_action(native_action(act=0, spec="31"), 1, C.MSG_SELECT_CARD)
    assert left["candidate_ref"] == "self.offered_deck_card.32"
    assert right["candidate_ref"] != left["candidate_ref"]


def test_exact_announce_number_and_option_identity():
    with pytest.raises(ValueError, match="精确 response"):
        semantic_action(native_action(act=0, spec="", number=255), 0, C.MSG_ANNOUNCE_NUMBER)
    value = native_action(act=0, spec="", number=255, response=1000000, exact_semantics_version=1)
    assert semantic_action(value, 0, C.MSG_ANNOUNCE_NUMBER)["announced_number"] == 1000000
    with pytest.raises(ValueError, match="精确 description"):
        semantic_action(native_action(), 0, C.MSG_SELECT_OPTION)
    assert semantic_action(native_action(description_code=123456), 0, C.MSG_SELECT_OPTION)["description_code"] == 123456


def test_database_effect_options_use_exact_description(tmp_path):
    from mirrorforce.distill.data import CardTexts
    path = tmp_path / "cards.cdb"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE texts(id INTEGER,name TEXT,desc TEXT,str1 TEXT,str2 TEXT)")
        db.execute("INSERT INTO texts VALUES(456,'Card','Full effect','Exact option 0','Exact option 1')")
        db.execute("INSERT INTO texts VALUES(4560,'Other','Other effect','Exact option 0','Exact option 1')")
    view = public_view(observer(), [native_action(description_code=(4560 << 4) | 1), native_action(act=9, spec="", code=0)])
    result = semantic_view(CardTexts(path).enrich(view))
    assert result["legal_actions"][0]["description"] == "Exact option 1"
    assert result["legal_actions"][0]["description_status"] == "exact_card_description"
    assert list(result["card_dictionary"]) == ["456", "4560"]


def test_public_reveal_and_prompt_trim_statistics():
    from mirrorforce.distill.data import prompt_details
    packet = bytes([0, 0, 1]) + struct.pack("<I", 456) + bytes([0, 2, 0])
    assert public_event(C.MSG_CONFIRM_CARDS, packet, 0)["cards"] == [{"card_code": 456, "ref": "self.hand.0"}]
    row = sample()
    row["public_history"] = [{"event": "old", "details": "x" * 10000}]
    _, ids, stats = prompt_details(row, Tokenizer())
    assert stats == {"history_removed": 1, "input_tokens": len(ids)}


def test_long_history_trim_has_bounded_tokenizer_work():
    from mirrorforce.distill.data import prompt_details
    class CountingTokenizer(Tokenizer):
        calls = 0
        def encode(self, text, **kwargs):
            self.calls += 1
            return super().encode(text, **kwargs)
    row = sample()
    row["public_history"] = [{"event": "old", "index": i, "text": "x" * 100} for i in range(1000)]
    tokenizer = CountingTokenizer()
    _, ids, stats = prompt_details(row, tokenizer)
    assert len(ids) + 1 <= 4096
    assert stats["history_removed"] > 0
    assert tokenizer.calls <= 13
    assert len(row["public_history"]) == 1000


@pytest.mark.parametrize("changes", [{"act": 999}, {"phase": 4}, {"spec": "bad"}])
def test_unknown_native_semantics_refused(changes):
    with pytest.raises(ValueError):
        semantic_action(native_action(**changes), 0)


def test_supervision_and_seat_metadata_not_in_prompt():
    row = sample()
    changed = copy.deepcopy(row)
    changed.update(teacher_logits=[-999., 999.], chosen=0, teacher_identity="SECRET_TEACHER", viewer=1)
    assert prompt_text(row, Tokenizer()) == prompt_text(changed, Tokenizer())
    text, _ = prompt_text(row, Tokenizer())
    assert "SECRET_TEACHER" not in text and "测试效果" in text


def test_long_history_removed_without_mutating_source_or_menu():
    row = sample()
    row["public_history"] = [{"event": "old", "payload": "x" * 10000}]
    before = copy.deepcopy(row)
    text, _ = prompt_text(row, Tokenizer())
    assert row == before
    assert TOKENS[0] in text and TOKENS[1] in text
    with pytest.raises(ValueError, match="context_overflow"):
        prompt_text(row, Tokenizer(), limit=20)


def test_valid_complete_sample_and_unfinished_refused():
    row = sample()
    assert validate(row) is row
    row["completed"] = False
    with pytest.raises(ValueError, match="整局"):
        validate(row)


def test_v1_and_bad_tokens_refused():
    row = sample()
    row["schema"] = "mirrorforce_action_distill/v1"
    with pytest.raises(ValueError, match="v1"):
        validate(row)
    row = sample()
    row["legal_actions"][0]["token"] = TOKENS[1]
    with pytest.raises(ValueError, match="token"):
        validate(row)


def test_tampered_menu_or_semantic_view_refused():
    row = sample()
    row["view"]["actions"][0]["effect"] = 999
    with pytest.raises(ValueError, match="摘要"):
        validate(row)
    row = sample()
    row["observation"]["lp"]["self"] = 1
    with pytest.raises(ValueError, match="审计"):
        validate(row)


def test_duplicate_decisions_and_cross_split_same_trace_refused(tmp_path):
    row = sample()
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="重复决策"):
        read_rows(path)
    other = sample(300000)
    other["game_id"] = "other"
    path.write_text(json.dumps(row) + "\n" + json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="跨划分"):
        read_rows(path)


def test_split_seed_cannot_be_forged():
    row = sample()
    row["split"] = "test"
    with pytest.raises(ValueError, match="种子"):
        validate(row)
    with pytest.raises(ValueError):
        split_for(True)


def test_public_messages_do_not_reveal_drawn_card():
    a = bytes([1, 1]) + struct.pack("<I", 123)
    b = bytes([1, 1]) + struct.pack("<I", 987)
    assert public_event(C.MSG_DRAW, a, 0) == public_event(C.MSG_DRAW, b, 0)
    event = public_event(C.MSG_NEW_PHASE, struct.pack("<H", C.PHASE_MAIN2), 0)
    assert event == {"event": "new_phase", "phase": "main_phase_2"}
    move = struct.pack("<IIII", 0, 1 | 1 << 8, 1 | 2 << 8, 0)
    assert public_event(C.MSG_MOVE, move, 0)["card_code"] == 0
    assert public_event(C.MSG_UPDATE_DATA, b"arbitrary hidden data", 0) is None


def test_public_activation_exact_description_and_malformed_packet():
    event = public_event(C.MSG_CHAINING, struct.pack("<II", 456, 4 << 8) + bytes([0, 4, 0]) + struct.pack("<I", 12345) + bytes([1]), 0)
    assert event["source"] == "self.monster.0" and event["description_code"] == 12345
    with pytest.raises(ValueError, match="长度"):
        public_event(C.MSG_CHAINING, b"", 0)


def test_extreme_logits_and_nonfinite_rejected():
    assert probabilities([10000., 10000.]) == [.5, .5]
    with pytest.raises(ValueError):
        probabilities([float("nan"), 0])
