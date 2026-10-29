"""无损紧凑文本编码：完整语义与动作菜单必须可逆。"""
import copy
import json
import pytest
from mirrorforce.distill.data import compact_input, decode_compact_input, public_view, semantic_view, TOKENS, SERIALIZER_VERSION


def test_roundtrip_optional_values_types_and_text_sharing():
    text = "完整卡牌效果不能被裁剪。" * 30
    original = {"observation": {"cards": [
        {"ref": f"self.monster.{i}", "attack": 1500, "level": 4, "text": text, "value": v}
        for i, v in enumerate([0, False, 1, 1., None, {}, [], ""]) ]},
        "card_dictionary": {"1234": {"name": "卡牌", "effect_text": text}},
        "public_history": [{"event": "activate", "card_code": 1234}] * 20,
        "legal_actions": [{"token": TOKENS[i], "kind": "activate", "description": text} for i in range(192)]}
    before = copy.deepcopy(original)
    encoded = compact_input(original)
    assert original == before
    decoded = decode_compact_input(encoded)
    assert json.dumps(decoded, sort_keys=True, ensure_ascii=False) == json.dumps(original, sort_keys=True, ensure_ascii=False)
    assert encoded["text_pool"] == [text]
    assert len(json.dumps(encoded, ensure_ascii=False)) < len(json.dumps(original, ensure_ascii=False)) / 3
    assert [row["token"] for row in decoded["legal_actions"]] == TOKENS


def test_empty_missing_and_null_are_distinct():
    original = [{"a": None}, {}, {"a": False}, {"a": 0}] * 10
    assert decode_compact_input(compact_input(original)) == original


def test_hidden_truth_and_metadata_do_not_enter_codec():
    from test_distill_data_v2 import observer, native_action
    left = semantic_view(public_view(observer(123), [native_action()]))
    right = semantic_view(public_view(observer(987), [native_action()]))
    assert compact_input(left) == compact_input(right)


def test_decode_rejects_invalid_pool_or_table_index():
    for invalid in [{"$text": -1}, {"$text": True}, {"$table": {"columns": ["a"], "defaults": [0], "rows": [[1, 2]]}}]:
        with pytest.raises(ValueError):
            decode_compact_input({"format": SERIALIZER_VERSION, "text_pool": [], "input": invalid})


def test_reserved_marker_cannot_collide_with_input():
    with pytest.raises(ValueError, match="保留键"):
        compact_input({"$text": "literal"})


def test_real_tokenizer_diagnostic_when_requested():
    """使用落盘测试only样本，不经训练入口，保存实际长度证据。"""
    import os
    from pathlib import Path
    source = os.environ.get("MF_DISTILL_PROFILE_SAMPLES")
    if not source:
        pytest.skip("未指定真实tokenizer诊断样本")
    from transformers import AutoTokenizer
    from mirrorforce.distill.data import model_input, prompt_details
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MF_DISTILL_REAL_TOKENIZER"], local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line]
    cap = int(os.environ.get("MF_DISTILL_PROFILE_CAP", "0"))
    if cap:
        indexes = sorted(set([0, len(rows)//2, len(rows)-1]))[:cap]
        rows = [rows[i] for i in indexes]
    measured = []
    context_limit = int(os.environ.get("MF_DISTILL_PROFILE_CONTEXT", "16384"))
    for i, row in enumerate(rows):
        semantic = model_input(row)
        assert decode_compact_input(compact_input(semantic)) == semantic
        full_text, ids, _ = prompt_details(row, tokenizer, limit=1000000)
        no_history = copy.deepcopy(row)
        no_history["public_history"] = []
        _, essential_ids, _ = prompt_details(no_history, tokenizer, limit=1000000)
        payload = compact_input(model_input(no_history))
        dictionary_tokens = len(tokenizer.encode(json.dumps(payload["input"].get("card_dictionary", {}), ensure_ascii=False)))
        pool_tokens = len(tokenizer.encode(json.dumps(payload["text_pool"], ensure_ascii=False)))
        try:
            _, fitted_ids, fit_info = prompt_details(row, tokenizer, limit=context_limit)
            fitted_length, removed, overflow = len(fitted_ids), fit_info["history_removed"], False
        except ValueError as exc:
            if "context_overflow" not in str(exc):
                raise
            fitted_length, removed, overflow = None, None, True
        measured.append({"sample": i, "full_tokens": len(ids), "essential_tokens": len(essential_ids),
                         "dictionary_tokens": dictionary_tokens, "shared_text_tokens": pool_tokens,
                         "fitted_tokens": fitted_length, "history_removed": removed, "overflow": overflow,
                         "cards": len(semantic["card_dictionary"]), "actions": len(semantic["legal_actions"])})
    result = {"serializer_version": SERIALIZER_VERSION, "source": source, "samples": len(rows),
              "context_limit": context_limit, "essential_overflows": sum(x["essential_tokens"] + 1 > context_limit for x in measured),
              "full_overflows": sum(x["full_tokens"] + 1 > context_limit for x in measured),
              "suffix_overflows": sum(x["overflow"] for x in measured), "measurements": measured}
    out = os.environ.get("MF_DISTILL_PROFILE_OUT", source + ".compact-profile.json")
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
