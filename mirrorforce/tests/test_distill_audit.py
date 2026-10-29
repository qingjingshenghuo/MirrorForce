"""菜单置换必须真正改变纯语义prompt，并保持监督与执行映射一致。"""
import copy
import pytest
from test_distill_data_v2 import sample
from mirrorforce.distill.audit import permute_menu
from mirrorforce.distill.data import model_input, validate, TOKENS


@pytest.mark.parametrize("with_view", [True, False])
def test_semantic_and_raw_menu_are_both_permuted(with_view):
    row = sample()
    if not with_view:
        row.pop("view")
    before = copy.deepcopy(row)
    changed = permute_menu(row, [1, 0])
    assert row == before
    assert model_input(changed) != model_input(row)
    assert changed["legal_actions"][0]["kind"] == row["legal_actions"][1]["kind"]
    assert changed["legal_actions"][0]["token"] == TOKENS[0]
    assert changed["teacher_logits"] == list(reversed(row["teacher_logits"]))
    assert changed["chosen_index"] == 0
    validate(changed)


def test_invalid_permutation_rejected():
    with pytest.raises(ValueError):
        permute_menu(sample(), [0, 0])
