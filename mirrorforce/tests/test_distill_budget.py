"""预算边界必须保留评测资金，拒绝非有限价格。"""
import pytest
from mirrorforce.distill.budget import Budget


def test_reserve_cannot_be_spent_by_training():
    budget = Budget(450, 2, 150)
    assert budget.available(250) == 50
    assert budget.available(300) == 0
    assert budget.available(300, evaluation=True) == 150
    assert Budget.is_evaluation("eval-online-kd")
    assert not Budget.is_evaluation("online-data-1")


@pytest.mark.parametrize("values", [(float("nan"), 2, 1), (450, 0, 150), (450, 2, 450), (501, 2, 150)])
def test_invalid_cost_rejected(values):
    with pytest.raises(ValueError):
        Budget(*values)
