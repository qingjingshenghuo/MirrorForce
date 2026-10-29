"""本机零云费用必须显式声明，云端配置不能绕过预算。"""
import pytest
from mirrorforce.distill.collect import validate_cost_mode


def test_local_and_paid_modes_are_explicit():
    validate_cost_mode(True, 0, 0)
    validate_cost_mode(False, 2, 50)


@pytest.mark.parametrize("local,rate,cap", [(False, 0, 0), (True, 1, 0), (False, float("nan"), 50), (False, 2, float("inf"))])
def test_invalid_cost_modes_rejected(local, rate, cap):
    with pytest.raises(ValueError):
        validate_cost_mode(local, rate, cap)
