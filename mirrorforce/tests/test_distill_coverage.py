"""防止等token组批静默丢弃长状态。"""
import random
import pytest
from mirrorforce.distill.train import exact_token_sampler, token_coverage


def test_long_states_are_rejected_before_training():
    report = token_coverage([4096, 16383], 16384)
    assert report['excluded_indexes'] == [1]
    with pytest.raises(ValueError, match='永久排除'):
        exact_token_sampler([4096, 16383], 16384)


def test_long_states_participate_when_quota_supports_them():
    sampler = exact_token_sampler([8192, 16384], 32768)
    rng = random.Random(42)
    draws = [sampler(rng) for _ in range(100)]
    assert all(sum([8192, 16384][i] for i in batch) == 32768 for batch in draws)
    assert {i for batch in draws for i in batch} == {0, 1}


def test_unreachable_quota_has_audit():
    assert not token_coverage([10, 20], 21)['quota_reachable']
