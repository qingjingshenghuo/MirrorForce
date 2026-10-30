"""固定盘面比较不能混用数据、标签或模板。"""
import copy
import pytest
from mirrorforce.distill.audit_compare import compare_audits


def fixture():
    return {'source_sha256': 'data', 'selected_indexes': [0], 'prompt_style': 'named_menu',
        'prompt_version': 'v1', 'context_limit': 16384, 'adapter_sha256': 'model',
        'records': [{'game_id': 'a', 'deal_seed': 200000, 'decision_id': 0,
            'state_sha256': 'state', 'target': 0, 'legal_count': 2, 'chosen': 1,
            'agreement': 0, 'kl': 2., 'menu_stable': 0}]}


def test_paired_gain_with_one_cluster_has_no_false_ci():
    before, after = fixture(), fixture()
    after['records'][0].update(agreement=1, kl=.5, menu_stable=1, chosen=0)
    result = compare_audits(before, after)
    assert result['agreement_delta'] == 1 and result['kl_delta'] == -1.5
    assert result['agreement_delta_ci95_by_deal'] is None


@pytest.mark.parametrize('key,value', [('source_sha256', 'other'), ('prompt_style', 'compact'), ('context_limit', 8192)])
def test_mismatched_conditions_refused(key, value):
    before, after = fixture(), fixture()
    after[key] = value
    with pytest.raises(ValueError, match='比较条件'):
        compare_audits(before, after)


def test_reordered_or_changed_state_refused():
    before, after = fixture(), fixture()
    after['records'][0]['state_sha256'] = 'other'
    with pytest.raises(ValueError, match='盘面或标签'):
        compare_audits(before, after)
