"""验证抽样上限、固定清单和分组统计。"""
from collections import Counter
from mirrorforce.distill.benchmark import sample_states, cluster_interval


def test_stratified_sampling_is_stable_and_caps_games_and_turns():
    rows = [{'game_id': str(g), 'first_player': str(g % 2),
             'observation': {'unique': g * 100 + i}, 'legal_actions': [{'kind': 'activate'}],
             'view': {'phase': i % 2, 'prompt_msg': i % 3, 'turn': i // 5}}
            for g in range(5) for i in range(30)]
    selected = sample_states(rows, 50, 42, 6, 2)
    assert selected == sample_states(rows, 50, 42, 6, 2)
    assert len(selected) == 30
    assert max(Counter(rows[i]['game_id'] for i in selected).values()) <= 6
    assert max(Counter((rows[i]['game_id'], rows[i]['view']['turn']) for i in selected).values()) <= 2
    assert len({rows[i]['view']['phase'] for i in selected}) == 2


def test_duplicates_are_not_counted_as_independent_states():
    row = {'game_id': 'a', 'observation': {'lp': 8000}, 'legal_actions': [{}]}
    assert len(sample_states([row, dict(row, game_id='b')])) == 1


def test_single_game_does_not_get_false_precision():
    assert cluster_interval([{'game_id': 'a', 'score': 1}] * 100, 'score') is None
    assert cluster_interval([{'game_id': 'a', 'score': 0}, {'game_id': 'b', 'score': 1}], 'score') == [0., 1.]


def test_swapped_seats_share_the_same_deal_cluster():
    assert cluster_interval([{'game_id': 'a', 'deal_seed': 200100, 'score': 1},
                             {'game_id': 'b', 'deal_seed': 200100, 'score': 0}], 'score') is None


def test_semantic_only_rows_keep_turn_caps():
    rows = [{'game_id': 'a', 'first_player': 'self',
             'observation': {'turn': i // 4, 'unique': i, 'phase': 'main_phase_1'},
             'legal_actions': [{'kind': 'activate'}]} for i in range(12)]
    selected = sample_states(rows, 20, 42, 20, 2)
    assert len(selected) == 6
