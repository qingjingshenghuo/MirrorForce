"""训练时置换菜单，保留到真实原生动作的双射及教师监督。"""
import copy
from .data import TOKENS, digest, model_input


def permute_training_menu(row, order):
    n = len(row['legal_actions'])
    if row.get('augmentation') or not isinstance(order, list) or any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
        raise ValueError('只能对原始菜单施加一次完整整数排列')
    changed = copy.deepcopy(row)
    changed.pop('view', None)
    changed['legal_actions'] = [{**row['legal_actions'][old], 'token': TOKENS[new]} for new,old in enumerate(order)]
    changed['teacher_logits'] = [row['teacher_logits'][old] for old in order]
    chosen = order.index(row['chosen_index'])
    changed.update(chosen=chosen,chosen_index=chosen,chosen_token=TOKENS[chosen])
    changed['action_mapping'] = [{'token': TOKENS[new], 'native_index': old} for new,old in enumerate(order)]
    changed['augmentation'] = {'kind':'menu_permutation/v1','order':order,
        'source_state_sha256':digest(model_input(row)),
        'source_legal_actions':copy.deepcopy(row['legal_actions']),
        'source_teacher_logits':list(row['teacher_logits']), 'source_chosen_index':row['chosen_index']}
    return changed
