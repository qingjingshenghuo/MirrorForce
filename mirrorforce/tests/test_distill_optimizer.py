"""动作优化器状态必须只覆盖新行，且恢复保留FP32累积量。"""
import copy
import torch
from mirrorforce.distill.optimizer import ActionRowAdamW


class StudentFixture:
    def __init__(self):
        self.ids = [10, 11]
        self.row_parameters = [torch.nn.Parameter(torch.ones(1000, 4))]
        self.lora = torch.nn.Parameter(torch.ones(2))
    def optimizer_groups(self, lr):
        return [{'params': [self.lora], 'lr': lr, 'weight_decay': .01},
                {'params': self.row_parameters, 'lr': lr, 'weight_decay': 0.}]


def update(student, optimizer):
    optimizer.zero_grad()
    student.lora.grad = torch.ones_like(student.lora)
    student.row_parameters[0].grad = torch.ones_like(student.row_parameters[0])
    optimizer.step()


def test_sparse_moments_and_exact_resume():
    a = StudentFixture()
    optimizer = ActionRowAdamW(a, 1e-4)
    update(a, optimizer)
    state = copy.deepcopy(optimizer.state_dict())
    moment = next(iter(state['rows']['state'].values()))['exp_avg']
    assert moment.shape == (2, 4)
    assert torch.equal(a.row_parameters[0][:10], torch.ones(10, 4))
    b = StudentFixture()
    b.row_parameters[0].data.copy_(a.row_parameters[0])
    b.lora.data.copy_(a.lora)
    restored = ActionRowAdamW(b, 1e-4)
    restored.load_state_dict(state)
    update(a, optimizer)
    update(b, restored)
    assert torch.equal(a.row_parameters[0], b.row_parameters[0])
    assert torch.equal(a.lora, b.lora)


def test_action_learning_rate_is_independent():
    student = StudentFixture()
    optimizer = ActionRowAdamW(student, 1e-4, action_lr=1e-3)
    update(student, optimizer)
    assert optimizer.dense.param_groups[0]['lr'] == 1e-4
    assert optimizer.rows.param_groups[0]['lr'] == 1e-3
    assert abs(float(student.row_parameters[0][10, 0]) - .999) < 1e-6
    assert torch.equal(student.row_parameters[0][:10], torch.ones(10, 4))
