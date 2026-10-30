"""LoRA使用标准AdamW，动作行仅保存实际可训练行的FP32状态。"""
import torch


class ActionRowAdamW:
    format = 'action_rows_adamw/v1'

    def __init__(self, student, lr, action_lr=None):
        groups = student.optimizer_groups(lr)
        self.dense = torch.optim.AdamW(groups[:1], lr=lr)
        self.parameters = student.row_parameters
        self.ids = student.ids
        self.values = [torch.nn.Parameter(p.detach()[self.ids].float().clone()) for p in self.parameters]
        self.rows = torch.optim.AdamW(self.values, lr=lr if action_lr is None else action_lr, weight_decay=0.)

    def zero_grad(self, set_to_none=True):
        self.dense.zero_grad(set_to_none=set_to_none)
        self.rows.zero_grad(set_to_none=set_to_none)
        for p in self.parameters:
            p.grad = None

    def step(self):
        for p, value in zip(self.parameters, self.values):
            value.grad = None if p.grad is None else p.grad[self.ids].float()
        self.dense.step()
        self.rows.step()
        with torch.no_grad():
            for p, value in zip(self.parameters, self.values):
                p[self.ids] = value.to(p.dtype)

    def state_dict(self):
        return {'format': self.format, 'dense': self.dense.state_dict(),
                'rows': self.rows.state_dict(), 'values': [p.detach().cpu() for p in self.values]}

    def load_state_dict(self, state):
        if state.get('format') != self.format:
            raise ValueError('优化器格式不兼容；请使用adapter启动新任务')
        if len(state['values']) != len(self.values):
            raise ValueError('动作行共享关系不一致')
        self.dense.load_state_dict(state['dense'])
        self.rows.load_state_dict(state['rows'])
        with torch.no_grad():
            for value, saved in zip(self.values, state['values']):
                value.copy_(saved)
