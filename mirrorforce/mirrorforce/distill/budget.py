"""实验阶段预算，训练不能花掉评测预留。"""
import math


class Budget:
    def __init__(self, cap, rate, evaluation_reserve=150):
        if not all(math.isfinite(x) for x in (cap, rate, evaluation_reserve)) or not 0 < cap <= 500 or rate <= 0 or not 0 <= evaluation_reserve < cap:
            raise ValueError("预算、单价或评测预留无效")
        self.cap, self.rate, self.reserve = cap, rate, evaluation_reserve

    def available(self, spent, evaluation=False):
        if not math.isfinite(spent) or spent < 0:
            raise ValueError("累计费用无效")
        return max(0., self.cap - spent - (0 if evaluation else self.reserve))

    @staticmethod
    def is_evaluation(name):
        return name.startswith(("eval-", "audit-"))
