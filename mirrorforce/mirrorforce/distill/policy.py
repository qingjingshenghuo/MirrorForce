"""学生实际轨迹采样与无教师客户端适配。"""
import math
import random
import time
from mirrorforce.netduel.agent_policy import RemotePolicy
from mirrorforce.netduel.actions import SelectResult, parse_select
from mirrorforce.netduel import constants as C
from .data import TOKENS, digest, probabilities


class DistillPolicy(RemotePolicy):
    def __init__(self, *args, chooser=None, texts=None, metadata=None, selection='sample', temperature=1., **kwargs):
        super().__init__(*args, **kwargs)
        if selection not in ('sample', 'greedy') or temperature <= 0:
            raise ValueError('非法选择模式或温度')
        self.chooser, self.texts, self.metadata = chooser, texts, dict(metadata or {})
        self.selection, self.temperature = selection, float(temperature)
        self.samples, self.history = [], []
        self.rng = random.Random(self.seed)
        self.teacher_ms = 0.

    def observe_game_message(self, msg, body):
        super().observe_game_message(msg, body)
        if self.active:
            from .data import public_event
            event = public_event(int(msg), bytes(body), self.seat)
            if event is not None:
                self.history.append(event)

    def _choose(self, row, teacher_logits):
        n = len(teacher_logits)
        if n == 1:
            return {'index': 0, 'token': TOKENS[0], 'logprob': 0., 'selection': 'forced', 'temperature': 1.}
        if self.chooser is not None:
            result = self.chooser.select(row, selection=self.selection, temperature=self.temperature,
                                         seed=self.rng.randrange(2 ** 31))
        else:
            index = max(range(n), key=teacher_logits.__getitem__)
            result = {'index': index, 'token': TOKENS[index], 'logprob': 0., 'selection': 'greedy', 'temperature': 1.}
        index = result['index']
        if type(index) is not int or not 0 <= index < n or result['token'] != TOKENS[index]:
            raise ValueError('学生输出或动作映射非法；禁止教师接管')
        if not math.isfinite(result['logprob']) or result['logprob'] > 1e-6:
            raise ValueError('行为概率无效')
        return result

    def _make_row(self, raw_view, logits, obs_sha, subdecision):
        from .data import build_sample
        start = time.perf_counter()
        view = self.texts.enrich(raw_view, self.history)
        view.update(lp=list(self._client._lp), turn=self._client.result.turns)
        meta = {**self.metadata, 'decision_id': len(self.responses), 'subdecision_id': subdecision,
                'viewer': self.seat, 'teacher_identity': digest(self.expected_identity), 'obs_sha256': obs_sha}
        row = build_sample(view, self.history, meta, logits, 0,
            behavior={'selection': self.selection, 'temperature': self.temperature, 'behavior_logprob': 0.,
                      'teacher_ms': self.teacher_ms, 'student_ms': 0., 'serialization_ms': None, 'input_tokens': None})
        row['serialization_ms'] = (time.perf_counter() - start) * 1000
        return row

    def _source_prompt(self):
        started = time.perf_counter()
        reply = self._call({'op': 'prompt', 'session': self.session,
                            'messages': [[m, b.hex()] for m, b in self.pending]})
        self.teacher_ms = (time.perf_counter() - started) * 1000
        self.pending = []
        return reply

    def _view(self, pending):
        reply = self._call({'op': 'distill_view', 'session': self.session, 'obs_sha256': pending['obs_sha256']})
        if reply['obs_sha256'] != pending['obs_sha256'] or reply['menu_sha256'] != digest(reply['view']['actions']):
            raise ValueError('教师视图或菜单响应已过期')
        return reply['view']

    def _commit(self, pending, index, menu_sha):
        started = time.perf_counter()
        reply = self._call({'op': 'commit', 'session': self.session, 'index': index,
                            'obs_sha256': pending['obs_sha256'], 'menu_sha256': menu_sha})
        self.teacher_ms = (time.perf_counter() - started) * 1000
        return reply

    def _parse(self, msg, body, ctx):
        standard = parse_select(msg, body, ctx)
        try:
            if not self.active or not self.pending or self.pending[-1] != (int(msg), bytes(body)):
                raise ValueError('客户端消息与当前提示不同步')
            reply = self._source_prompt()
            self.forced += bool(reply.get('forced'))
            for sub in range(4096):
                if 'response' in reply:
                    break
                pending = reply['pending']
                probabilities(pending['logits'])
                raw = self._view(pending)
                raw['prompt_msg'] = pending['msg']
                if len(raw['actions']) != pending['rows']:
                    raise ValueError('菜单与教师评分不一致')
                menu_sha = digest(raw['actions'])
                row = self._make_row(raw, pending['logits'], pending['obs_sha256'], sub)
                started = time.perf_counter()
                picked = self._choose(row, pending['logits'])
                elapsed = (time.perf_counter() - started) * 1000
                index = picked['index']
                row.update(chosen=index, chosen_index=index, chosen_token=picked['token'],
                           behavior_logprob=picked['logprob'], selection=picked['selection'], temperature=picked['temperature'],
                           student_ms=elapsed, latency_ms=elapsed, menu_sha256=menu_sha)
                if 'policy_logprob' in picked:
                    row['policy_logprob'] = picked['policy_logprob']
                if self.chooser is not None and pending['rows'] > 1:
                    if 'input_tokens' in picked:
                        # 使用实际前向的计数，避免仅为统计再次编码整个长盘面。
                        # 原始历史仍保留，删除数量可由离线审计重建。
                        row['input_tokens'] = picked['input_tokens']
                    else:
                        from .data import prompt_details
                        _, ids, info = prompt_details(row, self.chooser.tokenizer,
                                                     limit=getattr(self.chooser, "context_limit", 4096))
                        row['input_tokens'] = len(ids)
                        row['history_removed'] = info['history_removed']
                if pending['rows'] > 1 and not getattr(self, 'without_teacher', False):
                    self.samples.append(row)
                self.history.append({'event': 'own_decision', 'turn': self._client.result.turns,
                                     'action': row['legal_actions'][index], 'subdecision': sub})
                self.decisions.append({'forced': pending['rows'] == 1, 'chosen': index, 'student_ms': elapsed,
                    'selection': picked['selection'], 'legal_count': pending['rows'],
                    'action_kind': row['legal_actions'][index].get('kind'),
                    'turn': self._client.result.turns, 'phase': row.get('view', {}).get('phase'),
                    'model_called': self.chooser is not None and pending['rows'] > 1,
                    'input_tokens': row.get('input_tokens')})
                reply = self._commit(pending, index, menu_sha)
            else:
                raise ValueError('子选择数量超过上限')
            response = bytes.fromhex(reply['response'])
            self.responses.append(response.hex())
            player = body[1] if msg == C.MSG_SELECT_SUM else (body[0] if body else 0)
            return SelectResult(msg, player, auto_response=response, note='动作蒸馏', cards=standard.cards)
        except Exception as exc:
            self.failures.append(f'{type(exc).__name__}: {exc}')
            raise


class StudentPolicy(DistillPolicy):
    """不加载教师网络，直接使用原生客户端观察与响应编码。"""
    without_teacher = True

    def __init__(self, native, identity, *, chooser, texts, metadata, seed=0, selection='greedy', temperature=1.):
        super().__init__(None, identity, chooser=chooser, texts=texts, metadata=metadata, seed=seed,
                         selection=selection, temperature=temperature)
        self.native, self.config = native, dict(identity['client_config'])

    def bind(self, client):
        from mirrorforce.netduel.agent_client import AgentClientDuel
        self._client, self.seat = client, client.result.our_player
        kwargs = {'opponent_main': client.main, 'opponent_extra': client.extra} if self.config.get('public_opponent_recipe') else {}
        self.observer = AgentClientDuel(self.native, self.seat, client.main, client.extra, self.config, **kwargs)
        self.active, self.identity_verified = True, True

    def _local_pending(self):
        from tools.mf_runtime_policy_service import observation_sha256
        _, msg, rows = self.observer.prompt()
        return {'pending': {'msg': int(msg), 'rows': len(rows), 'logits': [0.] * len(rows),
                            'obs_sha256': observation_sha256(self.observer.observation())}}

    def _source_prompt(self):
        responses = [r for m, b in self.pending if (r := self.observer.feed(m, b)) is not None]
        self.pending = []
        if len(responses) > 1:
            raise ValueError('一次消息批产生多个原生响应')
        return {'response': bytes(responses[0]).hex(), 'forced': True} if responses else self._local_pending()

    def _view(self, pending):
        from .data import public_view
        return public_view(self.observer, self.observer.prompt()[2])

    def _commit(self, pending, index, menu_sha):
        out = self.observer.step(index)
        return {'response': bytes(out).hex()} if out is not None else self._local_pending()

    def on_duel_end(self, result):
        try:
            if self.active:
                for msg, body in self.pending:
                    self.observer.feed(msg, body)
                self.pending = []
        except Exception as exc:
            self.failures.append(f'{type(exc).__name__}: {exc}')
        self.active = False
