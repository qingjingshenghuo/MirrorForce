"""让发布模型加入本机可视客户端创建的房间，保存可核对的推理记录。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

from mirrorforce.netduel import agent_wire as wire
from mirrorforce.netduel.agent_policy import RemotePolicy
from mirrorforce.netduel.cards import CardPool, load_ydk
from mirrorforce.netduel.client import NetDuelClient


class LoggedPolicy(RemotePolicy):
    """在本地控制台记录真实模型决策次数，便于确认对战没有停滞。"""

    def _parse(self, msg, body, ctx):
        before = len(self.decisions)
        response = super()._parse(msg, body, ctx)
        if any(not row.get('forced', False) for row in self.decisions[before:]):
            print(f'AI 已完成选择：第 {self._client.result.turns} 回合，累计 {len(self.decisions)} 项决策。', flush=True)
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/tmp/mirrorforce-policy.sock')
    parser.add_argument('--host', help='默认使用 WSL 默认网关，即 Windows 主机')
    parser.add_argument('--port', type=int, default=7911)
    parser.add_argument('--name', default='MirrorForce')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    host = args.host
    if host is None:
        route = subprocess.check_output(['ip', 'route', 'show', 'default'], text=True).split()
        host = route[route.index('via') + 1]
    bundle = Path('/opt/mirrorforce/assets')
    output = args.out or Path('/opt/mirrorforce/runs') / ('human-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    connection = wire.connect('unix:' + args.socket, 30)
    try:
        identity = wire.call(connection, {'op': 'identity'})
    finally:
        connection.close()
    expected = 'dd71087ce1bf827d9b8360c310ae4be02714b50b9535f4959544d24372d28561'
    if identity.get('checkpoint_sha256') != expected or identity.get('weights') != 'iterate' \
            or identity.get('selection') != 'greedy':
        raise RuntimeError('服务并非预期的发布策略，拒绝加入')
    main_deck, extra_deck, _ = load_ydk(bundle / 'decks/SkyStriker.ydk')
    policy = LoggedPolicy('unix:' + args.socket, identity, timeout=180)
    client = NetDuelClient(host=host, port=args.port, name=args.name, main=main_deck,
                           extra=extra_deck, side=[], policy=policy, timeout=1800,
                           max_options=192, card_pool=CardPool(bundle / 'cards.cdb', bundle / 'code_list.txt'))
    print(f'正在连接本机房间 {host}:{args.port}；请在客户端选择配套卡组并开始。', flush=True)
    try:
        result = client.run()
        (output / 'result.json').write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        print(json.dumps(result.as_dict(), ensure_ascii=False), flush=True)
        if result.error:
            raise RuntimeError('对局未正常运行：' + result.error)
    finally:
        client.close()
        (output / 'policy-report.json').write_text(json.dumps(policy.report(), ensure_ascii=False, indent=2))
        print(f'推理记录：{output}', flush=True)


if __name__ == '__main__':
    main()
