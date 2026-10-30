"""将固定WindBot可执行文件接入仅接收本方消息的本机协议桥。"""
from pathlib import Path
import socket
import subprocess
import os
from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.actions import SelectResult, parse_select
from mirrorforce.netduel.agent_policy import RemotePolicy


class WindBotPolicy(RemotePolicy):
    def __init__(self, identity, exe, deck, database, *, seed=0, timeout=60):
        super().__init__(None, identity, seed=seed, timeout=timeout)
        self.exe, self.deck, self.database = map(lambda p: Path(p).resolve(), (exe, deck, database))
        self.process = self.transport = self.log = None

    def bind(self, client):
        self._client, self.seat = client, client.result.our_player
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(self.timeout)
        self.log = (self.exe.parent / f'bridge-{self.seed}-{self.seat}.log').open('w')
        try:
            self.process = subprocess.Popen(['mono', str(self.exe), 'Deck=SkyStriker',
                f'DeckFile={self.deck}', f'DbPath={self.database}', 'Host=127.0.0.1',
                f'Port={listener.getsockname()[1]}', 'Version=4962', 'Chat=False'],
                cwd=self.exe.parent, env={**os.environ, 'WINDBOT_SEED': str(self.seed)},
                stdout=self.log, stderr=subprocess.STDOUT)
            conn, _ = listener.accept()
            self.transport = P.PacketStream(conn)
            while True:
                op, _ = self.transport.recv(self.timeout)
                if op == P.CTOS.JOIN_GAME:
                    break
            info = client.host_info
            self.transport.send(P.STOC.JOIN_GAME, P._HOST_INFO.pack(*[getattr(info, k) for k in info.__slots__]))
            self.transport.send(P.STOC.TYPE_CHANGE, bytes([self.seat]))
            self.transport.send(P.STOC.DUEL_START)
            self.active = self.identity_verified = True
        except Exception:
            self.on_duel_end(None)
            raise
        finally:
            listener.close()

    def observe_game_message(self, msg, body):
        if self.active:
            self.transport.send(P.STOC.GAME_MSG, bytes([msg]) + bytes(body))

    def _parse(self, msg, body, ctx):
        standard = parse_select(msg, body, ctx)
        try:
            while True:
                op, payload = self.transport.recv(self.timeout)
                if op == P.CTOS.RESPONSE:
                    self.decisions.append({'forced': False})
                    return SelectResult(msg, standard.player, auto_response=payload, cards=standard.cards, note='固定WindBot')
                if op == P.CTOS.SURRENDER:
                    raise RuntimeError('WindBot投降，不得当作正常策略胜负')
        except Exception as exc:
            self.failures.append(str(exc))
            raise

    def on_duel_end(self, result):
        self.active = False
        if self.transport:
            self.transport.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()

    def report(self):
        return {'decisions': len(self.decisions), 'failures': self.failures}
