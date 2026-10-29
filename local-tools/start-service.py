"""启动或复用本机只读策略服务，并检查服务确实就绪。"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import time

from mirrorforce.netduel import agent_wire as wire

ROOT = Path('/opt/mirrorforce')
RUN = ROOT / 'run'
SOCKET = Path('/tmp/mirrorforce-policy.sock')
EXPECTED = 'dd71087ce1bf827d9b8360c310ae4be02714b50b9535f4959544d24372d28561'


def ready():
    if not SOCKET.exists():
        return False
    try:
        connection = wire.connect('unix:' + str(SOCKET), 2)
    except OSError:
        return False
    try:
        identity = wire.call(connection, {'op': 'identity'})
    finally:
        connection.close()
    if identity.get('checkpoint_sha256') != EXPECTED or identity.get('weights') != 'iterate' \
            or identity.get('selection') != 'greedy':
        raise RuntimeError('现有服务身份不同，拒绝复用或替换')
    return True


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'startup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            print('发布模型服务已就绪。', flush=True)
            return
        record = RUN / 'service.json'
        live = False
        if record.exists():
            previous = json.loads(record.read_text())
            proc = Path('/proc') / str(previous['pid']) / 'cmdline'
            live = proc.exists() and b'tools.mf_runtime_policy_service' in proc.read_bytes()
        if not live:
            if SOCKET.exists():
                SOCKET.unlink()
            stamp = time.strftime('%Y%m%d-%H%M%S')
            log_path = RUN / f'service-{stamp}.log'
            output = ROOT / 'runs' / f'service-{stamp}'
            with log_path.open('ab') as log:
                process = subprocess.Popen(['bash', str(Path(__file__).with_name('serve-policy.sh')),
                                            str(SOCKET), str(output)], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            previous = {'pid': process.pid, 'log': str(log_path), 'output': str(output)}
            record.write_text(json.dumps(previous, indent=2))
        print(f"正在加载模型，日志：{previous['log']}", flush=True)
        for _ in range(450):
            if ready():
                print('发布模型服务已就绪。', flush=True)
                return
            proc = Path('/proc') / str(previous['pid'])
            if not proc.exists() or ') Z ' in (proc / 'stat').read_text():
                raise RuntimeError('策略服务启动失败，请检查 ' + previous['log'])
            time.sleep(2)
        raise TimeoutError('启动观察超时；服务可能仍在编译，请检查现有进程和日志，不要直接重复启动')


if __name__ == '__main__':
    main()
