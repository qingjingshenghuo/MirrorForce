"""独占运行与可跨硬中断恢复的保守费用台账。"""
import json
import math
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ExperimentLock:
    """Linux flock 为存活证据；PID 仅用于诊断，不决定锁是否有效。"""
    def __init__(self, root):
        self.path = Path(root) / "experiment.lock"
        self.handle = None

    def __enter__(self):
        import fcntl
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("实验仍被存活进程锁定，禁止并行恢复") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "acquired_wall": time.time()}))
        self.handle.flush()
        return self

    @property
    def fileno(self):
        return self.handle.fileno()

    def __exit__(self, *exc):
        if self.handle is not None:
            # 不主动 LOCK_UN：子进程继承同一个 open-file description，父进程退出时仍须持锁。
            self.handle.close()
            self.handle = None


class CostJournal:
    """必须在 ExperimentLock 内使用；恢复包含不可观察空闲区间，不代表云账单。"""
    def __init__(self, root, rate, clock=time.time):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("费用单价必须有限且为正")
        self.root, self.rate, self.clock = Path(root), rate, clock
        self.path, self.active_path = self.root / "costs.json", self.root / "cost-active.json"
        self.ledger = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        if not isinstance(self.ledger, list) or any(not math.isfinite(x["estimated_yuan"]) or x["estimated_yuan"] < 0 for x in self.ledger):
            raise ValueError("费用台账损坏")
        ids = [x["attempt_id"] for x in self.ledger if "attempt_id" in x]
        if len(ids) != len(set(ids)):
            raise ValueError("费用台账包含重复 attempt_id")

    @property
    def spent(self):
        return sum(x["estimated_yuan"] for x in self.ledger)

    def begin(self, stage):
        if self.active_path.exists():
            raise RuntimeError("上次费用尚未恢复结算")
        record = {"attempt_id": uuid.uuid4().hex, "stage": stage, "start_wall": self.clock(),
                  "rate": self.rate, "pid": os.getpid()}
        atomic_json(self.active_path, record)
        return record["attempt_id"]

    def settle(self, attempt_id, status, recovered=False):
        if not self.active_path.exists():
            if any(x.get("attempt_id") == attempt_id for x in self.ledger):
                return
            raise RuntimeError("费用 attempt 不存在")
        active = json.loads(self.active_path.read_text(encoding="utf-8"))
        if active["attempt_id"] != attempt_id:
            raise ValueError("结算的 attempt_id 不匹配")
        if not any(x.get("attempt_id") == attempt_id for x in self.ledger):
            now = self.clock()
            seconds = max(0., now - active["start_wall"])
            if not all(math.isfinite(x) for x in (now, active["start_wall"], active["rate"])) or active["rate"] <= 0:
                raise ValueError("费用记录的时间或单价无效")
            if now < active["start_wall"]:
                raise ValueError("墙钟发生倒退，无法保守结算；保留 active 记录等待人工核对")
            record = {**active, "status": status, "end_wall": now, "seconds": seconds,
                      "estimated_yuan": seconds / 3600 * active["rate"], "recovered": recovered,
                      "cost_note": "墙钟保守估算，硬中断恢复包含不可观察空闲区间；不是云账单"}
            self.ledger.append(record)
            # 先原子发布结算，再删除 active。崩溃于两者之间时靠 attempt_id 去重。
            atomic_json(self.path, self.ledger)
        self.active_path.unlink()

    def recover(self):
        if self.active_path.exists():
            active = json.loads(self.active_path.read_text(encoding="utf-8"))
            self.settle(active["attempt_id"], "interrupted", recovered=True)
