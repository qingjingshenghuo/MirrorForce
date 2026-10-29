"""真实硬中断、存活锁与幂等费用结算。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from mirrorforce.distill.costs import ExperimentLock, CostJournal, atomic_json
from mirrorforce.distill.budget import Budget

pytestmark = pytest.mark.skipif(os.name != "posix", reason="部署环境使用 Linux flock")


def start_holder(root, rate=2):
    source = """
import sys,time
from mirrorforce.distill.costs import ExperimentLock,CostJournal
with ExperimentLock(sys.argv[1]):
    journal=CostJournal(sys.argv[1],float(sys.argv[2]))
    journal.begin('train')
    print('READY',flush=True)
    time.sleep(120)
"""
    proc = subprocess.Popen([sys.executable, "-c", source, str(root), str(rate)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "READY"
    return proc


def test_sigkill_recovery_charges_unobserved_interval_once(tmp_path):
    process = start_holder(tmp_path)
    active = json.loads((tmp_path / "cost-active.json").read_text())
    assert set(("attempt_id", "start_wall", "rate")) <= active.keys()
    process.kill()
    process.wait(timeout=10)
    with ExperimentLock(tmp_path):
        # 不假装知道宕机后实例何时停止，恢复保守计入30分钟间隔。
        journal = CostJournal(tmp_path, 2, clock=lambda: active["start_wall"] + 1800)
        journal.recover()
        assert journal.spent == 1.
        assert journal.ledger[0]["attempt_id"] == active["attempt_id"]
        assert journal.ledger[0]["recovered"] and journal.ledger[0]["status"] == "interrupted"
        journal.recover()
        assert len(journal.ledger) == 1 and not (tmp_path / "cost-active.json").exists()


def test_settled_but_active_residue_not_double_charged(tmp_path):
    with ExperimentLock(tmp_path):
        now = [1000.]
        journal = CostJournal(tmp_path, 2, clock=lambda: now[0])
        attempt = journal.begin("sft")
        active = json.loads(journal.active_path.read_text())
        now[0] += 1800
        journal.settle(attempt, "ok")
        assert journal.spent == 1.
        # 模拟costs.json已rename，而active尚未unlink时被kill。
        atomic_json(journal.active_path, active)
        restored = CostJournal(tmp_path, 2, clock=lambda: now[0] + 3600)
        restored.recover()
        assert restored.spent == 1. and len(restored.ledger) == 1
        restored.settle(attempt, "ok")
        assert restored.spent == 1.


def test_live_lock_refuses_parallel_resume_and_stale_pid_is_not_lock(tmp_path):
    process = start_holder(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="存活进程锁定"):
            with ExperimentLock(tmp_path):
                pass
    finally:
        process.kill()
        process.wait(timeout=10)
    # 遗留PID文件仍存在，但进程死亡后kernel锁消失，允许恢复。
    assert (tmp_path / "experiment.lock").exists()
    with ExperimentLock(tmp_path):
        CostJournal(tmp_path, 2).recover()


def test_inherited_lock_keeps_orphan_worker_protected(tmp_path):
    source = """
import subprocess,sys,time
from mirrorforce.distill.costs import ExperimentLock
with ExperimentLock(sys.argv[1]) as lock:
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)'],pass_fds=(lock.fileno,))
    print(child.pid,flush=True)
    time.sleep(120)
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    assert int(process.stdout.readline()) > 0
    process.kill()
    process.wait(timeout=10)
    with pytest.raises(RuntimeError):
        with ExperimentLock(tmp_path):
            pass
    time.sleep(1.7)
    with ExperimentLock(tmp_path):
        pass


def test_recovered_spend_preserves_evaluation_reserve(tmp_path):
    with ExperimentLock(tmp_path):
        clock = [1000.]
        journal = CostJournal(tmp_path, 2, clock=lambda: clock[0])
        journal.begin("train")
        restored = CostJournal(tmp_path, 2, clock=lambda: 1000. + 150 * 3600)
        restored.recover()
        budget = Budget(450, 2, 150)
        assert restored.spent == 300
        assert budget.available(restored.spent) == 0
        assert budget.available(restored.spent, evaluation=True) == 150


def test_attempt_identity_and_corrupt_ledger_refused(tmp_path):
    with ExperimentLock(tmp_path):
        journal = CostJournal(tmp_path, 2, clock=lambda: 1000.)
        attempt = journal.begin("stage")
        with pytest.raises(ValueError):
            journal.settle("wrong-id", "ok")
        with pytest.raises(RuntimeError):
            journal.begin("second")
        journal.settle(attempt, "ok")
        records = journal.ledger + journal.ledger
        atomic_json(journal.path, records)
        with pytest.raises(ValueError, match="重复"):
            CostJournal(tmp_path, 2)


def test_clock_rollback_cannot_silently_charge_zero(tmp_path):
    with ExperimentLock(tmp_path):
        journal = CostJournal(tmp_path, 2, clock=lambda: 1000.)
        attempt = journal.begin("train")
        restored = CostJournal(tmp_path, 2, clock=lambda: 900.)
        with pytest.raises(ValueError, match="倒退"):
            restored.recover()
        assert restored.active_path.exists() and not restored.path.exists()
