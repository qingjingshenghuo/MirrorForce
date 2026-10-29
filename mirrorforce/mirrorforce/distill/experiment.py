"""分阶段子进程、费用上限及三条实验分支；不创建收费云实例。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="复用已验证阶段，保留累计费用")
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    required = ["python", "cwd", "out", "teacher_command", "teacher_socket", "deck", "cards_db", "script_root", "native", "hourly_yuan"]
    if any(k not in cfg for k in required):
        raise ValueError("配置缺少字段：" + str(required))
    rate = float(cfg["hourly_yuan"])
    cap = float(cfg.get("budget_yuan", 450))
    from .budget import Budget
    budget = Budget(cap, rate, float(cfg.get("evaluation_reserve_yuan", min(150., cap / 3))))
    if not 0 < cap <= 500 or rate <= 0:
        raise ValueError("必须填写实际小时价格，进程预算不能超过 500 元")
    root, cwd = Path(cfg["out"]).resolve(), Path(cfg["cwd"]).resolve()
    py = [cfg["python"], "-u", "-m"]
    socket = Path(cfg["teacher_socket"])
    common = ["--teacher", str(socket), "--deck", cfg["deck"], "--cards-db", cfg["cards_db"],
              "--script-root", cfg["script_root"], "--native", cfg["native"], "--hourly-yuan", str(rate)]
    stages = ["demonstrations", "validation", "sft", "continued-sft", "offline-kd",
              "online-1", "online-2", "online-3", "evaluation"]
    if a.dry_run:
        print(json.dumps({"stages": stages, "root": str(root), "budget_yuan": cap,
                          "teacher_command": cfg["teacher_command"]}, ensure_ascii=False, indent=2))
        return
    root.mkdir(parents=True, exist_ok=a.resume)
    from .stages import StageLedger
    stage_ledger = StageLedger(root, cfg, a.resume)
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start = time.monotonic()
    ledger = json.loads((root / "costs.json").read_text()) if a.resume and (root / "costs.json").exists() else []
    spent = sum(item["estimated_yuan"] for item in ledger)
    deadline = start + (cap - spent) / rate * 3600
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")

    current_stage = "setup"

    def remaining(evaluation=None):
        evaluation = budget.is_evaluation(current_stage) if evaluation is None else evaluation
        seconds = deadline - time.monotonic() - (0 if evaluation else budget.reserve / rate * 3600)
        if seconds <= 0:
            raise TimeoutError("达到进程费用上限；请关闭云实例停止空闲计费")
        return seconds

    def run(name, command, teacher=False):
        nonlocal current_stage
        current_stage = name
        # 费用余量会随恢复变化，不属于阶段命令身份。
        identity_command = list(command)
        if "--budget-yuan" in identity_command:
            identity_command[identity_command.index("--budget-yuan") + 1] = "runtime-budget"
        if stage_ledger.reusable(name, identity_command):
            return
        attempt = 1
        output = root / name
        if output.exists():
            # 保留失败产物，显式生成新尝试目录，不覆盖已有记录。
            while (root / f"{name}-attempt-{attempt}").exists():
                attempt += 1
            archived = root / f"{name}-attempt-{attempt}"
            output.rename(archived)
            if "mirrorforce.distill.train" in command:
                checkpoints = sorted(archived.glob("step-*/training.pt"))
                if checkpoints:
                    command = list(command)
                    if "--adapter" in command:
                        idx = command.index("--adapter")
                        del command[idx:idx + 2]
                    command += ["--resume", str(checkpoints[-1].parent)]
        stage_ledger.mark(name, identity_command, "running")
        begun, svc = time.monotonic(), None
        status = "failed"
        try:
            if teacher:
                if socket.exists():
                    raise ValueError("教师 socket 已存在；请使用独立路径，避免接入未知进程")
                svc_log = (root / f"{name}-teacher.log").open("w")
                svc = subprocess.Popen(cfg["teacher_command"], cwd=cwd, env=env, stdout=svc_log, stderr=subprocess.STDOUT)
                from tools.mf_runtime_local_match import service_identity
                while True:
                    remaining()
                    if svc.poll() is not None:
                        raise RuntimeError("教师启动失败，请查看日志")
                    try:
                        service_identity(socket, 1)
                        break
                    except (OSError, RuntimeError):
                        if time.monotonic() - begun > 900:
                            raise TimeoutError("教师 900 秒内未就绪")
                        time.sleep(1)
            with (root / f"{name}.log").open("w") as log:
                if "--budget-yuan" in command:
                    command = list(command)
                    command[command.index("--budget-yuan") + 1] = str(remaining() / 3600 * rate)
                subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=remaining())
            status = "ok"
        finally:
            if svc is not None:
                svc.terminate()
                try:
                    svc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    svc.kill()
                    svc.wait()
                svc_log.close()
                # 只移除本轮启动服务所占用的 Unix socket。
                if socket.is_socket():
                    socket.unlink()
            ledger.append({"stage": name, "status": status, "seconds": time.monotonic() - begun,
                           "estimated_yuan": (time.monotonic() - begun) / 3600 * rate})
            (root / "costs.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            artifacts = []
            if status == "ok":
                artifacts = [root / f"{name}.log"]
                if output.is_dir():
                    artifacts.extend(p for p in output.rglob("*") if p.is_file())
            stage_ledger.mark(name, identity_command, status, artifacts)

    def collect(name, seed, pairs, limit, adapter=None):
        nonlocal current_stage
        current_stage = name
        command = py + ["mirrorforce.distill.collect"] + common + ["--out", str(root / name), "--seed", str(seed),
                   "--pairs", str(pairs), "--max-samples", str(limit), "--budget-yuan", str(cap)]
        if adapter:
            command += ["--student", "--adapter", str(adapter)]
            command += ["--context-limit", str(cfg.get("context_limit", 16384))]
            if seed >= 300000:
                command += ["--selection", "greedy", "--independent-student"]
        run(name, command, True)
        summary = json.loads((root / name / "summary.json").read_text())
        if summary["failures"] or summary["stop"] == "budget" or summary["games"] == 0:
            raise RuntimeError("采样异常、空结果或预算不足；保留日志并停止，不静默继续")
        return root / name / "samples.jsonl"

    def train(name, data, objective, adapter=None):
        command = py + ["mirrorforce.distill.train", "--data", str(data), "--out", str(root / name),
                        "--objective", objective, "--steps", str(cfg.get("steps", 200)),
                        "--context-limit", str(cfg.get("context_limit", 16384)),
                        "--tokens-per-update", str(cfg.get("tokens_per_update", 16384)),
                        "--token-budget", str(cfg.get("token_budget", 2000000))]
        if adapter:
            command += ["--adapter", str(adapter)]
        run(name, command)
        result = json.loads((root / name / "result.json").read_text())
        if result["steps"] == 0:
            raise RuntimeError("训练未执行更新")
        return result["checkpoint"]

    data = collect("demonstrations", 100000, cfg.get("data_pairs", 50), 10000)
    validation = collect("validation", 200000, 5, 2000)
    initial = train("sft", data, "sft")
    variants = {"sft": initial}
    for name, objective in [("continued-sft", "sft"), ("offline-kd", "kd")]:
        current = initial
        for round_id in range(1, 4):
            current = train(f"{name}-{round_id}", data, objective, current)
        variants[name] = current
    current = initial
    for round_id in range(1, 4):
        online = collect(f"online-data-{round_id}", 101000 + round_id * 100, cfg.get("online_pairs", 10), 2000, current)
        current = train(f"online-kd-{round_id}", online, "kd", current)
    variants["online-kd"] = current
    for name, adapter in variants.items():
        collect(f"eval-{name}", 300000, cfg.get("eval_pairs", 50), 100000, adapter)
        run(f"audit-{name}", py + ["mirrorforce.distill.audit", "--data", str(validation), "--adapter", adapter])
    (root / "variants.json").write_text(json.dumps(variants, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
