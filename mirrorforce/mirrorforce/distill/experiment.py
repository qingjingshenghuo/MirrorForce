"""分阶段子进程、费用上限及三条实验分支；不创建收费云实例。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def evaluation_command(cfg, py, root, name, adapter):
    """固定WinBot评测无需教师网络；保留显式冠军对手配置兼容入口。"""
    if not cfg.get('windbot_exe'):
        return None
    command = py + ['mirrorforce.distill.arena', '--rules', str(root / 'demonstrations' / 'manifest.json'),
        '--adapter', str(adapter), '--deck', cfg['deck'], '--cards-db', cfg['cards_db'],
        '--script-root', cfg['script_root'], '--native', cfg['native'],
        '--windbot-exe', cfg['windbot_exe'], '--out', str(root / f'eval-{name}'),
        '--seed', str(cfg.get('eval_seed', 300100)), '--pairs', str(cfg.get('eval_pairs', 50))]
    if cfg.get('prompt_style'):
        command += ['--prompt-style', cfg['prompt_style']]
    return command


def _main(argv, stack):
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
                          "initial_sft": cfg.get("initial_sft", {}),
                          "branch_training": cfg.get("branch_training", {}),
                          "requires_real_data_coverage_audit": True,
                          "teacher_command": cfg["teacher_command"]}, ensure_ascii=False, indent=2))
        return
    root.mkdir(parents=True, exist_ok=a.resume)
    from .costs import ExperimentLock, CostJournal
    run_lock = stack.enter_context(ExperimentLock(root))
    cost_journal = CostJournal(root, rate)
    cost_journal.recover()
    from .stages import StageLedger, source_identity, file_hash
    resources = {key: file_hash(cfg[key]) for key in ("deck", "cards_db", "native")}
    if cfg.get('windbot_exe'):
        bot_root = Path(cfg['windbot_exe']).parent
        for item in sorted(bot_root.rglob('*')):
            if item.is_file() and item.suffix in ('.exe', '.dll', '.json', '.config'):
                resources[str(item.resolve())] = file_hash(item)
    script_root = Path(cfg["script_root"])
    for item in sorted(script_root.rglob("*")):
        if item.is_file() and ("script" in item.relative_to(script_root).parts or item.suffix in (".json", ".npz", ".txt", ".ckpt")):
            resources[str(item.resolve())] = file_hash(item)
    for arg in cfg["teacher_command"]:
        if Path(arg).is_file():
            resources[str(Path(arg).resolve())] = file_hash(arg)
    stage_ledger = StageLedger(root, cfg, a.resume, {"source": source_identity(), "resources": resources})
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start = time.monotonic()
    spent = cost_journal.spent
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
        remaining()
        attempt_id = cost_journal.begin(name)
        begun, svc = time.monotonic(), None
        status = "failed"
        try:
            stage_ledger.mark(name, identity_command, "running")
            if teacher:
                if socket.exists():
                    raise ValueError("教师 socket 已存在；请使用独立路径，避免接入未知进程")
                svc_log = (root / f"{name}-teacher.log").open("w")
                svc = subprocess.Popen(cfg["teacher_command"], cwd=cwd, env=env, stdout=svc_log, stderr=subprocess.STDOUT,
                                       pass_fds=(run_lock.fileno,))
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
                               check=True, timeout=remaining(), pass_fds=(run_lock.fileno,))
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
            cost_journal.settle(attempt_id, status)
            if name == "final-report" and (root / "experiment-summary.json").is_file():
                from .costs import atomic_json
                summary_path = root / "experiment-summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["costs"] = cost_journal.ledger
                summary["cost_snapshot_includes_report_stage"] = True
                atomic_json(summary_path, summary)
            artifacts = []
            if status == "ok":
                artifacts = [root / f"{name}.log"]
                if output.is_dir():
                    artifacts.extend(p for p in output.rglob("*") if p.is_file())
                if name == "final-report":
                    artifacts.extend([root / "experiment-summary.json", root / "variants.json"])
            stage_ledger.mark(name, identity_command, status, artifacts)

    def collect(name, seed, pairs, limit, adapter=None):
        nonlocal current_stage
        current_stage = name
        command = py + ["mirrorforce.distill.collect"] + common + ["--out", str(root / name), "--seed", str(seed),
                   "--pairs", str(pairs), "--max-samples", str(limit), "--budget-yuan", str(cap)]
        if adapter:
            command += ["--student", "--adapter", str(adapter)]
            if cfg.get('prompt_style'):
                command += ['--prompt-style', cfg['prompt_style']]
            command += ["--context-limit", str(cfg.get("context_limit", 16384))]
            if seed >= 300000:
                command += ["--selection", "greedy", "--independent-student"]
        run(name, command, True)
        summary = json.loads((root / name / "summary.json").read_text())
        if summary["failures"] or summary["stop"] == "budget" or summary["games"] == 0:
            raise RuntimeError("采样异常、空结果或预算不足；保留日志并停止，不静默继续")
        return root / name / "samples.jsonl"

    def train(name, data, objective, adapter=None):
        options = cfg.get("initial_sft" if name == "sft" else "branch_training", {})
        command = py + ["mirrorforce.distill.train", "--data", str(data), "--out", str(root / name),
                        "--objective", objective, "--steps", str(options.get("steps", cfg.get("steps", 200))),
                        "--base", cfg.get("base", "Qwen/Qwen2.5-1.5B-Instruct"),
                        "--revision", cfg.get("revision", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
                        "--context-limit", str(cfg.get("context_limit", 16384)),
                        "--tokens-per-update", str(options.get("tokens_per_update", cfg.get("tokens_per_update", 16384))),
                        "--save-every", str(options.get("save_every", cfg.get("save_every", 50))),
                        "--keep-last", str(options.get("keep_last", cfg.get("keep_last", 0))),
                        "--min-observed-coverage", str(options.get("min_observed_coverage", 0)),
                        "--token-budget", str(options.get("token_budget", cfg.get("token_budget", 2000000)))]
        if adapter:
            command += ["--adapter", str(adapter)]
        if cfg.get('prompt_style'):
            command += ['--prompt-style', cfg['prompt_style']]
        if options.get('permute_menus', False):
            command += ['--permute-menus']
        if options.get('action_lr', cfg.get('action_lr')) is not None:
            command += ['--action-lr', str(options.get('action_lr', cfg.get('action_lr')))]
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
        command = evaluation_command(cfg, py, root, name, adapter)
        if command:
            run(f'eval-{name}', command)
        else:
            collect(f"eval-{name}", cfg.get('eval_seed', 300100), cfg.get("eval_pairs", 50), 100000, adapter)
        audit_command = py + ["mirrorforce.distill.audit", "--data", str(validation), "--adapter", adapter,
            '--context-limit', str(cfg.get('context_limit', 16384))]
        if cfg.get('prompt_style'):
            audit_command += ['--prompt-style', cfg['prompt_style']]
        run(f"audit-{name}", audit_command)
    (root / "variants.json").write_text(json.dumps(variants, indent=2), encoding="utf-8")
    run("final-report", py + ["mirrorforce.distill.report", "--root", str(root)])


def main(argv=None):
    from contextlib import ExitStack
    with ExitStack() as stack:
        return _main(argv, stack)


if __name__ == "__main__":
    main()
