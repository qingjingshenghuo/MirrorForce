"""离线 SFT、软蒸馏和在线轮次共用的可恢复训练入口。"""
import argparse
import json
import random
import hashlib
import os
import time
from pathlib import Path
from .data import read_rows


def configure_reproducibility(device):
    """固定同一运行环境的计算路径，避免只有随机种子相同但梯度漂移。"""
    import torch
    cuda = str(device).startswith("cuda")
    if cuda:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if torch.cuda.is_initialized() and workspace not in (":4096:8", ":16:8"):
            raise RuntimeError("CUDA已初始化但未配置确定性cuBLAS工作区，请在新进程训练")
        if workspace is not None and workspace not in (":4096:8", ":16:8"):
            raise ValueError("训练要求CUBLAS_WORKSPACE_CONFIG为:4096:8或:16:8")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
            "deterministic_algorithms": True, "tf32": False,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda else None,
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "capability": list(torch.cuda.get_device_capability(device)) if cuda else None}


def token_coverage(lengths, quota):
    """审计每种长度能否参与完整样本的精确配额更新。"""
    if not lengths or min(lengths) <= 0 or quota <= 0:
        raise ValueError("长度与配额必须为正数")
    coins = sorted(set(lengths))
    reachable = bytearray(quota + 1)
    reachable[0] = 1
    for total in range(1, quota + 1):
        reachable[total] = any(n <= total and reachable[total - n] for n in coins)
    eligible = [n <= quota and bool(reachable[quota - n]) for n in lengths]
    return {"quota": quota, "quota_reachable": bool(reachable[quota]),
            "input_lengths": lengths, "eligible": eligible,
            "excluded_indexes": [i for i, ok in enumerate(eligible) if not ok],
            "eligible_fraction": sum(eligible) / len(eligible)}


def exact_token_sampler(lengths, quota):
    """无截断地凑齐每次更新的有效 token 数；不可达就明确拒绝。"""
    groups = {}
    for i, length in enumerate(lengths):
        groups.setdefault(length, []).append(i)
    coins = sorted(groups)
    reachable = bytearray(quota + 1)
    reachable[0] = 1
    for total in range(1, quota + 1):
        reachable[total] = any(length <= total and reachable[total - length] for length in coins)
    if not reachable[quota]:
        raise ValueError("当前数据无法凑齐指定 token 配额；不能声称等预算对照")
    excluded = [i for i, n in enumerate(lengths) if n > quota or not reachable[quota - n]]
    if excluded:
        raise ValueError(f"精确token配额永久排除样本索引：{excluded}；请先审计并调整共同配额")

    def draw(rng):
        remainder, indexes = quota, []
        while remainder:
            possible = [n for n in coins if n <= remainder and reachable[remainder - n]]
            length = rng.choice(possible)
            indexes.append(rng.choice(groups[length]))
            remainder -= length
        return indexes
    return draw


def action_loss(scores, rows, objective, action_token_ids=None):
    import torch
    losses = []
    for score, row in zip(scores, rows):
        n = len(row["teacher_logits"])
        target = torch.tensor(row["teacher_logits"], device=score.device, dtype=torch.float32)
        if action_token_ids is None:
            raise ValueError("必须提供标准 LM 的动作 token 映射")
        ids = torch.tensor(action_token_ids[:n], device=score.device, dtype=torch.long)
        student_logp = score.index_select(0, ids).float().log_softmax(-1)
        if objective == "sft":
            losses.append(-score.float().log_softmax(-1)[ids[target.argmax()]])
        else:
            teacher_logp = target.log_softmax(-1)
            losses.append((teacher_logp.exp() * (teacher_logp - student_logp)).sum())
    return torch.stack(losses).mean()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--objective", choices=["sft", "kd"], required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--adapter", help="只加载权重，开启新实验或新轮次")
    p.add_argument("--resume", help="恢复同一数据和目标的优化器及随机状态")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument('--action-lr', type=float, help='新增动作行学习率，缺省与LoRA相同')
    p.add_argument('--permute-menus', action='store_true', help='在每步同步置换菜单与监督；仅支持非精确配额组批')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--context-limit", type=int, help="输入加动作的token上限，缺省继承检查点或使用4096")
    p.add_argument('--prompt-style', choices=['compact', 'named_menu'])
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--keep-last", type=int, default=0, help="仅保留本任务最新N个检查点，0表示全保留；发布阶段账本前清理")
    p.add_argument("--min-observed-coverage", type=float, default=0., help="实际独立状态访问率下限；冒烟可为0，正式实验需预先指定")
    p.add_argument("--token-budget", type=int, default=2000000, help="有效输入 token 上限，下一更新超限则停止")
    p.add_argument("--tokens-per-update", type=int, default=0, help="正数时按确切有效 token 数组批；覆盖 accumulate")
    a = p.parse_args(argv)
    if a.permute_menus and a.tokens_per_update or a.action_lr is not None and not 0 < a.action_lr < float('inf'):
        p.error('菜单增强不能冒充精确等token组批，动作学习率必须为有限正数')
    if a.adapter and a.resume or min(a.steps, a.accumulate, a.save_every, a.token_budget) <= 0 or a.tokens_per_update < 0 or a.keep_last < 0 or not 0 <= a.min_observed_coverage <= 1:
        p.error("参数冲突或预算不是正数")
    import torch
    import numpy as np
    runtime_identity = configure_reproducibility(a.device)
    np.random.seed(a.seed)
    from .student import Student
    from .data import prompt_text, digest, SERIALIZER_VERSION
    from .stages import source_identity, file_hash
    rows = read_rows(a.data)
    if not rows:
        raise ValueError("没有训练样本")
    data_sha = hashlib.sha256(a.data.read_bytes()).hexdigest()
    student = Student(a.base, a.resume or a.adapter, a.device, True, a.revision, a.seed, context_limit=a.context_limit, prompt_style=a.prompt_style)
    if student.prompt_style == 'compact':
        lengths = [len(prompt_text(row, student.tokenizer, limit=student.context_limit)[1]) for row in rows]
    else:
        from .data import prompt_details
        lengths = [len(prompt_details(row, student.tokenizer, student.context_limit, student.prompt_style)[1]) for row in rows]
    coverage = token_coverage(lengths, a.tokens_per_update) if a.tokens_per_update else {
        "input_lengths": lengths, "eligible": [True] * len(rows), "excluded_indexes": [], "eligible_fraction": 1.}
    # 审计独立于输出目录，配置失败也必须留下可检查的排除明细。
    audit_path = a.out.parent / (a.out.name + "-coverage.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    coverage["excluded_metadata"] = [{"index": i, "game_id": rows[i].get("game_id"),
        "turn": rows[i].get("view", {}).get("turn"),
        "phase": rows[i].get("view", {}).get("phase"),
        "action_kinds": sorted({str(action.get("kind", "unknown")) for action in rows[i].get("legal_actions", [])}),
        "prompt_msg": rows[i].get("view", {}).get("prompt_msg")} for i in coverage["excluded_indexes"]]
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != coverage:
        raise ValueError("覆盖审计文件已存在且配置不同，请使用新的输出目录")
    audit_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    sampler = exact_token_sampler(lengths, a.tokens_per_update) if a.tokens_per_update else None
    from .optimizer import ActionRowAdamW
    optimizer = ActionRowAdamW(student, a.lr, a.action_lr)
    rng, start, tokens = random.Random(a.seed), 0, 0
    sample_counts = [0] * len(rows)
    source = source_identity()
    base_manifest = Path(student.base) / "download-manifest.json"
    identity = {"data_sha": data_sha, "objective": a.objective, "accumulate": a.accumulate,
                "source": source, "serializer_version": SERIALIZER_VERSION,
                "tokenizer_sha256": digest(student.tokenizer.get_vocab()),
                "base_manifest_sha256": file_hash(base_manifest) if base_manifest.is_file() else None,
                "lr": a.lr, "seed": a.seed, "tokens_per_update": a.tokens_per_update,
                "base": student.base, "revision": student.revision, "action_ids": student.ids,
                "context_limit": student.context_limit,
                "prompt_style": student.prompt_style,
                "prompt_version": student.prompt_version,
                "min_observed_coverage": a.min_observed_coverage,
                "loss_version": "standard_lm_full_vocab_sft_legal_kd/v2",
                "runtime": runtime_identity}
    identity['optimizer_format'] = optimizer.format
    identity.update(action_lr=a.lr if a.action_lr is None else a.action_lr, permute_menus=a.permute_menus)
    if a.resume:
        state = torch.load(Path(a.resume) / "training.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("恢复时数据或训练配置发生改变")
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["random"])
        torch.set_rng_state(state["torch_random"])
        np.random.set_state(state["numpy_random"])
        sample_counts = state["sample_counts"]
        if torch.cuda.is_available() and state["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_random"])
        start, tokens = state["step"], state["tokens"]
        if a.steps < start or a.token_budget < tokens:
            raise ValueError("恢复预算小于检查点已完成的更新或 token 数")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "run.json").write_text(json.dumps({**vars(a), **identity, **source}, default=str, indent=2), encoding="utf-8")

    def save(step):
        path = a.out / f"step-{step:06d}"
        pending = a.out / f".step-{step:06d}.incomplete"
        student.save(pending)
        torch.save({"identity": identity, "step": step, "tokens": tokens,
                    "optimizer": optimizer.state_dict(), "random": rng.getstate(),
                    "torch_random": torch.get_rng_state(),
                    "numpy_random": np.random.get_state(), "sample_counts": sample_counts,
                    "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   pending / "training.pt")
        # 完整权重与优化器全部写入后才发布可恢复目录。
        pending.rename(path)
        if a.keep_last:
            import shutil
            for old in sorted(a.out.glob("step-*"))[:-a.keep_last]:
                if old.resolve().parent != a.out.resolve() or not (old / "training.pt").is_file():
                    raise ValueError("拒绝清理非本任务检查点")
                shutil.rmtree(old)
        return path

    last_saved, step = None, start
    with (a.out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(start + 1, a.steps + 1):
            before = rng.getstate()
            indexes = sampler(rng) if sampler else [rng.randrange(len(rows)) for _ in range(a.accumulate)]
            needed = sum(lengths[i] for i in indexes)
            batch = [rows[i] for i in indexes]
            if a.permute_menus:
                from .augmentation import permute_training_menu
                from .data import prompt_details, validate
                batch = []
                for index in indexes:
                    order = list(range(len(rows[index]['legal_actions'])))
                    rng.shuffle(order)
                    batch.append(validate(permute_training_menu(rows[index], order)))
                needed = sum(len(prompt_details(row,student.tokenizer,student.context_limit,student.prompt_style)[1]) for row in batch)
            if tokens + needed > a.token_budget:
                rng.setstate(before)
                step -= 1
                break
            optimizer.zero_grad(set_to_none=True)
            if str(a.device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            begun = time.monotonic()
            total = 0.
            for row in batch:
                scores, _ = student.logits([row])
                loss = action_loss(scores, [row], a.objective, student.ids) / len(indexes)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限值")
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            if str(a.device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.monotonic() - begun
            tokens += needed
            for index in indexes:
                sample_counts[index] += 1
            metric = {"step": step, "loss": total, "effective_tokens": tokens,
                      "update_decisions": len(indexes), "update_seconds": seconds,
                      "input_tokens_per_second": needed / seconds,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(a.device).startswith("cuda") else None,
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if str(a.device).startswith("cuda") else None}
            log.write(json.dumps(metric) + "\n")
            log.flush()
            print(json.dumps(metric), flush=True)
            if step % a.save_every == 0:
                last_saved = save(step)
        if last_saved is None or last_saved.name != f"step-{step:06d}":
            last_saved = save(step)
    (a.out / "result.json").write_text(json.dumps({"checkpoint": str(last_saved.resolve()), "steps": step,
        "effective_tokens": tokens, "note": "对照必须核对实际 token 数；相同上限不等于实际预算相同"}, ensure_ascii=False), encoding="utf-8")
    (a.out / "sampling-audit.json").write_text(json.dumps({"input_lengths": lengths,
        "sample_counts": sample_counts, "total_decisions": sum(sample_counts),
        "unique_decisions": sum(n > 0 for n in sample_counts),
        "observed_coverage": sum(n > 0 for n in sample_counts) / len(rows)}, indent=2), encoding="utf-8")
    if sum(n > 0 for n in sample_counts) / len(rows) < a.min_observed_coverage:
        raise ValueError("实际样本覆盖率低于预定下限；保留检查点和审计，不接受为完成阶段")


if __name__ == "__main__":
    main()
