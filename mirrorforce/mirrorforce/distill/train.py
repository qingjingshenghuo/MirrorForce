"""离线 SFT、软蒸馏和在线轮次共用的可恢复训练入口。"""
import argparse
import json
import random
import hashlib
from pathlib import Path
from .data import read_rows


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--context-limit", type=int, help="输入加动作的token上限，缺省继承检查点或使用4096")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--token-budget", type=int, default=2000000, help="有效输入 token 上限，下一更新超限则停止")
    p.add_argument("--tokens-per-update", type=int, default=0, help="正数时按确切有效 token 数组批；覆盖 accumulate")
    a = p.parse_args(argv)
    if a.adapter and a.resume or min(a.steps, a.accumulate, a.save_every, a.token_budget) <= 0 or a.tokens_per_update < 0:
        p.error("参数冲突或预算不是正数")
    import torch
    import numpy as np
    np.random.seed(a.seed)
    from .student import Student
    from .data import prompt_text, digest, SERIALIZER_VERSION
    from .stages import source_identity, file_hash
    rows = read_rows(a.data)
    if not rows:
        raise ValueError("没有训练样本")
    data_sha = hashlib.sha256(a.data.read_bytes()).hexdigest()
    student = Student(a.base, a.resume or a.adapter, a.device, True, a.revision, a.seed, context_limit=a.context_limit)
    lengths = [len(prompt_text(row, student.tokenizer, limit=student.context_limit)[1]) for row in rows]
    sampler = exact_token_sampler(lengths, a.tokens_per_update) if a.tokens_per_update else None
    optimizer = torch.optim.AdamW(student.optimizer_groups(a.lr), lr=a.lr)
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
                "loss_version": "standard_lm_full_vocab_sft_legal_kd/v2"}
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
        return path

    last_saved, step = None, start
    with (a.out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(start + 1, a.steps + 1):
            before = rng.getstate()
            indexes = sampler(rng) if sampler else [rng.randrange(len(rows)) for _ in range(a.accumulate)]
            needed = sum(lengths[i] for i in indexes)
            if tokens + needed > a.token_budget:
                rng.setstate(before)
                step -= 1
                break
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for i in indexes:
                scores, _ = student.logits([rows[i]])
                loss = action_loss(scores, [rows[i]], a.objective, student.ids) / len(indexes)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限值")
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            tokens += needed
            for index in indexes:
                sample_counts[index] += 1
            metric = {"step": step, "loss": total, "effective_tokens": tokens,
                      "update_decisions": len(indexes)}
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
        "sample_counts": sample_counts, "total_decisions": sum(sample_counts)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
