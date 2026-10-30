"""整局采样及固定发牌评测；失败局不进入训练数据。"""
import argparse
import json
import math
import tempfile
from pathlib import Path
from .data import CardTexts, digest, split_for
from .policy import DistillPolicy


def validate_cost_mode(local, hourly_yuan, budget_yuan):
    if not all(math.isfinite(x) for x in (hourly_yuan, budget_yuan)):
        raise ValueError("单价与预算必须是有限数值")
    if (local and (hourly_yuan != 0 or budget_yuan != 0)) or (not local and min(hourly_yuan, budget_yuan) <= 0):
        raise ValueError("本机模式必须明确使用0单价和0云预算；云端模式必须为正值")


def content_identity(path):
    """检查点身份只依赖内容，不能用路径冒充版本。"""
    import hashlib
    if path is None:
        return "untrained_base"
    root = Path(path)
    if not root.is_dir():
        raise ValueError("适配器目录不存在")
    h = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(file.relative_to(root).as_posix().encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--cards-db", type=Path, required=True)
    p.add_argument("--script-root", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--adapter")
    p.add_argument("--student", action="store_true", help="使用学生，省略则采集教师示范")
    p.add_argument("--independent-student", action="store_true", help="学生不查询教师评分，仅对手使用教师服务")
    p.add_argument("--windbot-exe", type=Path, help="固定WindBot可执行文件；仅独立学生评测")
    p.add_argument("--selection", choices=["sample", "greedy"], default="sample")
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--context-limit", type=int, help="学生上下文上限，包含唯一动作输出")
    p.add_argument('--prompt-style', choices=['compact', 'named_menu'])
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--game-seconds", type=float, default=1800)
    p.add_argument("--hourly-yuan", type=float, required=True)
    p.add_argument("--budget-yuan", type=float, default=50)
    p.add_argument("--local", action="store_true", help="本机免费验证模式，单价和云端预算必须均为0")
    a = p.parse_args(argv)
    if a.windbot_exe and not a.independent_student:
        p.error('WindBot仅用于独立学生评测')
    if a.adapter and not a.student or a.independent_student and not a.student or min(a.pairs, a.max_samples, a.game_seconds, a.temperature) <= 0:
        p.error("无效采样配置")
    try:
        validate_cost_mode(a.local, a.hourly_yuan, a.budget_yuan)
    except ValueError as exc:
        p.error(str(exc))
    split = split_for(a.seed)
    if split_for(a.seed + a.pairs - 1) != split:
        p.error("种子区间跨越数据划分")
    if split == "test" and a.student and a.selection != "greedy":
        p.error("固定测试必须使用 greedy")
    from tools.mf_runtime_local_match import service_identity, play_game, sha256
    from tools.mf_runtime_client_parity import host_core
    from mirrorforce.netduel.agent_policy import RemotePolicy
    from mirrorforce.netduel.cards import CardPool
    from mirrorforce.worldmodel.engine import load_ydk
    import time
    started = time.monotonic()
    identity = service_identity(a.teacher, 120)
    if not identity.get("checkpoint_sha256"):
        raise ValueError("正式采样必须使用真实检查点教师")
    for key, path in (("serving_native_sha256", a.native), ("cards_db_sha256", a.cards_db)):
        if identity.get(key) != sha256(path):
            raise ValueError("教师和采集器的原生模块或卡牌数据库不一致")
    a.out.mkdir(parents=True, exist_ok=False)
    if a.student:
        from .student import Student
        chooser = Student(a.base, a.adapter, a.device, context_limit=a.context_limit, prompt_style=a.prompt_style)
    else:
        chooser = None
    texts, cards, recipe = CardTexts(a.cards_db), CardPool(a.cards_db), load_ydk(a.deck)
    from .data import SERIALIZER_VERSION
    from .stages import source_identity
    source = source_identity()
    source_commit = source["source_commit"]
    student_sha = (content_identity(a.adapter) if a.adapter else
                   digest({"base": chooser.base, "revision": chooser.revision})) if a.student else "teacher"
    tokenizer_sha = digest(chooser.tokenizer.get_vocab()) if chooser is not None else "not_applicable"
    local_native = None
    if a.independent_student:
        import os
        from types import SimpleNamespace
        os.environ["MF_DUEL_NATIVE"] = str(a.native.resolve())
        from tools.mf_runtime_policy_service import load_native
        announce = list(a.cards_db.parent.glob("announce-tables-*.json"))
        if len(announce) != 1:
            raise ValueError("独立学生要求唯一 announce-tables 文件")
        local_native, _ = load_native(SimpleNamespace(script_root=a.script_root, cards_db=a.cards_db,
            code_list=a.cards_db.parent / "code_list.txt", backend="uniform", announce_tables=announce[0],
            announce_cap=identity["client_config"].get("announce_cap", 192)))
    manifest = {**vars(a), "teacher_identity": identity, "deck_sha256": sha256(a.deck),
                "cards_db_sha256": sha256(a.cards_db), "native_sha256": sha256(a.native),
               "split": split, "pairing": "role_deck_orders_preserved/v1", **source,
               "serializer_version": SERIALIZER_VERSION, "selection": a.selection if a.student else "greedy",
               "student_checkpoint_sha": student_sha, "tokenizer_hash": tokenizer_sha}
    manifest["context_limit"] = chooser.context_limit if chooser is not None else a.context_limit
    manifest['prompt_style'] = chooser.prompt_style if chooser is not None else 'not_applicable'
    manifest['prompt_version'] = chooser.prompt_version if chooser is not None else 'not_applicable'
    manifest['opponent_identity'] = ({'kind': 'windbot', 'exe_sha256': sha256(a.windbot_exe),
        'executor': 'SkyStriker'} if a.windbot_exe else {'kind': 'teacher', 'identity': identity})
    (a.out / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2), encoding="utf-8")
    count, results, stop = 0, [], "completed"
    with tempfile.TemporaryDirectory(dir=a.out) as scratch, \
            (a.out / "samples.jsonl").open("w", encoding="utf-8") as samples, \
            (a.out / "failed-samples.jsonl").open("w", encoding="utf-8") as failed_samples, \
            (a.out / "games.jsonl").open("w", encoding="utf-8") as games:
        core = host_core(a.native, a.cards_db, a.script_root, scratch)
        for pair in range(a.pairs):
            # 给一对完整对局预留最坏运行时间；不足则不开始下一对。
            if (time.monotonic() - started + 2 * a.game_seconds) / 3600 * a.hourly_yuan > a.budget_yuan:
                stop = "budget"
                break
            for first in (True, False):
                seed = a.seed + pair
                game_id = f"{manifest['deck_sha256'][:12]}-{seed}-{int(first)}"
                held, opponents = [], []

                def factory(role, seat):
                    args = ("unix:" + str(a.teacher), identity)
                    if role == "b":
                        if a.windbot_exe:
                            from .windbot import WindBotPolicy
                            return WindBotPolicy(identity, a.windbot_exe, a.deck, a.cards_db, seed=seed)
                        opponent = RemotePolicy(*args, seed=seed, timeout=120)
                        opponents.append(opponent)
                        return opponent
                    metadata = {"run_id": a.out.name, "game_id": game_id, "deal_seed": seed,
                        "first_player": "self" if first else "opponent", "deck_id": manifest["deck_sha256"],
                        "opponent_deck_id": manifest["deck_sha256"], "student_checkpoint_sha": student_sha,
                        "source_commit": source_commit, "engine_hash": manifest["native_sha256"],
                        "source_tree_sha256": source["source_tree_sha256"],
                        "cards_db_hash": manifest["cards_db_sha256"], "tokenizer_hash": tokenizer_sha}
                    options = dict(seed=seed, chooser=chooser, texts=texts, metadata=metadata,
                                   selection=a.selection, temperature=a.temperature)
                    if a.independent_student:
                        from .policy import StudentPolicy
                        policy = StudentPolicy(local_native, identity, **options)
                    else:
                        policy = DistillPolicy(*args, timeout=120, **options)
                    held.append(policy)
                    return policy

                result = play_game(core, cards, recipe, seed, first, {"a": a.teacher, "b": a.teacher},
                    {"a": identity, "b": identity}, max_seconds=a.game_seconds, policy_factory=factory,
                    preserve_role_deals=True)
                result.update(game_id=game_id,
                              teacher_queries=0 if a.independent_student else sum(len(p.decisions) for p in held),
                              opponent_teacher_queries=sum(len(p.decisions) for p in opponents),
                              model_decisions=sum(bool(r.get("model_called")) for p in held for r in p.decisions),
                              forced_menu_decisions=sum(bool(r.get("forced")) for p in held for r in p.decisions),
                              decision_trace=[r for p in held for r in p.decisions],
                              latency_ms=[r["student_ms"] for p in held for r in p.decisions if r.get("model_called")])
                rows = [r for policy in held for r in policy.samples]
                if not result["error"] and count + len(rows) <= a.max_samples:
                    for row in rows:
                        row.update(completed=True, training_eligible=True,
                                   episode_result=str(a.out / "games.jsonl") + "#" + game_id)
                        from .data import validate
                        validate(row)
                    # 全局校验通过后才发布本局，不能留下半局的已验收记录。
                    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                    episode_dir = a.out / "episodes"
                    episode_dir.mkdir(exist_ok=True)
                    temporary = episode_dir / (game_id + ".tmp")
                    temporary.write_text(payload, encoding="utf-8")
                    temporary.replace(episode_dir / (game_id + ".jsonl"))
                    samples.write(payload)
                    count += len(rows)
                    result["samples_saved"] = len(rows)
                else:
                    result["samples_saved"] = 0
                    for row in rows:
                        row.update(completed=not bool(result["error"]), training_eligible=False,
                                   exclusion_reason=result["error"] or "sample_limit")
                        failed_samples.write(json.dumps(row, ensure_ascii=False) + "\n")
                    failed_samples.flush()
                results.append(result)
                games.write(json.dumps(result, ensure_ascii=False) + "\n")
                games.flush()
                samples.flush()
                print(json.dumps({k: v for k, v in result.items() if k not in ("latency_ms", "decision_trace")}), flush=True)
            if split != "test" and count >= a.max_samples:
                stop = "sample_limit"
                break
    from .evaluate import summarize
    report = summarize(results, planned_games=a.pairs * 2)
    report.update(samples=count, stop=stop, estimated_yuan=(time.monotonic() - started) / 3600 * a.hourly_yuan,
                  cost_note="本机验证，不计云端费用" if a.local else "仅本次进程耗时估算，不含实例空闲和存储，最终以账单为准")
    (a.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
