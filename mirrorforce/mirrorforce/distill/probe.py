"""真实学生推理探针：只测资源、概率与生成一致性，不冒充胜率评估。"""
import argparse
import json
import math
from pathlib import Path
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-limit", type=int, default=16384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--points", choices=["first", "three"], default="first")
    a = p.parse_args(argv)
    rows = [json.loads(x) for x in a.samples.read_text().splitlines() if x]
    if not rows:
        raise ValueError("探针没有可用样本")
    if a.out.exists():
        raise ValueError("探针输出已存在，不覆盖证据")
    from .student import Student
    from .data import prompt_details
    start = time.monotonic()
    student = Student(a.base, device=a.device, revision=a.revision, context_limit=a.context_limit)
    loaded = time.monotonic()
    results = []
    for index in ([0] if a.points == "first" else sorted(set([0, len(rows)//2, len(rows)-1]))):
        begun = time.monotonic()
        try:
            _, ids, stats = prompt_details(rows[index], student.tokenizer, a.context_limit)
            result = student.select(rows[index], "greedy")
            generated = student.generate_action(rows[index])
            if generated != result["index"] or not math.isfinite(result["logprob"]):
                raise ValueError("标准生成与forward不一致或概率异常")
            results.append({"sample": index, "input_tokens": len(ids), **stats, **result,
                            "seconds": time.monotonic()-begun, "error": None})
        except Exception as exc:
            results.append({"sample": index, "seconds": time.monotonic()-begun,
                            "error": f"{type(exc).__name__}: {exc}"})
    import resource
    report = {"schema": "mirrorforce_real_student_probe/v1", "actual_llm": True,
        "actual_duel": False, "base": a.base, "revision": a.revision, "device": a.device,
        "context_limit": student.context_limit, "load_seconds": loaded-start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": student.torch.cuda.max_memory_allocated() if a.device.startswith("cuda") else None,
        "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return int(any(row["error"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
