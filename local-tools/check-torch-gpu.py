"""独立 CUDA PyTorch 与复用 JAX CUDA 库的真实 BF16 计算验证。"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if torch.__version__ != "2.5.1+cu124" or not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch 版本或 GPU 可用性不符合预期")
    start = time.perf_counter()
    with torch.no_grad():
        a = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
        result = a @ a
        torch.cuda.synchronize()
        total = float(result.float().sum().item())
    if result.device.type != "cuda" or total != 262144.:
        raise RuntimeError("真实 GPU BF16 矩阵结果不正确")
    # 单独验证 eager autograd/optimizer 可运行；不冒充真实 Qwen 的2步训练。
    parameter = torch.nn.Parameter(torch.ones(16, 16, device="cuda", dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        (parameter @ parameter).float().mean().backward()
        optimizer.step()
    torch.cuda.synchronize()
    parameter_mean = float(parameter.float().mean().item())
    if not parameter_mean < 1.:
        raise RuntimeError("GPU eager optimizer 未实际改变参数")
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "/nvidia/" in line and ".so" in line})
    dist = importlib.metadata.distribution("torch")
    nvidia = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
              if d.metadata.get("Name", "").startswith("nvidia-")}
    report = {"torch": torch.__version__, "torch_file": torch.__file__, "built_cuda": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(), "bf16_matmul_sum": total,
        "eager_adamw_steps": 2, "parameter_mean_after": parameter_mean, "seconds": time.perf_counter() - start,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"), "loaded_nvidia_libraries": mapped,
        "installed_nvidia_packages": nvidia, "wheel_declared_requirements": dist.requires,
        "dependency_note": "no-deps独立CUDA torch，复用已有JAX环境的较新cu12库；不是官方锁定依赖组合。",
        "claim_boundary": "证明GPU真实BF16计算及小张量eager优化；不证明Qwen全模型训练通过。"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
