# 本机 PyTorch CUDA 独立验证

2026-10-29，WSL Ubuntu22.04，Python3.11，RTX4060 Laptop 8188MiB，Windows NVIDIA 驱动560.94。

## 来源与独立目录

- wheel：torch2.5.1+cu124，cp311，linux_x86_64，908301937字节。
- 官方索引：<https://download.pytorch.org/whl/cu124/torch/>
- 索引所列下载：<https://download-r2.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-linux_x86_64.whl>
- SHA256：`6b2966ede9affe2fd69e0765691ca723ec870e0c34c7761f4d5b8e318383fdaf`，已核对官方索引并在下载后验证。
- Windows 下载位于 `local-resources/torch-cuda/`，下载脚本为 `local-tools/download-torch-cuda.ps1`。
- 通过 `pip install --no-deps --no-index --target /opt/mirrorforce/distill-gpu-deps <wheel>` 独立安装。
- CPU torch 仍在 `/opt/mirrorforce/distill-testdeps`，冠军 JAX 虚拟环境没有被覆盖。

## 依赖复用及补充

第一次导入失败明确缺少 `libcurand.so.10`。只额外下载安装官方精确依赖：

| 包 | 版本 | SHA256 |
|---|---|---|
| nvidia-curand-cu12 | 10.3.5.147 | a88f583d4e0bb643c49743469964103aa59f7f708d862c3ddb0fc07f851e3b8b |
| nvidia-nvtx-cu12 | 12.4.127 | 781e950d9b9f60d8241ccea575b32f5105a5baf4c2351cab5256a24869f12a1a |

来源由 PyTorch cu124 官方包索引确认，实际下载自 pypi.nvidia.com。脚本为 `local-tools/download-torch-cuda-deps.ps1`，补充依赖与 torch 安装在同一独立 GPU target。

其余 cu12 库复用既有 JAX 环境的 `venv/lib/python3.11/site-packages/nvidia/*/lib`。例如实际 cuBLAS12.9.2.10、cuDNN9.27.0.42、CUDA runtime12.9.79、NCCL2.32.3，与 torch wheel 的固定依赖要求不同。

**这是本机实测的混合依赖组合，不是 PyTorch 官方锁定依赖组合。** 没有安装 Triton3.1.0；已验证 eager 运算与小张量优化，不声称 torch.compile 可用。若后续真实模型出现符号或算子问题，应在独立目录补齐精确版本，不修改冠军 JAX 环境。

## 运行方法

```bash
bash /mnt/f/YGOagent/local-tools/torch-gpu-env.sh \
  /mnt/f/YGOagent/local-tools/check-torch-gpu.py \
  --out /mnt/f/YGOagent/local-resources/torch-cuda/gpu-check.json

# 执行学生模块：
bash /mnt/f/YGOagent/local-tools/torch-gpu-env.sh -m mirrorforce.distill.probe --help
```

包装器只为子进程设置 PYTHONPATH/LD_LIBRARY_PATH，不修改系统配置。可用 MF_ROOT/MF_REPO 覆盖根目录。

## 真实证据

- `torch.__version__=2.5.1+cu124`，built CUDA12.4。
- 实际导入路径 `/opt/mirrorforce/distill-gpu-deps/torch/__init__.py`。
- `torch.cuda.is_available()=true`，BF16支持。
- GPU BF16 的64×64全1矩阵乘法结果求和为262144。
- GPU小张量 eager AdamW执行2步，参数均值从1下降为0.9765625。
- 上述计算约2.49秒，不包括进程启动。

完整证据和实际映射动态库位于 `local-resources/torch-cuda/gpu-check.json`；官方下载清单和补充依赖清单位于同目录，环境版本记录为 `runtime-freeze.txt`。

边界：这里只证明 CUDA 可用、真实 BF16 运算和小张量优化；真实 Qwen 推理、训练、显存和恢复由单独 probe/训练任务验证。
