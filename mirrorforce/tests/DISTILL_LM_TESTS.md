# 标准动作 LM 测试

`test_distill_standard_lm.py` 在临时目录创建随机微型 Qwen2（1层，hidden32），不下载真实 Qwen 权重。覆盖 tied/untied 两种词表权重、标准完整词表 forward、1/2/192 合法动作受约束 generate、真实 LoRA 和新增行梯度、旧行冻结、适配器重载及标准合并模型等价。

测试依赖版本：Python3.11，torch2.5.1（CPU测试包），transformers4.46.3，peft0.13.2，accelerate1.1.1，tokenizers0.20.3，huggingface-hub0.26.2，safetensors0.4.5，psutil6.1.0，pytest8.3.3。GPU正式训练安装对应CUDA版PyTorch，不使用CPU测试包。

```bash
python -m pytest tests/test_distill_standard_lm.py -q
python -m mirrorforce.distill.student --adapter /path/adapter --out /path/merged --device cpu
```

模型 forward 使用 Qwen2 的 `num_logits_to_keep=1` 避免为全部输入位置构造完整词表 logits。保存时仅存新增词表行与 LoRA；加载时重建扩展词表并恢复共享关系。合并导出在副本上执行，不改变正在训练的对象。

注意：微型随机模型测试不等于真实 1.5B 训练或真实引擎对局通过。该测试将 prompt 构造替换为短固定 token 序列以隔离模型层；输入语义/信息边界由独立数据层测试负责。

最终执行证据：2026-10-29，WSL Ubuntu22.04 / Python3.11，标准 LM 与训练 CLI 测试共 11 passed in 17.75s。依赖完全位于 Linux `/opt/mirrorforce/distill-testdeps`，不依赖正在拷贝的 Windows pip target。完整环境 freeze 位于忽略目录 `local-resources/distill-tested-freeze.txt`。运行时设置 `PYTHONPATH=/opt/mirrorforce/distill-testdeps:/mnt/f/YGOagent/mirrorforce`、`OMP_NUM_THREADS=1`。
