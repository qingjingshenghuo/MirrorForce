# LLM 后训练开发验收记录

更新日期：2026-10-29。此文件记录实际证据，不替代根目录开发规范。

## 当前状态

目标仍在进行中，尚未完成完整交付。用户尚无 AutoDL 实例，当前优先完成本机验证，暂不创建云端收费任务。

| 层级 | 当前证据 | 结论 |
|---|---|---|
| 数据和信息边界 | v2 输入协议、历史与训练数据校验测试 | 基础测试通过，仍需真实轨迹覆盖 |
| 采样 | 模拟会话测试及真实原生引擎L1对局 | L1已通过，真实冠军/Qwen组合仍待验收 |
| 标准 LM | 随机微型 Qwen 的 tied/untied 梯度、标准生成、重载及合并导出测试 | 微型模型已验证，真实1.5B待验证 |
| 训练恢复 | 微型 Qwen 训练入口连续更新和中断恢复预测等价测试 | 真实微型模型训练通过，1.5B待验证 |
| 统计 | 完整发牌对、异常覆盖率、评测身份拒绝测试 | 基础测试通过，真实对照实验待运行 |
| L2～L4 | 尚无完整真实 Qwen 训练和三分支对照结果 | 未完成 |

## 已执行命令

WSL 发行版：MirrorForce-Ubuntu；Python：`/opt/mirrorforce/venv/bin/python`。

```bash
PYTHONPATH=/mnt/f/YGOagent/mirrorforce /opt/mirrorforce/venv/bin/python -m pytest \
  /mnt/f/YGOagent/mirrorforce/tests/test_distill_data_v2.py \
  /mnt/f/YGOagent/mirrorforce/tests/test_distill_sampling.py \
  /mnt/f/YGOagent/mirrorforce/tests/test_distill_stages.py \
  /mnt/f/YGOagent/mirrorforce/tests/test_distill_evaluation_v2.py -q
```

本次输出：`26 passed in 10.38s`。该证据只支持这些测试覆盖的范围，不证明真实冠军/学生对局或模型训练已完成。代码后续修改后必须重新验证。

追加证据：标准 LM、训练入口与旧基础测试合跑 `23 passed in 20.28s`，使用独立 Linux CPU PyTorch 环境。后续已补 NumPy 随机状态、采样频率与长度审计，正在回归。完整依赖记录见忽略目录 `local-resources/distill-tested-freeze.txt`。

精确菜单的独立原生构建已成功：`/opt/mirrorforce/native-build-distill-v3/duel_native.cpython-311-x86_64-linux-gnu.so`，SHA256 `9783da71e0382201e2cec1c5a30dfc9cb3c8bc884fb1341a5b82928961b21720`。原冠军模块未替换。

L1真实引擎测试：`1 passed in 56.21s`。uniform测试教师和随机学生的影子/独立两路径均48回合正常deckout结束，267个学生动作、LP和赢家一致；267条影子路径v2样本全部validate通过，独立路径无伪造教师监督。证据：`local-runs/distill-L1-v3/test_native_distillation_and_i0/L1-evidence.json`，其中明确 `actual_llm=false`，测试样本未写入正式训练数据。

Qwen 官方模型资源下载已启动，固定 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，目标目录 `/opt/mirrorforce/models/Qwen2.5-1.5B-Instruct`；未完成下载与摘要验证前不得视为可用。官方权重期望 SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`，文件大小3087467144字节。

## 已完成的集成调整

- SFT 使用标准 LM 完整词表交叉熵；KD 对合法动作 token 归一化。
- 优化器使用学生提供的参数组区分 LoRA 与词表行；学习率显式传入。
- 评测模式显式 greedy，学生独立推理适配不查询教师神经网络；固定教师仍作为对手。
- 阶段恢复检查配置、命令与产物摘要；中断训练优先从完整检查点恢复，保留失败目录。
- 评测比较要求资源、对手、序列化与选择策略身份一致；公开异常覆盖率和纯胜率。

## 后续验收顺序

### 最新已验证里程碑

追加整合回归：`96 passed, 2 skipped in 134.11s`。跳过项为需单独配置原生资源的L1和可选大轨迹诊断，二者已有独立真实证据。随后将未测量的序列化耗时与教师示范token数改为null（不能把未测当作0），相关数据/采样定向复测28项通过。

- 开发分支已推送：`codex/llm-action-distillation`，首个提交 `081eb1f31810d4ccd87cdb9adc5b3c6121b8146d`；主分支未替换。后续修改仍需单独测试和提交。
- 真实Qwen权重已完整校验，正式文件3087467144字节，SHA256与官方固定revision一致；清单位于忽略目录的 `download-manifest.json`。
- 真实CPU推理探针成功：`local-runs/real-qwen-probe-first.json`。3572输入token、标准forward选择与generate一致；加载49.15秒，两次推理151.02秒，进程峰值RSS9744872KiB（约9.3GiB）。这不是完整对局或训练证据。
- 16384上下文全267条诊断成功：`local-runs/compact-profile-all-16384.json`，614.75秒。必需输入P50/P95/max=7135/8298/8512；裁历史后P50/P95/max=16360/16382/16383；必需字段及最终输入均0溢出。220条需裁旧历史，每条均验证可逆还原和完整菜单保留。
- 恢复身份已绑定源码、实际tokenizer、serializer、数据摘要；编排锁定数据库、原生模块、Lua脚本和资源摘要，显式传递基础模型revision。新增外部资源身份变更测试；恢复相关 `5 passed in 100.88s`。
- 成本台账已补硬中断持久化、孤儿工作进程锁与attempt去重；模拟/真实SIGKILL等测试与预算/阶段测试共15项通过。它是保守墙钟估算，不是平台账单接口。
- 纯语义样本的菜单置换已修复；新增实际训练预算汇总与缺失证据拒绝。audit/report相关5项通过。

真实GPU PyTorch正在独立下载准备中，尚不声明GPU学生已可用；冠军教师新版原生接口服务正在预热。完整L2/L3/L4仍未验收。

### 真实 tokenizer 暴露的未解决问题

实际配置审查另修复了 Qwen 补齐词表兼容：模型151936行、tokenizer151665个token。新动作可以占用原未分配的补齐行，不能误判为碰撞，也不能缩减原模型矩阵。新增 tied/untied 补齐词表测试，包含旧词与未用补齐行不变、保存重载等价；标准 LM 专属测试 `11 passed in 20.62s`。

真实 Qwen tokenizer + 完整L1轨迹已完成：`1 passed in 118.41s`。这里的通过仅指测量任务完成，不代表长度满足模型输入约束。

设计变更已获用户明确同意：上下文改为可配置，正式对局目标16384，先测显存与成本。Student、train、collect、experiment与检查点已贯通配置；评测manifest要求上下文一致。新增上下文越界拒绝与重载保持测试。标准模型、恢复及采样相关回归 `23 passed in 87.75s`。

输入采用可逆语义表格与重复文本引用，完整卡牌效果与菜单保留；全267条真实tokenizer的16384覆盖诊断正在运行，不在结果产生前宣称无超限。新增 `mirrorforce.distill.probe`，用于权重就绪后的真实模型forward/generate一致性、内存和耗时测量；探针明确不代表完整对局或胜率评估。

证据：`local-runs/distill-L1-real-tokenizer-v2/test_native_distillation_and_i0/real-tokenizer-lengths.json`。

- 非强制决策267条。
- 带公开历史的输入长度：P50=34097、P95=56881、P99=58290、最大58518。
- 即使删除全部历史，263条仍无法装入4096 token。

因此当前序列化尚不能支持规定上下文内的完整真实学生对局。不得以L1模拟测试通过推断L2可用。接下来先实现可逆、无损的重复字段/文本压缩并复测；不得删除合法动作或卡牌效果来掩盖超限。若无损压缩仍不足，再明确提出上下文设计调整，不静默扩大验收条件。

全量代码回归已完成：`72 passed, 1 skipped in 107.73s`；跳过的是未设置原生环境变量的L1测试，L1已在独立命令真实验证。测试通过不消除上述真实输入长度问题。

集成追加记录：

- 训练检查点采用 `.incomplete` 临时目录，权重与优化器完整写入后才发布；新增评测资金预留控制。
- 上述训练恢复与预算测试合跑：`7 passed in 77.89s`。
- 全量 distill 测试首轮：70通过、1跳过、1失败；失败为旧统计测试 fixture 未提供新增必需的 seed/pairs 计划字段。已修复该 fixture，定向复测 `2 passed in 4.03s`，未通过放宽生产校验来消除失败。
- 实际 Qwen tokenizer 已下载并核对官方文件身份，正在用真实引擎轨迹测长度；3.09GB 权重通过 Windows 官方下载持续进行。此前 WSL snapshot_download 已明确终止于网络不可达，不再作为活动任务等待。
- 原始逐条裁剪历史的 tokenizer 长度测试在确认性能问题后主动中断（231.52秒），不计为通过。序列化已改为先验证无历史输入，再按历史后缀查找可容纳范围；新增1000条历史的编码次数上限测试。数据专属测试：`20 passed in 5.55s`；新旧数据合跑：`31 passed in 12.10s`。相同完整原生轨迹的真实 tokenizer 复测使用新目录 `local-runs/distill-L1-real-tokenizer-v2`，结果未完成前不填写长度数字。

1. 微型 Qwen：标准 forward/generate 一致、梯度隔离、保存恢复和合并导出。
2. 训练 CLI：连续更新与中断恢复等价、预算边界和检查点内容完整。
3. 真实原生引擎：测试教师与学生策略完成对局，校验 v2 数据与子选择。
4. 真实 1.5B 模型推理与本机显存测量；有资源后运行 L2/L3。
5. 云端正式三分支实验与结果报告，完成 L4 后再进行完整目标完成审计。
