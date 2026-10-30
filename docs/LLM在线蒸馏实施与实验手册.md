# LLM 在线策略蒸馏：实施与实验手册

本扩展基于 MirrorForce 开源引擎及发布冠军教师。新增部分是动作级 LLM、数据采集、SFT/软蒸馏、预算编排与评测。第一阶段只支持既有闪刀姬镜像资源，不声称通用游戏王能力；教师策略 logits 不是 Q。搜索不参与本实验。

## 当前交付与证据边界

代码入口已实现；完整 Qwen 训练、AutoDL 性能、训练适配器和胜率提升必须实际运行后产生，不能用单元测试或随机微型模型替代。初始示范默认最多 50 对对局、10000 条非强制决策；达到样本上限的整局被跳过而不是截断。游戏异常整局不入库，异常率仍进入报告。

## 环境与启动

云端沿用已经确认的 Python 3.11、原生引擎及发布资源。教师环境安装仓库 requirements.txt、local-tools/runtime-constraints.txt 及 jax[cuda12]==0.5.3。学生另建虚拟环境，安装 requirements-distill.txt；游戏采集还需原项目的 numpy、dm-env、gymnasium 等运行依赖，建议同步安装 requirements.txt 与约束，但学生无需加载 JAX GPU 模型。先验证原生引擎与学生 CUDA 可用。

本地新增代码需要同步到云端及运行教师的源码目录，不能继续运行尚未包含 distill_view 的旧服务。Linux 执行以下命令，工作目录为仓库的 mirrorforce/：

```bash
python -m mirrorforce.distill.experiment --config ../configs/distill-autodl.example.json --dry-run
```

先复制配置并修改所有绝对路径、实际小时价格（hourly_yuan 不能为 0）、新输出目录和唯一 socket。第一次将 steps 设为 2、data_pairs/online_pairs/eval_pairs 设为 1，验证全流程。去掉 --dry-run 才执行。

编排器依次启动教师、等待就绪、采样、关闭教师、训练学生，防止训练时与教师争抢显存。所有任务 stdout/stderr、配置和耗时均落盘。教师命令使用参数数组，不经字符串 shell 拼接。运行中的超时会停止当前子进程；失败不会偷偷切回教师替学生行动。

进程总预算默认 450 元，另外预留 50 元用于实例空闲和存储。费用为实际填写单价乘进程耗时，**不是云平台实时账单，也不会自动关闭云实例**。任务结束后需要在 AutoDL 控制台关机。预算上限触发后保留检查点与日志。

## 数据边界和动作协议

教师服务新增 distill_view 操作：session + obs_sha256 输入，返回当前客户端可见卡牌与原生菜单。过期观察拒绝读取；prompt 评分推进教师记忆，commit 提交学生选择并进入下一子选择。多卡选择由原生环境分解，禁止直接拿外部解析器的动作顺序替代原生顺序。

每条样本包含 schema、game_id、deal_seed、deck_id、split、viewer、view、history、teacher_logits、chosen、teacher_identity、obs_sha256、student_version、latency_ms。教师不接触真实对手隐藏状态；导出器对未知卡再次屏蔽身份与数值。当前使用客户端重建盘面、公开连锁、关系和近期本方决策历史，不提供自然语言思维链。

动作标记为 `<|action_000|>` 至 `<|action_191|>`。学生采用标准 Qwen 因果语言模型及其 lm_head；SFT 在完整词表上监督唯一动作 token，软蒸馏在当前合法动作 token 集合内计算教师到学生的 KL。推理只允许生成一个合法动作 token，也可直接取同一 lm_head 的合法动作 logits；两条路径已经做一致性验证。原有词表行冻结，新增动作对应的输入/输出行与注意力 LoRA 一起更新；输入输出是否共享取决于底座，代码同时支持两种情况。Qwen 的未分配补齐行可以用于新 token，无须缩减原权重矩阵。适配器目录包含 LoRA 的 safetensors、action_rows.safetensors、tokenizer 和 student.json，缺少任何部分都不能视为完整学生；恢复优化器还需要 training.pt。

输入采用 semantic_table_json/v3 可逆语义表和共享文本池，完整卡牌效果与动作菜单保留。用户已批准上下文可配置，正式对局使用 16384，训练与采样命令均应明确传入 `--context-limit 16384`，检查点保存该配置。超过上限时只移除最旧的公开历史；盘面、卡牌文本和完整菜单仍超限则报错。不得为了提高成功率静默截掉合法动作。16384 推理通过不等于同长度训练已经通过，须分别测量。

固定种子区间：训练 100000～109999，验证 200000～200999，测试 300000～300999。同一发牌的两次先后手属于同一划分。评测交换双方座位时也交换各自预定牌序；引擎后续随机事件仍可能因动作分叉不同，不能宣称全程随机过程相同。

## 独立命令与恢复

```bash
# 教师示范；把各路径替换为已部署的绝对路径。
python -m mirrorforce.distill.collect --teacher /tmp/mirrorforce-distill.sock \
  --deck /path/assets/decks/SkyStriker.ydk --cards-db /path/assets/cards.cdb \
  --script-root /path/assets --native /path/build/duel_native.cpython-311-x86_64-linux-gnu.so \
  --out /path/runs/demo --seed 100000 --pairs 10 --hourly-yuan 2 --budget-yuan 20

python -m mirrorforce.distill.train --data /path/runs/demo/samples.jsonl \
  --out /path/runs/sft --objective sft --steps 200 --tokens-per-update 16384 --context-limit 16384

# --adapter 只加载权重启动新轮次；--resume 恢复同一任务的优化器和随机状态。
python -m mirrorforce.distill.train --data /path/runs/demo/samples.jsonl \
  --out /path/runs/sft-resumed --objective sft --steps 300 --tokens-per-update 16384 --context-limit 16384 \
  --resume /path/runs/sft/step-000050
```

学生在线采样给 collect 增加 `--student --adapter /path/checkpoint --context-limit 16384`。独立测试还必须加 `--independent-student --selection greedy`，使用 seed 300000、pairs 50。独立模式下学生客户端从引擎重建本方可见观察，由 Qwen 选择所有非强制动作；单一合法动作按协议直接执行。教师仅作为对手运行，学生端 teacher_queries 必须为零。在线蒸馏模式的动作同样由学生选择，教师查询只产生监督并同步记忆，不允许接管。所有输出目录必须是新目录，已有实验不会被覆盖。training.pt 使用 PyTorch pickle 保存优化器，只恢复自己生成且可信的文件。

三个对照从同一 SFT 检查点开始：继续 SFT、固定数据软蒸馏、学生在线数据软蒸馏。三条分支均执行三轮，均每轮重新初始化优化器；在线轮次必须使用上一轮学生重新采样。KL 方向固定为教师到学生，温度 1，动作集合以合法菜单为准。

为严格匹配预算，编排器默认每个更新恰好使用 16384 个有效输入 token，通过完整样本长度组批实现；不可达则失败。此组批会改变不同长度样本的采样频率，三个分支采用相同规则并记录每步样本数。单独 train 命令不指定 tokens-per-update 时使用固定梯度累积次数，不应据此声称严格等 token。

200 万 token 上限下，16384 配额最多进行 122 次更新，而非 200；steps 是另一上限。以 result.json 和 metrics.jsonl 的实际更新次数、token 数为准。三组必须同时匹配实际累计更新数和 token 数，并另报采样费用。中断恢复要求数据摘要、目标、学习率、种子及组批配置一致。

CUDA训练启用严格确定性算法并关闭TF32，在初始化CUDA前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（训练入口会在新进程自动设置；若CUDA已初始化而配置缺失则拒绝）。恢复身份还校验PyTorch/CUDA版本、GPU和计算设置。源码与HEAD也必须保持一致，验证恢复期间不要修改源码或提交新版本；旧源码检查点可以作为 `--adapter` 开启新任务，不能伪装为同一优化器任务恢复。每步metrics另含更新墙钟、输入token吞吐和CUDA峰值分配/预留显存，这些不包含模型加载、保存检查点或独立教师进程。

## 评测与报告

```bash
python -m mirrorforce.distill.evaluate /path/online/games.jsonl --baseline /path/offline/games.jsonl
python -m mirrorforce.distill.audit --data /path/validation/samples.jsonl --adapter /path/checkpoint
```

evaluate 报告胜平负、失败数、完整发牌对数、平均对局得分（平局计 0.5）、按发牌对 bootstrap 的 95% 区间、延迟及教师查询次数。区分“得分率”和纯胜率。含失败的一对不进入置信区间，但失败数始终公开；比较仅使用两方案共同完整发牌对，公开排除数量，不把失败视为平局。少于两对不输出区间。

audit 报告持出样本 KL、教师动作一致率和菜单重排映射后一致率。合法动作率是受约束解码的工程属性，不代表学习效果。没有显著差异也应保留结果和错误分析。最后报告应附资源版本、基础模型 commit、训练配置、全部种子、实际费用、运行环境 freeze，以及上游贡献与本项目贡献的边界。

## 审查修复后的运行配置

`initial_sft` 与 `branch_training` 可分别配置 steps、token_budget、tokens_per_update、save_every、keep_last、min_observed_coverage；三个后续分支共用 branch_training，不能单独放宽某个分支。未填写时兼容原顶层配置。示例已用131072作为候选配额，真实数据仍必须先审计；200万token最多15次该配额更新，只用于试跑。

训练前会生成输出目录同级的 `<任务名>-coverage.json`，记录逐条长度、精确配额可参与性与排除元数据。配额可达但永久排除任何样本时同样拒绝训练。不能直接假定默认16384配额能覆盖16K输入；必须先用真实数据审计共同配额。sampling-audit另报实际访问状态数、覆盖率和重复次数。

keep_last=0保留全部检查点；正数仅在当前新训练输出目录内保留最新N份完整检查点，发生在该阶段产物账本发布前。最终检查点始终保留；外部引用或最佳验证检查点目前不支持在任务内自动注册保护，若需要保留任意历史最佳检查点，应设置keep_last=0并核算磁盘，不能误用滚动保留。

新采样结果的latency_ms只汇总实际调用学生模型的动作，额外记录强制菜单数量、模型调用数量和decision_trace。历史结果保持原统计口径，不与新版延迟直接混合比较。

## 后续多卡组

第一阶段报告完成后，再评估卡牌资源覆盖和教师训练成本。依次完成多卡组环境验证、混合对局训练、逐卡组基线与遗忘评测、持出配对评测，教师质量达标后才生成新的蒸馏数据。当前数据协议保留 deck_id，但不代表已获得跨卡组能力。
