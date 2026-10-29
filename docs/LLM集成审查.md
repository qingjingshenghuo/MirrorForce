# LLM 集成只读审查

审查时间：2026-10-29。范围：阶段恢复与费用、训练恢复、菜单重排、v3 文本协议兼容、采样信息边界。本次只写本报告，不修改被审查模块。

注意：`SCHEMA=mirrorforce_action_distill/v2` 与 `SERIALIZER_VERSION=semantic_table_json/v3` 是两个不同版本标识；当前并未把样本 schema 升级到 v3。

## 已确认的问题

### P1：完成阶段的复用没有绑定当前源码和外部输入内容

位置：`mirrorforce/mirrorforce/distill/stages.py:34`、`:50`、`:61`。

StageLedger 只保存配置的 JSON 摘要、命令字符串以及输出产物哈希。`source_identity()` 存在，但没有进入台账的复用身份。命令中的输入文件只是路径，原地替换代码、教师启动脚本、基础模型或 Lua 资源后，旧阶段仍可被判为 reusable。

已执行最小复现：在临时目录建立 input.py 和 result，标记 train 阶段成功后改写 input.py，再用相同配置和命令恢复。真实输出：

```text
{'changed_source_reused': True}
```

输出产物被改写会拒绝复用，这是已有正确行为；但它不能代替检查输入与源码。

建议：台账锁定源码树摘要，以及每阶段的输入文件/目录内容摘要。数据、起始适配器、教师 checkpoint/receipt、数据库、脚本树、原生模块、tokenizer 和 serializer 身份纳入对应阶段。输出日志不作为唯一成功证明。添加“同路径修改输入/源码而输出不变”的回归测试。

### P1：训练恢复记录源码身份，但没有校验它

位置：`mirrorforce/mirrorforce/distill/train.py:89`～`:109`。

training.pt.identity 包含数据文件 SHA、学习率、种子、动作 token IDs、上下文长度、基础模型及 revision，这些已有校验有效。但 source_identity 仅写入 run.json，不进入 training.pt.identity；serializer_version 和 tokenizer 内容摘要也没有进入恢复校验。

触发：保持数据文件与命令不变，改动损失实现或 prompt serializer，再从原训练 checkpoint 恢复。身份比较仍可能通过，而新旧更新规则或输入 token 序列已经不同。仅保留固定的 loss_version 字符串不能检测忘记手工升级版本的代码改动。

建议：保存并比较实际源码树摘要、serializer_version、tokenizer 内容摘要、关键优化器配置。若确需迁移，应显式作为新实验加载 adapter，不沿用 optimizer 状态冒充原任务恢复。

### P1：硬中断的费用没有可靠记入累计账本

位置：`mirrorforce/mirrorforce/distill/experiment.py:43`、`:81`、`:119`～`:129`。

累计 spent 只来自已写完 costs.json 的条目；费用在 run 的 finally 中结算。阶段开始时的 stages.json 只有 status=running，没有可跨进程恢复的开始时间或预扣费用。若父进程被 SIGKILL、机器断开或系统重启，finally 不执行，已经运行很久的阶段在下次 --resume 时可能不计费用，重新获得接近原预算的运行时间。

正常 Python 异常会执行 finally，此问题特指不可捕获的硬中断，不能用普通异常测试证明不存在。

建议：原子记录每个 attempt 的 wall-clock 起始时间与单价，定期写入累计费用；恢复时对 running attempt 做保守费用结算，或要求明确录入云账单时间。不能宣称脚本预算等于平台账单硬上限。

### P1：编排入口尚未固定基础模型 revision

位置：`mirrorforce/mirrorforce/distill/experiment.py:147`～`:155`；`collect.py:40`～`:73`。

train CLI 支持 --base/--revision，但 experiment 生成训练命令时没有传递这两个参数。collect 支持 --base，却没有 --revision。示例配置也没有基础模型目录与固定 revision。初始 SFT 因而默认解析远端模型名；从本地固定资源开始以及完全离线运行未在总编排中贯通。

当前已经下载的固定 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。不能仅因为后续 adapter 保存了 revision，就声称首次运行已经由配置锁定。

建议：配置增加 base/revision，并贯通首次训练和无适配器学生采样。记录资源 manifest 的内容摘要。已有 adapter 阶段应校验其声明的基础资源与全局实验配置一致。

### P2：菜单重排 audit 不支持验证器允许的纯语义样本

位置：`mirrorforce/mirrorforce/distill/audit.py:34`；`data.py::validate/model_input`。

v2 数据验证器允许没有原始审计副本 view、但具有完整四部分语义输入的样本；model_input 也优先使用语义字段。audit 却无条件访问 changed['view']['actions']，因此这类合法输入执行菜单重排时会 KeyError。

已使用当前测试样本构造函数，在内存移除 view 后调用真实 validate，得到：

```text
{'semantic_only_accepted': True}
```

此外，audit 重排了 view.actions、legal_actions 与 teacher_logits，却没有更新 action_mapping、chosen/chosen_token 和 menu_sha256。当前 Student 直接读取 model_input，所以这不一定导致当前一致率数值错误；但重排对象不再是有效协议样本，不适合直接交给未来严格校验入口。

建议：提供单一的重排辅助函数，明确区分“纯模型输入重排”与“完整样本重排”。前者只重排语义菜单并返回逆映射；后者更新所有相依字段后通过 validate。测试需覆盖纯语义样本、带审计副本样本、同名动作和 n=192。

### P2：没有自动产出最终等预算比较证据

位置：`experiment.py` 最后的 variants 输出；`train.py::result.json/sampling-audit.json`。

三个分支使用相同参数是良好基础，训练也输出实际 token 与更新数，但总编排尚未汇总并断言三分支实际累计更新数、有效输入 token 数是否相同。若某阶段恢复步数不同、配置预算被改动或运行提前结束，仅各自 result.json 存在不能证明对照公平。

建议：最终读取每分支三轮 result.json、sampling-audit.json 和阶段身份，汇总实际更新/输入 token/输出决策及成本，加入显式 comparable 标记；不满足约束时阻止生成“等预算提升”结论。

## 已核对的正确行为与边界

- train 恢复已校验实际数据文件 SHA；同内容不同路径不会因路径本身被拒绝，数据原地改写会被拒绝。
- checkpoint 先写 `.incomplete`，权重和 optimizer 写完后再 rename，完成阶段的输出文件也有内容摘要校验。
- 当前 audit 已使用完整词表的 `student.ids[:n]`，没有把词表前 n 项误当动作 logits。
- v3 serializer 由 model_input/compact_input 处理，菜单重排仍能进入同一文本协议；“必须存在 view”的问题与压缩编码本身不同。
- collect 校验教师与本地 native、cards_db 的哈希，学生模型版本取适配器目录内容摘要，避免仅用路径作为版本。
- 采样器仅从教师客户端观察器/独立 AgentClientDuel 导出状态，公开事件解析采用白名单，监督字段不进入 model_input；本次没有发现把真实对手隐藏手牌直接写入 prompt 的路径。
- 未发现直接泄漏不等于完成了对所有消息类型的信息流形式验证；仍以隐藏真相不变性、实际消息路由和原生对局测试为证据。
- 失败局不进入训练 samples，failed-samples 单独保留；测试分支使用独立学生适配器，不用教师 logits 来替学生做选择。
- CLI summary 已按 pairs*2 计算计划完成率，未观察对局与失败对局不会被当作正常平局。

## 额外的可复现性缺口

1. collect manifest 包含 script_root 路径，但尚未包含 Lua 脚本树哈希。仅 native/db 哈希相同不能证明规则脚本相同；正式对比应加入脚本、code_list、announce/card tables 身份。
2. source_identity 的源码后缀白名单不含 C++ `.cpp/.hpp` 和 PowerShell `.ps1`，即使将此摘要加入恢复校验，仍应修订白名单或直接显式列出受信源树。
3. 教师示范的 input_tokens 仍为默认 0，因为只在 chooser 非空时计算；这应标为“未测量”，不可在数据长度分析时解释成零 token 输入。
4. audit 未验证 --limit 为正；负数会执行 Python 切片而不是拒绝无效配置。

## 建议修复顺序

1. 先锁定 source/input/model revision 并修复恢复费用结算，再开始正式收费训练。
2. 统一菜单重排辅助函数与协议校验，补纯语义样本测试。
3. 增加全分支预算/身份汇总与脚本资源指纹，再出正式对照报告。

本报告记录的是审查时源码；后续修复后需逐项补上回归测试与运行证据。不能把“已提出修复建议”写成“问题已解决”。
# 修复状态补充

本审查正文是发现问题时的源码快照，不代表最新实现仍具有全部问题。后续已修复并测试：源码/资源恢复绑定、训练恢复serializer与tokenizer身份、固定模型revision、硬中断成本台账与存活锁、纯语义菜单置换、实际等预算汇总、Lua与源码扩展名指纹，以及未测字段使用null。最新验收证据请读取同目录《LLM开发验收记录》。真实1.5B训练及完整对照实验仍未完成。
