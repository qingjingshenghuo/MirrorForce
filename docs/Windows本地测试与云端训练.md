# Windows 本地测试与云端训练路线

更新日期：2026-10-28。

本文件保留最初的路线规划；本机 WSL2、GPU 推理和 MDPro3 人机对战现已部署并验证。当前使用步骤见 [本机人机对战使用说明](本机人机对战使用说明.md)，实际验证结果见 [部署进度](本机人机对战部署进度.md)。下面的“尚未”描述最初规划时的状态，不代表当前部署结果。

## 当前机器和已完成事项

- 系统：Windows；WSL 状态检查明确报告尚未安装。
- 显卡：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存。
- Windows NVIDIA 驱动：560.94。
- 检查时 F 盘约有 66.7 GiB 空闲空间；C 盘约有 28 GiB 空闲空间。
- 已克隆项目及其规则引擎子模块。
- 已下载 v1.0.0 的冠军资源包并解压。
- 已核对整个压缩包 SHA-256、MANIFEST 中 9 项文件摘要及脚本目录 13527 个文件的数量。
- 尚未安装 WSL、编译原生库、安装 Linux Python 依赖或执行推理与训练测试。

本机适合先验证模型推理、小规模对局和极小批次的训练链路。8 GiB 是否能容纳冠军模型的完整训练前向、反向、历史及优化器状态，需要运行后测量；模型参数量或检查点大小不能单独决定显存需求。

## 已下载资源

Windows 目录：`F:\YGOagent\local-resources\mirrorforce-champion-1084`。

对应 WSL 路径：`/mnt/f/YGOagent/local-resources/mirrorforce-champion-1084`。

| 文件或目录 | 内容 |
|---|---|
| `../mirrorforce-champion-1084.tar.gz` | 原始压缩包，约 363 MiB |
| `../mirrorforce-champion-1084.tar.gz.sha256` | 上游摘要文件 |
| `../verification.json` | 本次下载、文件核验与机器状态记录 |
| `checkpoint/dd71087ce1bf827d9b8360c310ae4be02714b50b9535f4959544d24372d28561.ckpt` | 第 1084 轮训练检查点，包含策略、EMA、Critic 和优化器状态 |
| `checkpoint/*.receipt.json` | 检查点来源记录 |
| `cards.cdb`、`code_list.txt` | 卡片数据库和编号映射 |
| `script/` | 配套 Lua 脚本及补丁记录 |
| `frozen_semantics.npz` | 冻结语义表 |
| `card-tables-*.npz`、`announce-tables-*.json` | 卡片表和声明候选表 |
| `decks/SkyStriker.ydk` | 闪刀姬卡组 |

压缩包 SHA-256：

```text
8df616fb436580fba1bf6aa699d20d93aafe67947e6ead78ee8949ccd68325e9
```

解压后的文件总大小约 611.4 MiB，Windows 实际磁盘占用可能因小文件和分配单元而更大。MANIFEST 没有逐一列出 Lua 文件摘要，脚本整体完整性由原始压缩包摘要保证，并另核对文件数量。

## 第一步：建立 WSL2 环境

需要用户在管理员 PowerShell 中安装系统组件，可能要求重启；本次没有自动更改系统组件或重启电脑。

```powershell
wsl --install -d Ubuntu-22.04
```

根据安装提示重启，首次打开 Ubuntu 创建 Linux 用户，再检查：

```powershell
wsl --list --verbose
```

发行版应显示版本 2。若命令提示版本或发行版不可用，先运行 `wsl --list --online` 按实际输出选择，参考下方微软文档。

在 Ubuntu 中执行：

```bash
nvidia-smi
```

需要确认 WSL 内也能识别 GPU。JAX 文档将 WSL2 的 NVIDIA GPU 支持列为实验性，原生 Windows 不提供该 GPU 路线。项目固定 JAX 0.5.3，后续应保持版本，不能直接用最新 JAX 命令升级全部依赖。

后续在 Linux 虚拟磁盘中放置代码、构建目录和脚本，可以避免在 `/mnt/f` 上反复读取上万个小文件。已下载压缩包可作为导入源。C 盘空间有限，配置大型 WSL 虚拟磁盘前应关注其存放位置；较新 WSL 支持安装位置选项，具体以实际版本帮助为准。

## 第二步：解决原生构建依赖

当前发布附件只有冠军数据包和摘要文件，没有 Linux 原生扩展，也没有 `mirrorforce/cxx/duelpool/deps.json` 指定的精确依赖快照。

仍需解决以下构建输入：pybind11、fmt、concurrentqueue、unordered_dense、SQLiteCpp、SQLite 源码、Python 3.11 头文件，以及 Lua 5.3 头文件和库。构建脚本逐目录检查哈希，安装同版本系统包不能保证通过。

后续有两条路径：

1. 取得上游匹配的依赖快照，沿用原构建脚本。
2. 在 WSL 中从公开来源重新准备依赖，建立独立的本地构建清单和来源记录，编译并验证。

第二条属于构建适配，不能将新依赖的哈希冒充原始快照，也不能仅删掉检查便声称严格复现。本次没有虚构不存在的依赖下载地址。

## 第三步：验证发布模型

前置条件：Python 3.11 环境、锁定版本依赖、可加载的 `duel_native` 和配套资源。

顺序：

1. `jax.devices()` 确认可见 GPU。
2. 原生扩展导入成功，运行必要的引擎测试。
3. 按 `docs/usage.md` 启动策略服务，显式使用 `--weights iterate --selection greedy`。
4. 用同一个服务完成至少一组成对本地对局，检查 `summary.json`，确保没有把操作失败当作正常输赢。
5. 记录首次编译时间、推理耗时、峰值显存和错误日志。

GPU 紧张时可先在启动 Python 前设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

该选项用于避免 JAX 启动时大比例预分配显存，不保证模型运行所需的实际峰值能够放下。

## 第四步：本机训练冒烟测试

目标是验证采样、反向传播、优化器更新和保存恢复，不用短时胜率评价训练成果。

建议首轮配置候选：单 GPU、单 Actor 线程、2 个环境、每段 8 步、每轮 32 个决策、2 次优化器更新、每微批 1 个 segment。该组参数满足当前代码的整除条件，但尚未运行验证，仍需完整配置与资源才能执行。

优先对冠军规模网络测试极小批次；若显存不足，则使用单独命名的小网络从头验证训练流程。小网络不能直接加载冠军权重，也不能将其结果视为冠军训练复现。

从头训练时先不启用 Critic 优势来源；如先省略中央 Critic，应在实验配置中明确记录。保持卡组、动作容量和观测定义，避免同时改变太多变量。

重要迁移区别：本机 32 决策的冒烟训练检查点，不保证可直接接到冠军规模训练。当前恢复契约即使声明拓扑变化，仍会校验每轮决策数和优化器步数。正式云端实验可以从发布检查点按上游允许的配置恢复，或以明确配置从头启动；变更训练配方需要单独处理契约和来源记录。

## 第五步：迁移到 AutoDL

先将下列内容迁移到云端：

- 固定提交的源码和引擎子模块。
- 原始资源压缩包、摘要及核验记录。
- 已验证的依赖准备方式、构建脚本和构建记录。
- Python 依赖版本与实际安装环境记录。
- 本地冒烟配置、日志及明确的训练目标。

云端先重新确认 Python、驱动、JAX 和 GPU 识别，再构建或验证原生库，不应假定本机 `.so` 在另一 Linux 发行版上一定兼容。

先用短时任务测量每秒决策数、显存和主机内存，再决定并行环境数、批大小及租用规模。当前尚未实测，不给出租卡时长或复现费用承诺。

## 资料

- [项目发布页](https://github.com/tommy85/MirrorForce/releases/tag/v1.0.0)
- [微软 WSL 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)
- [WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
- [JAX 安装与平台支持](https://docs.jax.dev/en/latest/installation.html)
- [本项目技术文档](技术文档.md)
- [上游运行指南](usage.md)
- [上游训练指南](training.md)
