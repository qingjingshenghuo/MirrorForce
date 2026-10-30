# 本机轻量验收证据

本目录保存原始JSON结果的副本，不包含权重或对局训练数据。

- deterministic-resume-verification.json：真实1.5B两步恢复检查，同设备环境权重与固定输入动作分布差异0。
- independent-games-summary.json：真实学生1对2局，0异常、0学生教师查询、2负；不是能力提升证明。
- training-profile.json：4次短输入更新、1次长输入更新。后续长更新主动停止，不能视为完整长上下文稳定性验证。
- cpu-tested-freeze.txt：本机CPU测试环境实际包清单，不能直接作为GPU学生或JAX教师的统一安装锁。

对应源码基线和局限见上级目录的LLM开发验收记录。GPU环境的CUDA库组合、下载摘要与使用方式见本机PyTorch-CUDA独立验证文档；云端仍须重新验证驱动、学生和教师环境。恢复文件仅限可信来源。

另附qwen-download-manifest、teacher-resource-identity、gpu-runtime-check与GPU学生/教师分别导出的freeze，便于核对资源版本和原生摘要。freeze含本机wheel路径，迁移前需要按对应版本和SHA重新取得wheel，不能直接当成跨机器安装锁；GPU学生复用了部分教师CUDA动态库，属于已实测的混合组合，不能假定所有驱动兼容。本目录不含大资源文件，仍需下载和原生构建才能重建环境。
