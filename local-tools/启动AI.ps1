# 保留原有中文入口，实际启动逻辑集中在英文文件中。
& (Join-Path $PSScriptRoot 'Start-MirrorForce.ps1') @args