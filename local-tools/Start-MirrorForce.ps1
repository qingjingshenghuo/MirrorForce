param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$wslExe = 'C:\Program Files\WSL\wsl.exe'
if (-not (Test-Path -LiteralPath $wslExe)) { $wslExe = 'wsl.exe' }

if ($CheckOnly) {
    Get-Command $wslExe -ErrorAction Stop | Out-Null
    foreach ($name in @('start-service.py', 'serve-policy.sh', 'play-room.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name))) {
            throw ('缺少启动文件：' + $name)
        }
    }
    Write-Host 'STARTUP_CHECK_OK：CMD 和 PowerShell 启动入口检查通过。'
    return
}

Write-Host '正在检查 MirrorForce 推理服务，首次启动需要编译预热。'
& $wslExe -d MirrorForce-Ubuntu -u root -- env PYTHONPATH=/opt/mirrorforce/repo/mirrorforce /opt/mirrorforce/venv/bin/python -u /mnt/f/YGOagent/local-tools/start-service.py
if ($LASTEXITCODE -ne 0) { throw '策略服务启动失败，请查看提示的日志' }
Write-Host '请先在 MDPro3 中创建本地主机，选择 MirrorForce-SkyStriker 卡组，并设置不检查卡组。'
& $wslExe -d MirrorForce-Ubuntu -u root -- env PYTHONPATH=/opt/mirrorforce/repo/mirrorforce /opt/mirrorforce/venv/bin/python -u /mnt/f/YGOagent/local-tools/play-room.py
if ($LASTEXITCODE -ne 0) { throw '房间连接或对局失败，请查看上述错误' }
