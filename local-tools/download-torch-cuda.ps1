param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../local-resources/torch-cuda'))
$ErrorActionPreference = 'Stop'
$indexUrl = 'https://download.pytorch.org/whl/cu124/torch/'
$url = 'https://download-r2.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-linux_x86_64.whl'
$sha = '6b2966ede9affe2fd69e0765691ca723ec870e0c34c7761f4d5b8e318383fdaf'
$bytes = 908301937
$filename = 'torch-2.5.1+cu124-cp311-cp311-linux_x86_64.whl'
$index = (Invoke-WebRequest -UseBasicParsing -Uri $indexUrl -TimeoutSec 60).Content
if (-not $index.Contains("$url#sha256=$sha")) { throw '官方索引未确认预期 URL 和 SHA256' }
$root = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($root) | Out-Null
$file = Join-Path $root $filename
if (-not (Test-Path -LiteralPath $file)) {
    $partial = "$file.partial"
    & curl.exe --location --fail --retry 8 --retry-delay 3 --connect-timeout 30 --max-time 7200 --continue-at - --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "下载失败，保留断点：$partial" }
    if ((Get-Item -LiteralPath $partial).Length -ne $bytes -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw 'CUDA wheel 长度或摘要不匹配' }
    Move-Item -LiteralPath $partial -Destination $file
}
if ((Get-Item -LiteralPath $file).Length -ne $bytes -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw '已存在 wheel 校验失败' }
$manifest = [ordered]@{package='torch';version='2.5.1+cu124';python='cp311';platform='linux_x86_64';index=$indexUrl;source=$url;bytes=$bytes;sha256=$sha;downloaded_at=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $root 'download-manifest.json'),($manifest | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
Write-Output 'PyTorch CUDA wheel 已下载并校验。'
