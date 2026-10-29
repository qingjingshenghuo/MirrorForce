param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../local-resources/torch-cuda'))
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($root) | Out-Null
$packages = @(
    @{name='nvidia-curand-cu12'; file='nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl'; sha='a88f583d4e0bb643c49743469964103aa59f7f708d862c3ddb0fc07f851e3b8b'},
    @{name='nvidia-nvtx-cu12'; file='nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl'; sha='781e950d9b9f60d8241ccea575b32f5105a5baf4c2351cab5256a24869f12a1a'}
)
$records = @()
foreach ($package in $packages) {
    $index = "https://download.pytorch.org/whl/cu124/$($package.name)/"
    $url = "https://pypi.nvidia.com/$($package.name)/$($package.file)"
    $body = (Invoke-WebRequest -UseBasicParsing -Uri $index -TimeoutSec 60).Content
    if (-not $body.Contains("$url#sha256=$($package.sha)")) { throw '官方索引来源未核实' }
    $target = Join-Path $root $package.file
    if (-not (Test-Path -LiteralPath $target)) {
        & curl.exe --location --fail --retry 8 --connect-timeout 30 --max-time 1800 --continue-at - --output "$target.partial" $url
        if ($LASTEXITCODE -ne 0) { throw "依赖下载失败：$target" }
        if ((Get-FileHash -LiteralPath "$target.partial" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.sha) { throw '依赖摘要不匹配' }
        Move-Item -LiteralPath "$target.partial" -Destination $target
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.sha) { throw '已有依赖摘要不匹配' }
    $records += @{file=$package.file; sha256=$package.sha; bytes=(Get-Item -LiteralPath $target).Length; source=$url; index=$index}
}
[IO.File]::WriteAllText((Join-Path $root 'supplemental-dependencies.json'),($records | ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
Write-Output '缺失的精确版本 cuRAND/NVTX 已下载并验证。'
