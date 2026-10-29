param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../local-resources/Qwen2.5-1.5B-Instruct')
)
$ErrorActionPreference = 'Stop'
$revision = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
$modelSha = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
$modelBytes = 3087467144
$repo = 'Qwen/Qwen2.5-1.5B-Instruct'
$api = "https://huggingface.co/api/models/$repo/revision/${revision}?blobs=true"
$metadata = Invoke-RestMethod -Uri $api -TimeoutSec 60
if ($metadata.sha -ne $revision) { throw '模型 revision 校验失败' }
$out = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($out) | Out-Null
$names = @('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','LICENSE','README.md','model.safetensors')
$records = @()
function Get-GitBlobSha([string]$Path) {
    $content = [IO.File]::ReadAllBytes($Path)
    $prefix = [Text.Encoding]::UTF8.GetBytes("blob $($content.Length)`0")
    $algorithm = [Security.Cryptography.SHA1]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($prefix + $content)).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
foreach ($name in $names) {
    $entry = @($metadata.siblings | Where-Object rfilename -eq $name)[0]
    if (-not $entry) { throw "官方 revision 缺少文件 $name" }
    $target = Join-Path $out $name
    $partial = "$target.partial"
    $verified = $false
    if (Test-Path -LiteralPath $target) {
        if ($name -eq 'model.safetensors') {
            $verified = (Get-Item -LiteralPath $target).Length -eq $modelBytes -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $modelSha
        } else {
            $verified = (Get-Item -LiteralPath $target).Length -eq $entry.size -and (Get-GitBlobSha $target) -eq $entry.blobId
        }
        if (-not $verified) { throw "已有文件校验失败，保留现场：$target" }
    }
    $url = "https://huggingface.co/$repo/resolve/$revision/${name}?download=true"
    if (-not $verified) {
        Write-Output "下载 $name，支持 partial 断点恢复"
        & curl.exe --location --fail --retry 8 --retry-delay 3 --connect-timeout 30 --max-time 7200 --speed-limit 1024 --speed-time 120 --continue-at - --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw "下载失败，保留断点：$partial" }
        if ($name -eq 'model.safetensors') {
            if ((Get-Item -LiteralPath $partial).Length -ne $modelBytes -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $modelSha) {
                throw '权重长度或 SHA256 与官方固定资源不符'
            }
        } else {
            $actualBlob = Get-GitBlobSha $partial
            if ($actualBlob -ne $entry.blobId) { throw "Git blob SHA1 不符：$name" }
        }
        Move-Item -LiteralPath $partial -Destination $target
    }
    $records += [ordered]@{file=$name; bytes=(Get-Item -LiteralPath $target).Length; sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant(); source=$url; git_blob=$entry.blobId}
}
$manifest = [ordered]@{repo=$repo; revision=$revision; downloaded_at=[DateTime]::UtcNow.ToString('o'); files=$records}
[IO.File]::WriteAllText((Join-Path $out 'download-manifest.json'), ($manifest | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
Write-Output "完成：$out，固定 revision 与权重 SHA256 已验证。"
