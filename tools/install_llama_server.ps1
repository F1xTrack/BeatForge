param(
    [string]$Destination = 'N:\llama.cpp\bin-b11985',
    [string]$ArchivePath = 'N:\llama.cpp\llama-b11985-bin-win-cpu-x64.zip'
)

$ErrorActionPreference = 'Stop'
$tag = 'b11985'
$url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cpu-x64.zip"
$expectedSha256 = '1c93f23afd48f8434c4cdc4d9f0a46f47a0d98ea6b3214007cbddc10c34c0c16'

$archiveParent = Split-Path -Parent $ArchivePath
if ($archiveParent -and -not (Test-Path -LiteralPath $archiveParent)) {
    New-Item -ItemType Directory -Path $archiveParent | Out-Null
}
if (-not (Test-Path -LiteralPath $ArchivePath)) {
    Invoke-WebRequest -Uri $url -OutFile $ArchivePath -UseBasicParsing
}

$actualSha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "llama.cpp archive SHA-256 mismatch: expected $expectedSha256, got $actualSha256"
}

if (-not (Test-Path -LiteralPath $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}
Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
$server = Join-Path $Destination 'llama-server.exe'
if (-not (Test-Path -LiteralPath $server)) {
    throw "verified archive did not contain llama-server.exe: $server"
}
Write-Output ([pscustomobject]@{
    status = 'installed'
    release = $tag
    server = (Resolve-Path -LiteralPath $server).Path
    sha256 = $actualSha256
} | ConvertTo-Json -Compress)
