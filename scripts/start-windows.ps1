[CmdletBinding()]
param([switch]$Background)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$main = Join-Path $root 'main.py'
if (-not (Test-Path $python)) { throw 'Missing .venv. Run scripts\setup-windows.ps1 first.' }
if (-not (Test-Path (Join-Path $root 'static\index.html'))) { throw 'Missing frontend assets. Run scripts\setup-windows.ps1 first.' }
$executables = @($python, (Join-Path $root '.runtime\python\python.exe'))
$mainPattern = [regex]::Escape($main) + '"?\s+--serve-only(?:\s|$)'
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -in $executables -and $_.CommandLine -match $mainPattern
}
if ($existing) { Write-Host 'This deployment is already running.'; return }

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = "$(Join-Path $root '.runtime\node');$env:PATH"
$env:XDG_CACHE_HOME = Join-Path $root '.cache'
$env:HF_HOME = Join-Path $root '.cache\huggingface'
$env:NLTK_DATA = Join-Path $root '.cache\nltk_data'
$env:TIKTOKEN_CACHE_DIR = Join-Path $root '.cache\tiktoken'
$env:TEMP = Join-Path $root '.cache\tmp'
$env:TMP = $env:TEMP
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir, $env:TEMP | Out-Null
Push-Location $root
try {
    if ($Background) {
        $process = Start-Process -FilePath $python -ArgumentList @('"' + $main + '"', '--serve-only') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'web.stdout.log') -RedirectStandardError (Join-Path $logDir 'web.stderr.log')
        Write-Host "Started process $($process.Id). Logs: $logDir"
        Write-Host 'The listening address is configured by WEBUI_HOST / WEBUI_PORT in .env.'
    } else {
        & $python $main --serve-only
        exit $LASTEXITCODE
    }
} finally { Pop-Location }
