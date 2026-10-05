$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$main = Join-Path $root 'main.py'
$executables = @((Join-Path $root '.venv\Scripts\python.exe'), (Join-Path $root '.runtime\python\python.exe'))
$mainPattern = [regex]::Escape($main) + '"?\s+--serve-only(?:\s|$)'
$processes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -in $executables -and $_.CommandLine -match $mainPattern
})
if (-not $processes) { Write-Host 'This deployment is not running.'; return }
foreach ($process in $processes) {
    Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
}
Write-Host 'Stopped this deployment. Its configuration and data are preserved.'
