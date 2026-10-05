[CmdletBinding()]
param([switch]$SkipFrontend, [switch]$SkipPython)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pythonRuntime = Join-Path $root '.runtime\python\python.exe'
$python = Join-Path $root '.venv\Scripts\python.exe'
$nodeDir = Join-Path $root '.runtime\node'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_CACHE_DIR = Join-Path $root '.cache\pip'
$env:NPM_CONFIG_CACHE = Join-Path $root '.cache\npm'
$env:TEMP = Join-Path $root '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

Push-Location $root
try {
    if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
    if (-not $SkipPython) {
        if (-not (Test-Path $pythonRuntime)) {
            $systemPython = Get-Command py -ErrorAction SilentlyContinue
            if (-not $systemPython) { $systemPython = Get-Command python -ErrorAction SilentlyContinue }
            if (-not $systemPython) { throw 'No Python runtime found. Install Python 3.12 or restore .runtime\python.' }
            $pythonRuntime = $systemPython.Source
        }
        if (-not (Test-Path $python)) {
            & $pythonRuntime -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv.' }
        }
        & $python -m pip install -r requirements.txt --no-compile --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
        & $python -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency validation failed.' }
    }
    if (-not $SkipFrontend) {
        $npm = Join-Path $nodeDir 'npm.cmd'
        if (Test-Path $npm) {
            $env:PATH = "$nodeDir;$env:PATH"
        } else {
            $systemNpm = Get-Command npm -ErrorAction SilentlyContinue
            if (-not $systemNpm) { throw 'No npm runtime found. Install Node.js 20.19+ or restore .runtime\node.' }
            $npm = $systemNpm.Source
        }
        Push-Location (Join-Path $root 'apps\dsa-web')
        try {
            & $npm ci --ignore-scripts --no-fund --no-audit
            if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
            & $npm run build
            if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
        } finally { Pop-Location }
        & $python scripts/check_static_assets.py
        if ($LASTEXITCODE -ne 0) { throw 'Static asset verification failed.' }
    }
    Write-Host 'Setup complete. Start with scripts\start-windows.cmd.'
} finally { Pop-Location }
