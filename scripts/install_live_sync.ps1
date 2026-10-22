# Install or remove the live-sync setup (commit hook + scheduled task).
#
#   powershell -ExecutionPolicy Bypass -File scripts/install_live_sync.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/install_live_sync.ps1 -Remove
#
# Everything applies to this clone only.

param([switch]$Remove)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$taskName = "go2w-locomotion-live-sync"
$syncScript = Join-Path $repo "scripts\sync_repo.py"
$logFile = Join-Path $repo "logs\autosync.log"

function Get-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) { return $git.Source }
    foreach ($candidate in @(
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files\Git\bin\git.exe"
    )) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "git not found"
}

$git = Get-Git

if ($Remove) {
    schtasks /Delete /TN $taskName /F 2>$null | Out-Null
    & $git -C $repo config --unset core.hooksPath 2>$null
    Write-Host "[live-sync] removed scheduled task and commit hook"
    Write-Host "[live-sync] pull.ff=only is left in place (harmless; unset manually if unwanted)"
    exit 0
}

if (-not (Test-Path $syncScript)) {
    throw "missing $syncScript"
}

# 1) commit-time push hook
& $git -C $repo config core.hooksPath .githooks
# 2) pulls may only fast-forward
& $git -C $repo config pull.ff only

# 3) scheduled pull/ push every 2 minutes
$pythonw = Join-Path $repo ".venv311\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    $pythonwCmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $pythonwCmd) { $pythonwCmd = Get-Command python.exe }
    if (-not $pythonwCmd) { throw "no python interpreter found for the scheduled task" }
    $pythonw = $pythonwCmd.Source
}

$action = "`"$pythonw`" `"$syncScript`" --quiet --push"
schtasks /Create /TN $taskName /SC MINUTE /MO 2 /TR $action /F | Out-Null

if (-not (Test-Path (Split-Path -Parent $logFile))) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $logFile) -Force | Out-Null
}

# touch the log so the first scheduled run has a destination
Add-Content -Path $logFile -Value "$(Get-Date -Format o) [live-sync] installed (task every 2 min, hook on commit)"

schtasks /Run /TN $taskName | Out-Null
Start-Sleep -Seconds 4

Write-Host "[live-sync] installed:"
Write-Host "  core.hooksPath = .githooks (push on commit)"
Write-Host "  pull.ff = only"
Write-Host "  scheduled task '$taskName' every 2 minutes"
Write-Host "  log: $logFile"
Write-Host ""
Write-Host "[live-sync] last log lines:"
Get-Content -Path $logFile -Tail 5 | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "[live-sync] verify with: python scripts/sync_repo.py --status"
Write-Host "[live-sync] remove with: powershell -ExecutionPolicy Bypass -File scripts/install_live_sync.ps1 -Remove"
