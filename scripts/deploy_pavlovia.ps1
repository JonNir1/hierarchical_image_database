#Requires -Version 5.1
<#
.SYNOPSIS
    Rebuild pavlovia_deploy from main and push to Pavlovia.

.DESCRIPTION
    1. Fast-forwards local main from origin/main
    2. Resets pavlovia_deploy to main
    3. Appends the include-based Pavlovia gitignore block
    4. Removes dev-only files from the index (include-based)
    5. Commits and force-pushes to the pavlovia remote
    6. Returns you to main

    Pavlovia receives only: .gitignore, index.html, SpAM_Task/, images/
    Any new directory added to the repo root is automatically excluded.

.EXAMPLE
    .\scripts\deploy_pavlovia.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Invoke-Git {
    git @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $args exited with code $LASTEXITCODE"
    }
}

# Resolve repo root (parent of the directory containing this script)
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

# ---------------------------------------------------------------------------
# 1. Ensure local main is current
# ---------------------------------------------------------------------------
Write-Host "==> Updating local main from origin..."
Invoke-Git checkout main
Invoke-Git pull --ff-only origin main

# ---------------------------------------------------------------------------
# 2. Rebuild pavlovia_deploy from main
# ---------------------------------------------------------------------------
Write-Host "==> Rebuilding pavlovia_deploy from main..."
Invoke-Git checkout pavlovia_deploy
Invoke-Git reset --hard main

# ---------------------------------------------------------------------------
# 3. Append Pavlovia-only gitignore block (include-based)
# ---------------------------------------------------------------------------
Write-Host "==> Appending Pavlovia-only gitignore block..."
$PavloviaBlock = @"


# -- pavlovia_deploy branch only ---------------------------------------------
# Include-based: keep only what Pavlovia needs to serve the experiment.
# Everything else at repo root is excluded automatically -- no need to update
# this block when new dev directories are added to the repo.
/*
!/.gitignore
!/index.html
!/SpAM_Task/
!/images/
"@
Add-Content -Path ".gitignore" -Value $PavloviaBlock -NoNewline:$false

# ---------------------------------------------------------------------------
# 4. Remove dev-only files from the index (keep only Pavlovia essentials)
# ---------------------------------------------------------------------------
Write-Host "==> Removing dev-only files from index..."
$ToRemove = git ls-files | Where-Object {
    $_ -notmatch '^(\.gitignore$|index\.html$|SpAM_Task/|images/)'
}
if ($ToRemove) {
    $ToRemove | ForEach-Object { Invoke-Git rm --cached --ignore-unmatch $_ }
}

# ---------------------------------------------------------------------------
# 5. Commit and push
# ---------------------------------------------------------------------------
Write-Host "==> Committing..."
Invoke-Git add .gitignore
Invoke-Git commit -m "deploy: re-derive from main + exclude developer-only paths"

Write-Host "==> Pushing to Pavlovia..."
Invoke-Git push pavlovia pavlovia_deploy:main --force-with-lease

# ---------------------------------------------------------------------------
# 6. Return to main
# ---------------------------------------------------------------------------
Invoke-Git checkout main

Write-Host ""
Write-Host "Done. Pavlovia is up to date with main."
