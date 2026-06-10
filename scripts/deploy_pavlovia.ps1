#Requires -Version 5.1
<#
.SYNOPSIS
    Sync pavlovia_deploy with main and push to Pavlovia.

.DESCRIPTION
    Strategy:
    Uses "git checkout main -- ." to sync file content from main WITHOUT
    resetting branch history. pavlovia_deploy always fast-forwards, so
    Pavlovia's server-side "git pull" never hits "unrelated histories".
    No force-push needed or used.

    1. Fast-forwards local main from origin/main
    2. Switches to pavlovia_deploy; syncs content from main
    3. Appends the include-based Pavlovia gitignore block
    4. Removes dev-only files from the index (include-based)
    5. Commits (if anything changed) and pushes to the pavlovia remote
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
# 2. Sync content from main
#    "git checkout main -- ." copies main's files into the index and working
#    tree without moving the branch pointer -- pavlovia_deploy stays put and
#    history always extends forward (no reset, no rewrite).
# ---------------------------------------------------------------------------
Write-Host "==> Syncing content from main into pavlovia_deploy..."
Invoke-Git checkout pavlovia_deploy
Invoke-Git checkout main -- .

# ---------------------------------------------------------------------------
# 3. Append Pavlovia-only gitignore block (include-based)
#    "git checkout main -- ." restores main's .gitignore (no Pavlovia block),
#    so we always re-append it here.
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
#    Plain push (no --force): history always fast-forwards.
#    If the push is rejected, someone pushed directly to Pavlovia GitLab --
#    investigate before redeploying.
# ---------------------------------------------------------------------------
Write-Host "==> Committing..."
Invoke-Git add .gitignore
git diff --cached --quiet
$NothingStaged = ($LASTEXITCODE -eq 0)
if ($NothingStaged) {
    Write-Host "No changes since last deploy -- skipping commit."
} else {
    Invoke-Git commit -m "deploy: sync from main + exclude developer-only paths"
}

Write-Host "==> Pushing to Pavlovia..."
Invoke-Git push pavlovia pavlovia_deploy:main

# ---------------------------------------------------------------------------
# 6. Return to main
# ---------------------------------------------------------------------------
Invoke-Git checkout main

Write-Host ""
Write-Host "Done. Pavlovia is up to date with main."
