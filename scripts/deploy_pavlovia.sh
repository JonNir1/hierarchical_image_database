#!/usr/bin/env bash
# deploy_pavlovia.sh — Sync pavlovia_deploy with main and push to Pavlovia.
#
# Usage (from anywhere inside the repo):
#   bash scripts/deploy_pavlovia.sh
#
# Strategy:
#   Uses "git checkout main -- ." to sync file content from main WITHOUT
#   resetting branch history. pavlovia_deploy always fast-forwards, so
#   Pavlovia's server-side "git pull" never hits "unrelated histories".
#   No force-push needed or used.
#
# What it does:
#   1. Fast-forwards local main from origin/main
#   2. Switches to pavlovia_deploy; syncs content from main
#   3. Appends the include-based Pavlovia gitignore block
#   4. Removes dev-only files from the index (include-based)
#   5. Commits (if anything changed) and pushes to the pavlovia remote
#   6. Returns you to main
#
# Pavlovia receives only: .gitignore, index.html, SpAM_Task/, images/
# Any new directory added to the repo root is automatically excluded.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Ensure local main is current
# ---------------------------------------------------------------------------
echo "==> Updating local main from origin..."
git checkout main
git pull --ff-only origin main

# ---------------------------------------------------------------------------
# 2. Sync content from main
#    "git checkout main -- ." copies main's files into the index and working
#    tree without moving the branch pointer — pavlovia_deploy stays put and
#    history always extends forward (no reset, no rewrite).
# ---------------------------------------------------------------------------
echo "==> Syncing content from main into pavlovia_deploy..."
git checkout pavlovia_deploy
git checkout main -- .

# ---------------------------------------------------------------------------
# 3. Append Pavlovia-only gitignore block (include-based)
#    "git checkout main -- ." restores main's .gitignore (no Pavlovia block),
#    so we always re-append it here.
# ---------------------------------------------------------------------------
echo "==> Appending Pavlovia-only gitignore block..."
cat >> .gitignore << 'GITIGNORE_BLOCK'

# ── pavlovia_deploy branch only ─────────────────────────────────────────────
# Include-based: keep only what Pavlovia needs to serve the experiment.
# Everything else at repo root is excluded automatically — no need to update
# this block when new dev directories are added to the repo.
/*
!/.gitignore
!/index.html
!/SpAM_Task/
!/images/
GITIGNORE_BLOCK

# ---------------------------------------------------------------------------
# 4. Remove dev-only files from the index
#    Keep only: .gitignore, index.html, SpAM_Task/, images/
# ---------------------------------------------------------------------------
echo "==> Removing dev-only files from index..."
TO_REMOVE=$(git ls-files | grep -vE '^(\.gitignore$|index\.html$|SpAM_Task/|images/)' || true)
if [ -n "$TO_REMOVE" ]; then
    echo "$TO_REMOVE" | xargs git rm --cached --ignore-unmatch
fi

# ---------------------------------------------------------------------------
# 5. Commit and push
#    Plain push (no --force): history always fast-forwards.
#    If the push is rejected, someone pushed directly to Pavlovia GitLab —
#    investigate before redeploying.
# ---------------------------------------------------------------------------
echo "==> Committing..."
git add .gitignore
if git diff --cached --quiet; then
    echo "No changes since last deploy — skipping commit."
else
    git commit -m "deploy: sync from main + exclude developer-only paths"
fi

echo "==> Pushing to Pavlovia..."
git push pavlovia pavlovia_deploy:main

# ---------------------------------------------------------------------------
# 6. Return to main
# ---------------------------------------------------------------------------
git checkout main

echo ""
echo "Done. Pavlovia is up to date with main."
