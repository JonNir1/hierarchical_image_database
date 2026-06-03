#!/usr/bin/env bash
# deploy_pavlovia.sh — Rebuild pavlovia_deploy from main and push to Pavlovia.
#
# Usage (from anywhere inside the repo):
#   bash scripts/deploy_pavlovia.sh
#
# What it does:
#   1. Fast-forwards local main from origin/main
#   2. Resets pavlovia_deploy to main
#   3. Appends the include-based Pavlovia gitignore block
#   4. Removes dev-only files from the index (include-based)
#   5. Commits and force-pushes to the pavlovia remote
#   6. Returns you to main
#
# Pavlovia receives only: .gitignore, index.html, SpAM_Task/, images/
# Any new directory added to the repo root is automatically excluded —
# no changes to this script needed.

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
# 2. Rebuild pavlovia_deploy from main
# ---------------------------------------------------------------------------
echo "==> Rebuilding pavlovia_deploy from main..."
git checkout pavlovia_deploy
git reset --hard main

# ---------------------------------------------------------------------------
# 3. Append Pavlovia-only gitignore block (include-based)
#    main's .gitignore handles project-level ignores; this block restricts
#    the root to only what Pavlovia needs to serve the experiment.
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
# ---------------------------------------------------------------------------
echo "==> Committing..."
git add .gitignore
git commit -m "deploy: re-derive from main + exclude developer-only paths"

echo "==> Pushing to Pavlovia..."
git push pavlovia pavlovia_deploy:main --force-with-lease

# ---------------------------------------------------------------------------
# 6. Return to main
# ---------------------------------------------------------------------------
git checkout main

echo ""
echo "Done. Pavlovia is up to date with main."
