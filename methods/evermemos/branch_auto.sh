#!/usr/bin/env bash
set -e

# ===== Arguments =====
CURRENT_DATE=$1   # Current iteration start date, e.g. 20251201
NEXT_DATE=$2      # Next iteration start date, e.g. 20251215

if [[ -z "$CURRENT_DATE" || -z "$NEXT_DATE" ]]; then
  echo "Usage: branch_auto.sh <CURRENT_DATE> <NEXT_DATE>"
  exit 1
fi

CURRENT_DEV="${CURRENT_DATE}-dev"
RELEASE_BRANCH="release/${CURRENT_DATE}"
NEXT_DEV="${NEXT_DATE}-dev"

echo "=== Iteration Close Start ==="
echo "CURRENT_DEV=$CURRENT_DEV"
echo "RELEASE_BRANCH=$RELEASE_BRANCH"
echo "NEXT_DEV=$NEXT_DEV"

git config user.email "ci@gitlab.com"
git config user.name "GitLab CI"

git fetch origin

# 1. Verify current iteration branch exists
if ! git show-ref --verify --quiet "refs/remotes/origin/${CURRENT_DEV}"; then
  echo "ERROR: ${CURRENT_DEV} does not exist"
  exit 1
fi

# 2. Create release branch (based on current iteration)
if git show-ref --verify --quiet "refs/remotes/origin/${RELEASE_BRANCH}"; then
  echo "Release branch already exists, skipping"
else
  git checkout -B "${RELEASE_BRANCH}" "origin/${CURRENT_DEV}"
  git push origin "${RELEASE_BRANCH}"
fi

# 3. Create next iteration branch (based on dev)
if git show-ref --verify --quiet "refs/remotes/origin/${NEXT_DEV}"; then
  echo "Next dev branch already exists, skipping"
else
  git checkout -B "${NEXT_DEV}" "origin/${CURRENT_DEV}"
  git push origin "${NEXT_DEV}"
fi

echo "=== Iteration Close Done ==="