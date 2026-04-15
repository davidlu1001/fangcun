#!/usr/bin/env bash
# Sync the `space-deploy` branch from `main` and force-push to HF Space.
#
# HF Spaces rejects binary files without Xet/LFS, so the deploy branch
# excludes docs/samples/*.png (gallery is GitHub-only) and substitutes
# the README gallery section with a pointer link.
#
# Usage:
#     scripts/deploy_space.sh

set -euo pipefail

main_sha="$(git rev-parse --short main)"
echo "→ Syncing space-deploy from main @ ${main_sha}"

# Refuse to run with uncommitted changes — they would leak into the deploy.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "✗ Working tree is dirty. Commit or stash first." >&2
    exit 1
fi

original_branch="$(git symbolic-ref --short HEAD)"
trap 'git checkout "$original_branch" 2>/dev/null || true' EXIT

git checkout space-deploy
# Replace tree with main's, then strip excluded paths.
git checkout main -- .
git rm -rf --cached docs/samples/ >/dev/null 2>&1 || true
rm -rf docs/samples/

# Patch README gallery section to point users at GitHub.
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
src = p.read_text(encoding="utf-8")
start = src.find("## 样品 Gallery")
end = src.find("## 功能")
if start == -1 or end == -1:
    raise SystemExit("README.md missing 样品 Gallery / 功能 sections — patch manually")
replacement = (
    "## 样品 Gallery\n\n"
    "样品图片见 [GitHub README](https://github.com/davidlu1001/fangcun#样品-gallery)。"
    "本页面为可交互体验，请使用上方界面直接生成。\n\n"
)
p.write_text(src[:start] + replacement + src[end:], encoding="utf-8")
PY

git add -A
if git diff --cached --quiet; then
    echo "✓ space-deploy already up to date with main @ ${main_sha}"
else
    git commit -m "deploy: HF Space build from main @ ${main_sha}"
fi

echo "→ Pushing to space:main"
git push space space-deploy:main --force
echo "✓ Deployed."
