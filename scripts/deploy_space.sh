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
# Check BOTH tracked (diff) and untracked files: untracked paths like
# .claude/, memory/, or auto-generated CLAUDE.md indexes would otherwise
# get swept into the deploy via `git add -A` below.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "✗ Working tree has uncommitted tracked changes. Commit or stash first." >&2
    exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "✗ Working tree has untracked files (would leak into deploy):" >&2
    git ls-files --others --exclude-standard | sed 's/^/  /' >&2
    echo "Remove/ignore them before deploying." >&2
    exit 1
fi

original_branch="$(git symbolic-ref --short HEAD)"
trap 'git checkout "$original_branch" 2>/dev/null || true' EXIT

git checkout space-deploy
# space-deploy is an orphan branch with no shared history with main —
# that's what keeps the docs/samples/ PNGs (rejected by HF Xet) out of
# the Space repo. Don't `git reset --hard main` here: it would attach
# space-deploy to main's history and push the PNGs.
# Instead: pull main's tree into the working dir, drop index entries that
# are on space-deploy but not on main, strip excluded paths, and commit.
git checkout main -- .
# Drop index entries for anything present on space-deploy but not on main
# (e.g. CLAUDE.md indexes from a previous leaky deploy). Use -z so
# unicode filenames (`宇宙洪荒_baiwen_square.png` etc.) survive the loop.
while IFS= read -r -d '' f; do
    git cat-file -e "main:$f" 2>/dev/null || git rm -f --cached -- "$f" >/dev/null
done < <(git ls-files -z)
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
