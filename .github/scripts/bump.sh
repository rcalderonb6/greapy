#!/usr/bin/env bash
set -euo pipefail

BUMP_TYPE="patch"
DRY_RUN=false
NO_PUSH=false
CREATE_RELEASE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [patch|minor|major]

Bump the cplots version, commit, tag, and push.

Arguments:
  patch | minor | major   Component to bump (default: patch)

Options:
  -d, --dry-run     Print what would happen without making any changes
  -n, --no-push     Commit and tag locally, skip git push
  -r, --release     Create a GitHub release after pushing (requires gh CLI)
  -h, --help        Show this help
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        patch|minor|major) BUMP_TYPE="$1"; shift ;;
        -d|--dry-run)      DRY_RUN=true;   shift ;;
        -n|--no-push)      NO_PUSH=true;   shift ;;
        -r|--release)      CREATE_RELEASE=true; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) die "unknown argument: $1"; usage >&2; exit 1 ;;
    esac
done

# Pre-flight: clean working tree
[[ -z "$(git status --porcelain)" ]] || die "working tree is dirty — commit or stash changes first"

# Pre-flight: warn if not on main
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
    read -r -p "warning: not on main (current: $BRANCH). Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

# Compute new version without applying it
OLD_VERSION=$(uv version | awk '{print $NF}')
IFS='.' read -r V_MAJOR V_MINOR V_PATCH <<< "$OLD_VERSION"
case "$BUMP_TYPE" in
    major) NEW_VERSION="$((V_MAJOR+1)).0.0" ;;
    minor) NEW_VERSION="${V_MAJOR}.$((V_MINOR+1)).0" ;;
    patch) NEW_VERSION="${V_MAJOR}.${V_MINOR}.$((V_PATCH+1))" ;;
esac

echo "  bump type : $BUMP_TYPE"
echo "  old       : $OLD_VERSION"
echo "  new       : $NEW_VERSION"

if $DRY_RUN; then
    echo "(dry run — no changes made)"
    exit 0
fi

# Apply
uv version --bump "$BUMP_TYPE"
git add pyproject.toml uv.lock
git commit -m "Bump version to $NEW_VERSION"
git tag "v$NEW_VERSION"
echo "Tagged v$NEW_VERSION"

if $NO_PUSH; then
    echo "(--no-push: skipping git push)"
    exit 0
fi

git push && git push --tags
echo "Pushed v$NEW_VERSION"

if $CREATE_RELEASE; then
    command -v gh &>/dev/null || die "gh CLI not found — install it to create GitHub releases"
    gh release create "v$NEW_VERSION" --generate-notes --title "v$NEW_VERSION"
    echo "GitHub release v$NEW_VERSION created"
fi

echo "Done: $OLD_VERSION → $NEW_VERSION"
