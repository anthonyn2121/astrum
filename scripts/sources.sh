#!/usr/bin/env bash
#
# Print the repo's hand-written source files of one language, NUL-separated,
# for formatters/linters (`make fmt`, `make fmt-check`, `make lint`):
#   scripts/sources.sh cpp|py | xargs -0 -r <tool>
# Files come from git — tracked plus untracked-but-not-ignored — so build dirs,
# virtualenvs and the ignored local workspaces (controllers/rl, …) are never
# touched; tracked files deleted in the working tree are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    cpp) globs=('*.cpp' '*.hpp' '*.cc' '*.h') ;;
    py) globs=('*.py') ;;
    *)
        echo "usage: $0 cpp|py" >&2
        exit 2
        ;;
esac

git ls-files -z --cached --others --exclude-standard -- "${globs[@]}" |
    while IFS= read -r -d '' file; do
        if [[ -e "${file}" ]]; then
            printf '%s\0' "${file}"
        fi
    done
