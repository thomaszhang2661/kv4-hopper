#!/bin/bash
# 11-19: rerun the jobs behind one paper table or figure, as recorded in repro/README.md (SubmitLines from sacct).
#   repro/reproduce.sh list                 # the labels
#   repro/reproduce.sh tab:xqa              # print the sbatch lines (dry run)
#   repro/reproduce.sh tab:xqa --submit     # submit them on the cluster (run from ~/kv4-hopper on the login node)
# Before submitting (KV4_COMMIT is a label of our development repository, not a checkout here): drop
# any Slurm account flag (-A ...), and add ABACKEND=FLASHINFER where the README says the original
# shell exported it. Speed jobs: H200 SXM only (--exclude=d4077), all rows of one table in one job.
set -euo pipefail
cd "$(dirname "$0")/.."
README=repro/README.md
labels() { grep -o '^## \(Table\|Figure\).*' "$README" | grep -o '`[a-z]*:[a-z0-9]*`' | tr -d '`' | sort -u; }
[ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 1; }
if [ "$1" = list ]; then labels; exit 0; fi
label=$1
labels | grep -qx "$label" || { echo "unknown label $label; try: $0 list" >&2; exit 1; }
# the sbatch lines of the section whose heading names the label, up to the next "## " heading
cmds=$(awk -v l="\`$label\`" '/^## /{on = index($0, l) > 0; next} on && /^ *sbatch /{sub(/^ +/, ""); print}' "$README")
[ -n "$cmds" ] || { echo "$label: no jobs (no data, or see README)" >&2; exit 0; }
if [ "${2:-}" != --submit ]; then echo "$cmds"; exit 0; fi
command -v sbatch >/dev/null || { echo "--submit runs on the cluster login node (sbatch not found)" >&2; exit 1; }
while IFS= read -r c; do
  c=$(printf "%s" "$c" | sed -E "s/-A [^ ]+ //g")
  echo "+ $c"; eval "$c"
done <<< "$cmds"
