#!/bin/bash
# Copies the released raw results into the script folders, so that every generator and paper/build_all.sh run without re-executing the experiments.
set -e
cd "$(dirname "$0")"
for d in real astro runs gefs; do
  for f in results/$d/*; do
    b=$(basename "$f")
    if [ -d "$f" ]; then mkdir -p "$d/$b"; cp "$f"/* "$d/$b/"; continue; fi
    case "$b" in
      *.json.gz|*.npz.gz|*.pkl.gz|*.gpkg.gz) gunzip -c "$f" > "$d/${b%.gz}" ;;
      *) cp "$f" "$d/$b" ;;
    esac
  done
done
echo "results restored"
