#!/usr/bin/env bash
# Run probe jobs from a job file; finished shards are skipped, so the script can be re-run after an interruption.
# Usage: code/run_jobs.sh <jobfile> [results-root, default results]
# Job line: model revision tok corpus docs chunk exact(0/1)
set -euo pipefail
cd "$(dirname "$0")/.."
jobs="$1"; root="${2:-results}"
while read -r model rev tok corpus docs chunk exact; do
  [[ -z "${model}" || "${model}" == \#* ]] && continue
  slug="${model//\//__}"
  out="${root}/${slug}/${rev}/${corpus}_c${chunk}_d${docs/:/-}.npz"
  if [[ -f "${out}" && -f "${out%.npz}.json" ]]; then echo "skip ${out}"; continue; fi
  flag=""; [[ "${exact}" == "1" ]] && flag="--exact-ref"
  mkdir -p "$(dirname "${out}")"
  python code/probe.py --model "${model}" --revision "${rev}" --tok "${tok}" --corpus "${corpus}" \
    --docs "${docs}" --chunk "${chunk}" ${flag} --threads "$(nproc)" --out "${out}" \
    > "${out%.npz}.log" 2>&1
done < "${jobs}"
