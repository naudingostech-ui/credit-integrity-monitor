#!/usr/bin/env sh
# Reproduces every number in the README (macOS / Linux). Run from this folder.
set -e
mkdir -p results; out=results/robustness.txt
{ echo "=== Full period ==="; python ablation.py
  echo; echo "=== Time split: first half ==="; python ablation.py --start 2021-11-01 --end 2022-01-01
  echo; echo "=== Time split: second half ==="; python ablation.py --start 2022-02-01 --end 2022-06-01
  for age in 4 8; do echo; echo "=== Threshold: old = $age+ years ==="; python ablation.py --old-age $age; done
  echo; echo "=== v0.3 backtest ==="; python backtest.py; } > "$out"
echo "Done: $out"
