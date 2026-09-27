# Reproduces every number in the README. Run from this folder:
#   powershell -ExecutionPolicy Bypass -File robustness.ps1
$out = "results\robustness.txt"
New-Item -ItemType Directory -Force results | Out-Null
"=== Full period ===" | Out-File $out -Encoding utf8
python ablation.py | Out-File $out -Append -Encoding utf8
"`n=== Time split: first half ===" | Out-File $out -Append -Encoding utf8
python ablation.py --start 2021-11-01 --end 2022-01-01 | Out-File $out -Append -Encoding utf8
"`n=== Time split: second half ===" | Out-File $out -Append -Encoding utf8
python ablation.py --start 2022-02-01 --end 2022-06-01 | Out-File $out -Append -Encoding utf8
foreach ($age in 4, 8) {
  "`n=== Threshold: old = $age+ years ===" | Out-File $out -Append -Encoding utf8
  python ablation.py --old-age $age | Out-File $out -Append -Encoding utf8
}
"`n=== v0.3 backtest ===" | Out-File $out -Append -Encoding utf8
python backtest.py | Out-File $out -Append -Encoding utf8
"Done: $out"
