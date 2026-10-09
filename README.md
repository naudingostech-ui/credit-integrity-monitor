# Credit integrity monitor

Independent, explainable integrity monitoring for environmental credit markets.
The first module is a **zombie credit detector**, backtested on the 2021-22 episode
in which millions of old, low-demand carbon credits were bridged onto Polygon.

**Case study (one page):** https://naudingostech-ui.github.io/credit-integrity-monitor/

## The question

Using only data that was public at the time, could an automated monitor have ranked
the projects feeding the bridge above everything else moving in the market?

## Result

One signal, the volume of credits at least six years past their vintage that moved in a
90-day window, ranked bridged projects far better than simply watching volume.

|                                         | Detector | Volume baseline |
|-----------------------------------------|---------:|----------------:|
| Ranking quality (AUC; 0.5 = coin flip)  | **0.85** | 0.70 |
| Real hits among the top 20 alerts       | **~8**   | ~3 |
| Worst single month (AUC)                | **0.79** | 0.65 |
| First half only, Nov 2021 to Jan 2022   | **0.88** | 0.72 |
| Second half only, Feb to Jun 2022       | **0.84** | 0.70 |
| "Old" = 4+ years instead of 6           | **0.84** | 0.70 |
| "Old" = 8+ years instead of 6           | **0.83** | 0.70 |
| Same test on Berkeley's registry files  | **0.82** | 0.71 |

8 monthly evaluations, about 1,000 active projects each. Pass criteria were fixed before
the results were seen.

**Second data source.** The whole test was rerun on the raw registry files published by the
Berkeley Carbon Trading Project, processed with separate code (`ingest_vrod.py`). The same
signal came out on top again, without any re-tuning. Both sources draw on the same registries,
so this rules out a quirk in one dataset's processing, not the limits of a single episode.

## Reproduce it

Requires Python 3.10+ and about 1 GB of free disk space.

```
cd zombie_detector
pip install -r requirements.txt
python ingest.py            # downloads CarbonPlan OffsetsDB and normalises it
python backtest.py          # headline numbers
python ablation.py          # every signal and combination, side by side
```

All robustness checks in one go: `robustness.ps1` (Windows) or `robustness.sh`
(macOS/Linux). Output goes to `zombie_detector/results/`.
No internet? `python synth.py` creates synthetic data to test the code (its scores say
nothing about real performance).

## Method

- **No hindsight.** A score for month T uses only transactions dated up to T.
- **Label.** A project is a positive if at least 20% of its retirements in the window carry
  bridge markers (Toucan or C3 identifiers, Polygon addresses). Markers are used only for
  grading, never as input.
- **Baseline.** Rank projects by how much they moved. A monitor that cannot beat this adds nothing.
- **Explainable.** No machine learning. Each alert lists the tags that fired (old vintages,
  long dormancy, backlog cleared, sudden burst), computed as robust z-scores against the
  projects active that month.

## What didn't work, and why that matters

The popular story was dormant projects waking up when a price appeared. The data disagreed:
dormancy ranked bridged projects worse than chance (AUC 0.42), because most of the flow came
from active projects clearing out old inventory. A five-signal model scored 0.62 while the
best single signal inside it scored 0.85. `diagnose.py` and `ablation.py` are the tools that
exposed both problems; they are kept so anyone can rerun the same checks.

## Limits

- The label is "routed through a bridge", not "low quality".
- One historical episode. Transfer to future waves, including nature and biodiversity
  credits, is untested.
- The final signal was chosen from 12 candidates on the same months; the time split,
  threshold checks and the second data source exist for that reason. Its worst month on the
  Berkeley data (March 2022) was barely above the baseline.
- OffsetsDB stopped updating regularly in mid-2026, so live monitoring needs direct registry ingestion.

## Layout

```
docs/index.html            one-page case study
zombie_detector/
  ingest.py                OffsetsDB -> internal schema
  ingest_vrod.py           Berkeley registry files -> the same schema (second source)
  detector.py              point-in-time signals, score and explanation tags
  backtest.py              monthly evaluation against the volume baseline
  ablation.py              each signal and combination on the same months
  diagnose.py              per-signal AUC, label audit, missed positives
  synth.py                 synthetic data for offline tests
  robustness.ps1 / .sh     reproduces every number above
  results/                 saved outputs
```

## Data

Berkeley Carbon Trading Project, *Voluntary Registry Offsets Database*, registry files v2026-08,
https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database, CC BY 4.0.
The raw registry files are not redistributed here; download them from the link above.

CarbonPlan, *OffsetsDB*, https://carbonplan.org/research/offsets-db, used under CarbonPlan's
[terms of data access](https://github.com/carbonplan/offsets-db-data/blob/main/TERMS_OF_DATA_ACCESS).
This project is independent and not affiliated with CarbonPlan.

## Contact

Ed Orlovski, naudingostech@gmail.com. Critique of the method is especially welcome.

## License

Code: MIT. See `LICENSE`.
