The current structure in task3.ipynb is good for Task 3 goals:

Heuristic observability and deterministic heuristic benchmark first.
Shared DQN backbone fixed.
Epsilon vs Boltzmann comparison with tracked results.
Warm-start ablation as an extra controlled experiment.
Deterministic validation, plots, and CSV exports.
Here is the practical Colab flow to follow.

Phase 1: FAST validation (quick, mandatory first)

Open task3.ipynb in Colab.
In Cell 4, keep FAST_RUN = True.
Run cells in order from Cell 2 to Cell 50.
Check key outputs:
Cell 10 and Cell 11: heuristic ranking table and summary.
Cell 30: warm-start ablation table.
Cell 37 and Cell 40: epsilon and boltzmann mini-grid summaries.
Cell 44: deterministic validation summary.
Cell 49: CSV export confirmation.
If all these pass, your pipeline is healthy.

Phase 2: FULL run (report-quality)

Go to Cell 4 and set FAST_RUN = False.
Increase seed coverage manually in Cell 35:
Change SEED_GRID_FAST from 2 seeds to at least 5 seeds, for example [42, 43, 44, 45, 46].
Optionally increase warm-start ablation cost in Cell 30 by raising WARMSTART_N_FAST only if runtime allows.
Re-run from Cell 14 onward, because constants and schedules change behavior downstream.
If you want fully clean outputs, run all cells from Cell 2 to Cell 50 again.
Important: this notebook uses episodes and max steps, not epochs. Your effective training budget is controlled by:

N_EPISODES_EXP and MAX_STEPS_PER_EP in Cell 4.
Per-experiment overrides in mini-grid and ablation cells.
Decision points you need to make

FAST vs FULL mode: Cell 4.
Number of seeds for statistical stability: Cell 35.
Whether warm-start ablation is part of final report or optional appendix: Cell 30 and discussion cells.
Best heuristic selection: automatic or manual?

Automatic.
In Cell 30, top-3 heuristics are selected automatically from the sorted heuristic table produced earlier.
If tied, order follows the sorted table order, so you do not need manual selection unless you want to force specific heuristics.
Recommended analysis workflow after FULL run

Use Cell 41 and Cell 43 tables to compare strategy-level performance.
Use Cell 44 deterministic summary for final comparison claim.
Use Cell 46 and Cell 47 plots for presentation figures.
Use CSV files from Cell 49 for report tables.
If you want, I can give you a final exact FULL configuration preset for Cell 4 and Cell 35 optimized for a 2-4 hour Colab session.

