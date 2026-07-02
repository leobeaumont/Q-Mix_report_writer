# Baseline reference — 2026-07-02 (pre-upgrade code)

Captured for `qmix_upgrade_plan.md` Stage 0.1, before any upgrade-plan change.

- `handcrafted_trace.json` — execution trace of the last full pre-change run
  (2026-07-01, task "controled fission and fusion reactions", copied from the
  repo-root trace produced by that run).
- `report_raw.md` — the raw report of the same run
  (`output/2026-07-01_141533_controled-fission-and-fusion-reactions/`).

Offline test battery on the pre-change code (2026-07-02):

| Suite | Result |
|---|---|
| test_review_pipeline_fixes | 5 passed |
| test_directive_review | 3 passed |
| test_remove_duplicate_section | 2 passed |
| test_pbds_activation | 5 passed |
| test_pbds_pipeline | 9 passed, 0 skipped |
| test_pbds_researcher_integration | 5 passed |
| test_upgrade_plan_guards | 20 passed |
| test_upgrade_plan_acceptance | 0 passed, 12 pending, 0 failed |

Live suites (`test_remove_section_live.py`, `test_scorer.py`) not run — require Ollama.

Regression checkpoints (plan 2.5 / 3.4 / 4.8) compare a fresh run's trace against
`handcrafted_trace.json`: same phase sequence, same round patterns per phase,
sections + bibliography + abstract present in the final report.
