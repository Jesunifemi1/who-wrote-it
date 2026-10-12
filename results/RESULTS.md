# Results at a glance

Test accuracy, 6 classes (chance = 0.167). Prompt-level split: test prompts never seen in training.

| View | TextCNN | BiLSTM | TF-IDF + LogReg |
|---|---|---|---|
| Output only (RQ1) | 0.577 ± 0.012 (3 seeds) | 0.576 ± 0.011 (3 seeds) | 0.620 |
| Input only (RQ2) | 0.167 ± 0.001 (3 seeds) | 0.167 ± 0.000 (3 seeds) | 0.167 |
| Input + output (RQ2) | 0.600 ± 0.011 (3 seeds) | 0.577 ± 0.012 (3 seeds) | 0.608 |

RQ4 ablations (test accuracy after retraining on transformed text):

| Text | TextCNN | BiLSTM |
|---|---|---|
| no_open_close | 0.511 (1 run) | n/a |
| no_format | 0.428 ± 0.006 (2 seeds) | 0.443 (1 run) |
| words_only | 0.396 (1 run) | n/a |
| content_only | 0.240 ± 0.017 (2 seeds) | 0.309 (1 run) |
| skeleton | 0.589 ± 0.012 (2 seeds) | 0.601 (1 run) |
| first64 | 0.537 ± 0.012 (2 seeds) | 0.516 (1 run) |

Stylometric features only (gradient-boosted trees): 0.590

See report/Project_Report.pdf for RQ2 confound, RQ3 cross-task / cross-source / cross-version results and the full analysis.
