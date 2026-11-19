# Evaluation Metrics Summary

## 1. ConPlag v2 — 5-Fold Problem-Disjoint Cross-Validation (`strip_boilerplate=True`)

| Model | Micro F1 | Micro Precision | Micro Recall | Micro ROC AUC | Micro FPR | Macro F1 (mean ± std) | Macro AUC (mean ± std) | Threshold | Low-FPR Thr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_tfidf` | 0.6738 | 0.7302 | 0.6255 | 0.8287 | 0.0879 | 0.6658 ± 0.0907 | 0.8348 ± 0.0776 | 0.4664 | 0.5126 |
| `sublinear_tfidf` | 0.7104 | 0.7568 | 0.6693 | 0.8773 | 0.0818 | 0.6864 ± 0.0878 | 0.8812 ± 0.0469 | 0.4339 | 0.4711 |
| `markov_jsd` | 0.6167 | 0.5301 | 0.7371 | 0.8034 | 0.2485 | 0.5539 ± 0.0783 | 0.8093 ± 0.0578 | 0.9433 | 0.9914 |
| `markov_frobenius` | 0.6040 | 0.5155 | 0.7291 | 0.7972 | 0.2606 | 0.5369 ± 0.0582 | 0.8029 ± 0.0552 | -0.9179 | -0.3160 |
| `pairwise_lgbm` | 0.7598 | 0.7510 | 0.7689 | 0.9107 | 0.0970 | 0.7204 ± 0.1059 | 0.9229 ± 0.0472 | 0.3555 | 0.5210 |

## 2. ConPlag v2 → v1 Cross-Domain Shift & Boilerplate Ablation

| Model | Regime | v2 OOF F1 | v1 Zero-Shot F1 | F1 Shift (v1 - v2) | |F1 Shift| | v1 ROC AUC | v1 FPR |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_tfidf` | `strip=True` | 0.6738 | 0.6308 | -0.0430 | 0.0430 | 0.8019 | 0.0439 |
| `sublinear_tfidf` | `strip=True` | 0.7104 | 0.6618 | -0.0486 | 0.0486 | 0.8485 | 0.0364 |
| `markov_jsd` | `strip=True` | 0.6167 | 0.5698 | -0.0469 | 0.0469 | 0.7398 | 0.1864 |
| `markov_frobenius` | `strip=True` | 0.6040 | 0.5282 | -0.0758 | 0.0758 | 0.7310 | 0.2530 |
| `pairwise_lgbm` | `strip=True` | 0.7598 | 0.7500 | -0.0098 | 0.0098 | 0.8223 | 0.0136 |
| `raw_tfidf` | `strip=False` | 0.6538 | 0.5435 | -0.1103 | 0.1103 | 0.7628 | 0.2409 |
| `sublinear_tfidf` | `strip=False` | 0.6813 | 0.5992 | -0.0821 | 0.0821 | 0.7899 | 0.1439 |
| `markov_jsd` | `strip=False` | 0.6010 | 0.4992 | -0.1018 | 0.1018 | 0.6950 | 0.3121 |
| `markov_frobenius` | `strip=False` | 0.5942 | 0.5117 | -0.0825 | 0.0825 | 0.7098 | 0.3424 |
| `pairwise_lgbm` | `strip=False` | 0.7294 | 0.6683 | -0.0611 | 0.0611 | 0.8048 | 0.0303 |

## 3. IR-Plag Zero-Shot Evaluation (Trained on ConPlag v2, `strip_boilerplate=True`)

### Overall Metrics

| Model | Operating Point | Threshold | Precision | Recall | F1 | ROC AUC | FPR |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_tfidf` | F1-optimal | 0.4664 | 0.8320 | 0.8648 | 0.8481 | 0.7223 | 0.5905 |
| `raw_tfidf` | Low-FPR (<=5% v2) | 0.5126 | 0.8324 | 0.8254 | 0.8289 | 0.7223 | 0.5619 |
| `sublinear_tfidf` | F1-optimal | 0.4339 | 0.8238 | 0.8958 | 0.8583 | 0.6936 | 0.6476 |
| `sublinear_tfidf` | Low-FPR (<=5% v2) | 0.4711 | 0.8315 | 0.8479 | 0.8396 | 0.6936 | 0.5810 |
| `markov_jsd` | F1-optimal | 0.9433 | 0.7925 | 0.7099 | 0.7489 | 0.6441 | 0.6286 |
| `markov_jsd` | Low-FPR (<=5% v2) | 0.9914 | 0.8319 | 0.5437 | 0.6576 | 0.6441 | 0.3714 |
| `markov_frobenius` | F1-optimal | -0.9179 | 0.7911 | 0.7042 | 0.7452 | 0.6284 | 0.6286 |
| `markov_frobenius` | Low-FPR (<=5% v2) | -0.3160 | 0.8638 | 0.5183 | 0.6479 | 0.6284 | 0.2762 |
| `pairwise_lgbm` | F1-optimal | 0.3555 | 0.8041 | 0.7746 | 0.7891 | 0.6318 | 0.6381 |
| `pairwise_lgbm` | Low-FPR (<=5% v2) | 0.5210 | 0.8014 | 0.6366 | 0.7096 | 0.6318 | 0.5333 |

### Per-Level Recall (L1–L6) and NON_PLAG FPR

| Model | Operating Point | L1 | L2 | L3 | L4 | L5 | L6 | NON_PLAG FPR |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_tfidf` | F1-optimal | 1.0000 | 0.9107 | 0.8772 | 0.8500 | 0.7966 | 0.7619 | 0.5905 |
| `raw_tfidf` | Low-FPR | 1.0000 | 0.8750 | 0.8596 | 0.8000 | 0.7288 | 0.6984 | 0.5619 |
| `sublinear_tfidf` | F1-optimal | 1.0000 | 0.9286 | 0.9298 | 0.9167 | 0.8305 | 0.7778 | 0.6476 |
| `sublinear_tfidf` | Low-FPR | 0.9667 | 0.8750 | 0.8596 | 0.8667 | 0.7627 | 0.7619 | 0.5810 |
| `markov_jsd` | F1-optimal | 1.0000 | 0.9464 | 1.0000 | 0.7333 | 0.3559 | 0.2698 | 0.6286 |
| `markov_jsd` | Low-FPR | 1.0000 | 0.9107 | 0.8947 | 0.4167 | 0.0678 | 0.0317 | 0.3714 |
| `markov_frobenius` | F1-optimal | 1.0000 | 0.9464 | 1.0000 | 0.7167 | 0.3390 | 0.2698 | 0.6286 |
| `markov_frobenius` | Low-FPR | 1.0000 | 0.8750 | 0.8070 | 0.4000 | 0.0508 | 0.0317 | 0.2762 |
| `pairwise_lgbm` | F1-optimal | 1.0000 | 1.0000 | 0.9298 | 0.8333 | 0.5254 | 0.3968 | 0.6381 |
| `pairwise_lgbm` | Low-FPR | 1.0000 | 0.9643 | 0.7368 | 0.6000 | 0.3051 | 0.2540 | 0.5333 |
