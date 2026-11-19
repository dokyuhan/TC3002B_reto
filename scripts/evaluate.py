"""Cross-domain and baseline evaluation pipeline for plagiarism detection models."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.baseline import (
    TfidfCosineBaseline,
    frobenius_distance,
    jensen_shannon_divergence,
)
from src.models.classifier import PairwisePlagiarismClassifier
from scripts.train import (
    DEFAULT_CONPLAG_V2_MANIFEST,
    DEFAULT_SMOOTHING_ALPHA,
    PreparedDataset,
    compute_binary_metrics,
    find_f1_optimal_threshold,
    find_low_fpr_threshold,
    prepare_manifest_dataset,
)

DEFAULT_CONPLAG_V1_MANIFEST = Path("data/processed/conplag_v1_manifest.parquet")
DEFAULT_IR_PLAG_MANIFEST = Path("data/processed/ir_plag_manifest.parquet")
DEFAULT_EVAL_DIR = Path("docs/evaluation")

MODEL_ORDER: tuple[str, ...] = (
    "raw_tfidf",
    "sublinear_tfidf",
    "markov_jsd",
    "markov_frobenius",
    "pairwise_lgbm",
)
IR_PLAG_LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")


def _markov_jsd_scores(X: np.ndarray) -> np.ndarray:
    """Compute similarity score 1 - JSD(P_A, P_B) so higher indicates greater similarity."""
    return np.asarray(
        [1.0 - jensen_shannon_divergence(X[i, 0], X[i, 1]) for i in range(X.shape[0])],
        dtype=np.float64,
    )


def _markov_frobenius_scores(X: np.ndarray) -> np.ndarray:
    """Compute similarity score -||P_A - P_B||_F so higher indicates greater similarity."""
    return np.asarray(
        [-frobenius_distance(X[i, 0], X[i, 1]) for i in range(X.shape[0])],
        dtype=np.float64,
    )


def _compute_curves(y_true: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Compute ROC and PR curve arrays serialized as finite float lists."""
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    prec, rec, pr_thr = precision_recall_curve(y_true, scores)
    return {
        "roc_curve": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) if np.isfinite(v) else float(np.max(scores) + 1.0) for v in roc_thr],
        },
        "pr_curve": {
            "precision": [float(v) for v in prec],
            "recall": [float(v) for v in rec],
            "thresholds": [float(v) for v in pr_thr],
        },
    }


def _evaluate_conplag_v2_cv(
    dataset: PreparedDataset,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Run 5-fold problem-disjoint CV on ConPlag v2 for all 5 models and train full models."""
    unique_folds = sorted(int(f) for f in np.unique(dataset.fold_ids))
    n_samples = dataset.y.shape[0]

    oof_scores: dict[str, np.ndarray] = {name: np.zeros(n_samples, dtype=np.float64) for name in MODEL_ORDER}
    per_fold_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in MODEL_ORDER}

    # Unsupervised Markov similarity scores are computed directly per pair.
    jsd_all = _markov_jsd_scores(dataset.X)
    frob_all = _markov_frobenius_scores(dataset.X)

    for fold in unique_folds:
        val_mask = dataset.fold_ids == fold
        train_mask = ~val_mask

        train_codes_a = [c for c, m in zip(dataset.codes_a, train_mask) if m]
        train_codes_b = [c for c, m in zip(dataset.codes_b, train_mask) if m]
        val_codes_a = [c for c, m in zip(dataset.codes_a, val_mask) if m]
        val_codes_b = [c for c, m in zip(dataset.codes_b, val_mask) if m]

        raw_tfidf = TfidfCosineBaseline(sublinear_tf=False, max_features=20000).fit(
            train_codes_a + train_codes_b
        )
        sub_tfidf = TfidfCosineBaseline(sublinear_tf=True, max_features=20000).fit(
            train_codes_a + train_codes_b
        )

        lgbm = PairwisePlagiarismClassifier()
        lgbm.fit(
            dataset.X[train_mask],
            dataset.y[train_mask],
            perplexities=(dataset.perp_a[train_mask], dataset.perp_b[train_mask]),
            groups=dataset.group_ids[train_mask],
        )

        fold_train_scores = {
            "raw_tfidf": raw_tfidf.score_pairs(train_codes_a, train_codes_b),
            "sublinear_tfidf": sub_tfidf.score_pairs(train_codes_a, train_codes_b),
            "markov_jsd": jsd_all[train_mask],
            "markov_frobenius": frob_all[train_mask],
            "pairwise_lgbm": lgbm.predict_proba(
                dataset.X[train_mask],
                perplexities=(dataset.perp_a[train_mask], dataset.perp_b[train_mask]),
            )[:, 1],
        }
        fold_val_scores = {
            "raw_tfidf": raw_tfidf.score_pairs(val_codes_a, val_codes_b),
            "sublinear_tfidf": sub_tfidf.score_pairs(val_codes_a, val_codes_b),
            "markov_jsd": jsd_all[val_mask],
            "markov_frobenius": frob_all[val_mask],
            "pairwise_lgbm": lgbm.predict_proba(
                dataset.X[val_mask],
                perplexities=(dataset.perp_a[val_mask], dataset.perp_b[val_mask]),
            )[:, 1],
        }

        for name in MODEL_ORDER:
            oof_scores[name][val_mask] = fold_val_scores[name]
            fold_thr = find_f1_optimal_threshold(dataset.y[train_mask], fold_train_scores[name])
            m = compute_binary_metrics(dataset.y[val_mask], fold_val_scores[name], threshold=fold_thr)
            per_fold_metrics[name].append(
                {
                    "fold": float(fold),
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "f1": float(m["f1"]),
                    "roc_auc": float(m["roc_auc"]),
                    "fpr": float(m["fpr"]),
                }
            )

    # Fit full-dataset models on ConPlag v2 for zero-shot cross-domain evaluation.
    full_raw_tfidf = TfidfCosineBaseline(sublinear_tf=False, max_features=20000).fit(
        dataset.codes_a + dataset.codes_b
    )
    full_sub_tfidf = TfidfCosineBaseline(sublinear_tf=True, max_features=20000).fit(
        dataset.codes_a + dataset.codes_b
    )
    full_lgbm = PairwisePlagiarismClassifier().fit(
        dataset.X,
        dataset.y,
        perplexities=(dataset.perp_a, dataset.perp_b),
        groups=dataset.group_ids,
    )
    fitted_models: dict[str, object] = {
        "raw_tfidf": full_raw_tfidf,
        "sublinear_tfidf": full_sub_tfidf,
        "markov_jsd": None,
        "markov_frobenius": None,
        "pairwise_lgbm": full_lgbm,
    }

    cv_results: dict[str, dict[str, object]] = {}
    for name in MODEL_ORDER:
        scores = oof_scores[name]
        opt_thr = find_f1_optimal_threshold(dataset.y, scores)
        low_fpr_thr = find_low_fpr_threshold(dataset.y, scores, max_fpr=0.05)

        macro_summary: dict[str, dict[str, float]] = {}
        for key in ("precision", "recall", "f1", "roc_auc", "fpr"):
            vals = np.asarray([r[key] for r in per_fold_metrics[name]], dtype=np.float64)
            macro_summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }

        curves = _compute_curves(dataset.y, scores)
        cv_results[name] = {
            "threshold": opt_thr,
            "threshold_low_fpr": low_fpr_thr,
            "micro_oof": compute_binary_metrics(dataset.y, scores, threshold=opt_thr),
            "micro_oof_low_fpr": compute_binary_metrics(dataset.y, scores, threshold=low_fpr_thr),
            "macro_cv": macro_summary,
            "per_fold": per_fold_metrics[name],
            "roc_curve": curves["roc_curve"],
            "pr_curve": curves["pr_curve"],
        }

    return cv_results, fitted_models


def _score_all_models(
    fitted_models: dict[str, object],
    dataset: PreparedDataset,
) -> dict[str, np.ndarray]:
    """Score a target dataset using all 5 models trained on ConPlag v2."""
    raw_tfidf: TfidfCosineBaseline = fitted_models["raw_tfidf"]  # type: ignore[assignment]
    sub_tfidf: TfidfCosineBaseline = fitted_models["sublinear_tfidf"]  # type: ignore[assignment]
    lgbm: PairwisePlagiarismClassifier = fitted_models["pairwise_lgbm"]  # type: ignore[assignment]

    return {
        "raw_tfidf": raw_tfidf.score_pairs(dataset.codes_a, dataset.codes_b),
        "sublinear_tfidf": sub_tfidf.score_pairs(dataset.codes_a, dataset.codes_b),
        "markov_jsd": _markov_jsd_scores(dataset.X),
        "markov_frobenius": _markov_frobenius_scores(dataset.X),
        "pairwise_lgbm": lgbm.predict_proba(
            dataset.X,
            perplexities=(dataset.perp_a, dataset.perp_b),
        )[:, 1],
    }


def _evaluate_ir_plag_levels(
    y_true: np.ndarray,
    scores: np.ndarray,
    levels: list[str],
    threshold: float,
) -> dict[str, float]:
    """Compute per-level recall for L1..L6 and FPR for NON_PLAG at threshold."""
    y_pred = (scores >= threshold).astype(np.uint8)
    levels_arr = np.asarray(levels, dtype=object)

    out: dict[str, float] = {}
    for lvl in IR_PLAG_LEVELS:
        mask = levels_arr == lvl
        if np.any(mask):
            out[lvl] = float(np.mean(y_pred[mask] == 1))
        else:
            out[lvl] = 0.0

    non_plag_mask = levels_arr == "NON_PLAG"
    if np.any(non_plag_mask):
        out["NON_PLAG_fpr"] = float(np.mean(y_pred[non_plag_mask] == 1))
    else:
        out["NON_PLAG_fpr"] = 0.0
    return out


def _build_markdown_tables(summary: dict[str, object]) -> str:
    """Format evaluation metrics into GitHub-flavored Markdown tables."""
    lines: list[str] = [
        "# Evaluation Metrics Summary",
        "",
        "## 1. ConPlag v2 — 5-Fold Problem-Disjoint Cross-Validation (`strip_boilerplate=True`)",
        "",
        "| Model | Micro F1 | Micro Precision | Micro Recall | Micro ROC AUC | Micro FPR | Macro F1 (mean ± std) | Macro AUC (mean ± std) | Threshold | Low-FPR Thr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    v2_stripped: dict[str, dict[str, object]] = summary["conplag_v2_cv"]  # type: ignore[assignment]
    for name in MODEL_ORDER:
        m = v2_stripped[name]
        micro = m["micro_oof"]  # type: ignore[index]
        macro = m["macro_cv"]  # type: ignore[index]
        lines.append(
            f"| `{name}` | {micro['f1']:.4f} | {micro['precision']:.4f} | {micro['recall']:.4f} | "
            f"{micro['roc_auc']:.4f} | {micro['fpr']:.4f} | "
            f"{macro['f1']['mean']:.4f} ± {macro['f1']['std']:.4f} | "
            f"{macro['roc_auc']['mean']:.4f} ± {macro['roc_auc']['std']:.4f} | "
            f"{m['threshold']:.4f} | {m['threshold_low_fpr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. ConPlag v2 → v1 Cross-Domain Shift & Boilerplate Ablation",
            "",
            "| Model | Regime | v2 OOF F1 | v1 Zero-Shot F1 | F1 Shift (v1 - v2) | |F1 Shift| | v1 ROC AUC | v1 FPR |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    v1_eval: dict[str, dict[str, dict[str, object]]] = summary["conplag_v1_cross_domain"]  # type: ignore[assignment]
    for regime_key, regime_label in (
        ("strip_boilerplate_true", "strip=True"),
        ("strip_boilerplate_false", "strip=False"),
    ):
        for name in MODEL_ORDER:
            entry = v1_eval[regime_key][name]
            metrics = entry["metrics"]  # type: ignore[index]
            lines.append(
                f"| `{name}` | `{regime_label}` | {entry['v2_oof_f1']:.4f} | "
                f"{metrics['f1']:.4f} | {entry['f1_shift']:+.4f} | {entry['abs_f1_shift']:.4f} | "
                f"{metrics['roc_auc']:.4f} | {metrics['fpr']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 3. IR-Plag Zero-Shot Evaluation (Trained on ConPlag v2, `strip_boilerplate=True`)",
            "",
            "### Overall Metrics",
            "",
            "| Model | Operating Point | Threshold | Precision | Recall | F1 | ROC AUC | FPR |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    ir_eval: dict[str, dict[str, object]] = summary["ir_plag_zero_shot"]  # type: ignore[assignment]
    for name in MODEL_ORDER:
        entry = ir_eval[name]
        for op_key, op_label in (("at_threshold", "F1-optimal"), ("at_threshold_low_fpr", "Low-FPR (<=5% v2)")):
            m = entry[op_key]  # type: ignore[index]
            lines.append(
                f"| `{name}` | {op_label} | {m['threshold']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['fpr']:.4f} |"
            )

    lines.extend(
        [
            "",
            "### Per-Level Recall (L1–L6) and NON_PLAG FPR",
            "",
            "| Model | Operating Point | L1 | L2 | L3 | L4 | L5 | L6 | NON_PLAG FPR |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for name in MODEL_ORDER:
        entry = ir_eval[name]
        for op_key, op_label in (
            ("per_level_at_threshold", "F1-optimal"),
            ("per_level_at_threshold_low_fpr", "Low-FPR"),
        ):
            lv = entry[op_key]  # type: ignore[index]
            lines.append(
                f"| `{name}` | {op_label} | {lv['L1']:.4f} | {lv['L2']:.4f} | {lv['L3']:.4f} | "
                f"{lv['L4']:.4f} | {lv['L5']:.4f} | {lv['L6']:.4f} | {lv['NON_PLAG_fpr']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    conplag_v2_manifest: Path = DEFAULT_CONPLAG_V2_MANIFEST,
    conplag_v1_manifest: Path = DEFAULT_CONPLAG_V1_MANIFEST,
    ir_plag_manifest: Path = DEFAULT_IR_PLAG_MANIFEST,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> dict[str, object]:
    """Run full 5-model evaluation across ConPlag v2, ConPlag v1 (with ablation), and IR-Plag."""
    v2_stripped = prepare_manifest_dataset(
        conplag_v2_manifest,
        language="java",
        strip_boilerplate=True,
        smoothing_alpha=smoothing_alpha,
    )
    v1_stripped = prepare_manifest_dataset(
        conplag_v1_manifest,
        language="java",
        strip_boilerplate=True,
        smoothing_alpha=smoothing_alpha,
    )
    ir_stripped = prepare_manifest_dataset(
        ir_plag_manifest,
        language="java",
        strip_boilerplate=True,
        smoothing_alpha=smoothing_alpha,
    )

    v2_cv_stripped, fitted_stripped = _evaluate_conplag_v2_cv(v2_stripped)

    # Zero-shot evaluation on ConPlag v1 (strip_boilerplate=True).
    v1_scores_stripped = _score_all_models(fitted_stripped, v1_stripped)
    v1_stripped_results: dict[str, dict[str, object]] = {}
    for name in MODEL_ORDER:
        thr = float(v2_cv_stripped[name]["threshold"])
        m_v1 = compute_binary_metrics(v1_stripped.y, v1_scores_stripped[name], threshold=thr)
        v2_f1 = float(v2_cv_stripped[name]["micro_oof"]["f1"])  # type: ignore[index]
        f1_shift = float(m_v1["f1"]) - v2_f1
        v1_stripped_results[name] = {
            "v2_oof_f1": v2_f1,
            "f1_shift": f1_shift,
            "abs_f1_shift": abs(f1_shift),
            "metrics": m_v1,
        }

    # Boilerplate ablation: train on ConPlag v2 (strip=False) and evaluate on ConPlag v1 (strip=False).
    v2_unstripped = prepare_manifest_dataset(
        conplag_v2_manifest,
        language="java",
        strip_boilerplate=False,
        smoothing_alpha=smoothing_alpha,
    )
    v1_unstripped = prepare_manifest_dataset(
        conplag_v1_manifest,
        language="java",
        strip_boilerplate=False,
        smoothing_alpha=smoothing_alpha,
    )
    v2_cv_unstripped, fitted_unstripped = _evaluate_conplag_v2_cv(v2_unstripped)
    v1_scores_unstripped = _score_all_models(fitted_unstripped, v1_unstripped)
    v1_unstripped_results: dict[str, dict[str, object]] = {}
    for name in MODEL_ORDER:
        thr = float(v2_cv_unstripped[name]["threshold"])
        m_v1 = compute_binary_metrics(v1_unstripped.y, v1_scores_unstripped[name], threshold=thr)
        v2_f1 = float(v2_cv_unstripped[name]["micro_oof"]["f1"])  # type: ignore[index]
        f1_shift = float(m_v1["f1"]) - v2_f1
        v1_unstripped_results[name] = {
            "v2_oof_f1": v2_f1,
            "f1_shift": f1_shift,
            "abs_f1_shift": abs(f1_shift),
            "metrics": m_v1,
        }

    # Zero-shot evaluation on IR-Plag (strip_boilerplate=True).
    ir_scores = _score_all_models(fitted_stripped, ir_stripped)
    ir_results: dict[str, dict[str, object]] = {}
    for name in MODEL_ORDER:
        thr = float(v2_cv_stripped[name]["threshold"])
        thr_low = float(v2_cv_stripped[name]["threshold_low_fpr"])
        curves = _compute_curves(ir_stripped.y, ir_scores[name])
        ir_results[name] = {
            "at_threshold": compute_binary_metrics(ir_stripped.y, ir_scores[name], threshold=thr),
            "at_threshold_low_fpr": compute_binary_metrics(
                ir_stripped.y, ir_scores[name], threshold=thr_low
            ),
            "per_level_at_threshold": _evaluate_ir_plag_levels(
                ir_stripped.y, ir_scores[name], ir_stripped.levels, threshold=thr
            ),
            "per_level_at_threshold_low_fpr": _evaluate_ir_plag_levels(
                ir_stripped.y, ir_scores[name], ir_stripped.levels, threshold=thr_low
            ),
            "roc_curve": curves["roc_curve"],
            "pr_curve": curves["pr_curve"],
        }

    summary: dict[str, object] = {
        "config": {
            "smoothing_alpha": float(smoothing_alpha),
            "models": list(MODEL_ORDER),
        },
        "conplag_v2_cv": v2_cv_stripped,
        "conplag_v2_cv_unstripped": v2_cv_unstripped,
        "conplag_v1_cross_domain": {
            "strip_boilerplate_true": v1_stripped_results,
            "strip_boilerplate_false": v1_unstripped_results,
        },
        "ir_plag_zero_shot": ir_results,
    }

    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / "metrics_summary.json"
    md_path = eval_dir / "metrics_tables.md"

    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_markdown_tables(summary), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint to run full evaluation and write summary JSON and Markdown tables."""
    summary = run_evaluation()
    lgbm_v2 = summary["conplag_v2_cv"]["pairwise_lgbm"]["micro_oof"]  # type: ignore[index]
    lgbm_v1_s = summary["conplag_v1_cross_domain"]["strip_boilerplate_true"]["pairwise_lgbm"]  # type: ignore[index]
    lgbm_v1_u = summary["conplag_v1_cross_domain"]["strip_boilerplate_false"]["pairwise_lgbm"]  # type: ignore[index]
    lgbm_ir = summary["ir_plag_zero_shot"]["pairwise_lgbm"]  # type: ignore[index]

    print(
        f"[ConPlag v2 CV - pairwise_lgbm] F1={lgbm_v2['f1']:.4f}, P={lgbm_v2['precision']:.4f}, "
        f"R={lgbm_v2['recall']:.4f}, AUC={lgbm_v2['roc_auc']:.4f}, FPR={lgbm_v2['fpr']:.4f}"
    )
    print(
        f"[ConPlag v1 shift - strip=True]  v2_F1={lgbm_v1_s['v2_oof_f1']:.4f} -> "
        f"v1_F1={lgbm_v1_s['metrics']['f1']:.4f} (shift={lgbm_v1_s['f1_shift']:+.4f})"
    )
    print(
        f"[ConPlag v1 shift - strip=False] v2_F1={lgbm_v1_u['v2_oof_f1']:.4f} -> "
        f"v1_F1={lgbm_v1_u['metrics']['f1']:.4f} (shift={lgbm_v1_u['f1_shift']:+.4f})"
    )
    print(
        f"[IR-Plag zero-shot @ thr]     F1={lgbm_ir['at_threshold']['f1']:.4f}, "
        f"AUC={lgbm_ir['at_threshold']['roc_auc']:.4f}, FPR={lgbm_ir['at_threshold']['fpr']:.4f}"
    )
    print(
        f"[IR-Plag zero-shot @ low_fpr] F1={lgbm_ir['at_threshold_low_fpr']['f1']:.4f}, "
        f"FPR={lgbm_ir['at_threshold_low_fpr']['fpr']:.4f}, "
        f"levels={lgbm_ir['per_level_at_threshold_low_fpr']}"
    )


if __name__ == "__main__":
    main()
