"""Training pipeline for calibrated LightGBM plagiarism detector and TF-IDF baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.markov import pair_to_tensor
from src.models.baseline import TfidfCosineBaseline, transition_perplexity
from src.models.classifier import PairwisePlagiarismClassifier
from src.preprocessing.normalizer import NUM_STATES, normalize_source
from src.preprocessing.parser import parse_to_state_sequence

DEFAULT_CONPLAG_V2_MANIFEST = Path("dataset/preprocessed/conplag_v2_manifest.parquet")
DEFAULT_MODELS_DIR = Path("models")
MODEL_VERSION = "0.1.0"
DEFAULT_SMOOTHING_ALPHA = 0.01
MAX_BUNDLE_BYTES = 1_048_576  # 1 MB


@dataclass(frozen=True)
class PreparedDataset:
    """Extracted Markov tensors, perplexities, normalized texts, and metadata for a manifest."""

    X: np.ndarray
    y: np.ndarray
    perp_a: np.ndarray
    perp_b: np.ndarray
    codes_a: list[str]
    codes_b: list[str]
    fold_ids: np.ndarray
    group_ids: np.ndarray
    levels: list[str]


def prepare_manifest_dataset(
    manifest_path: Path,
    language: str = "java",
    strip_boilerplate: bool = True,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> PreparedDataset:
    """Parse a parquet manifest into (N, 3, K, K) tensors, perplexities, and normalized texts."""
    df = pl.read_parquet(manifest_path)
    codes_a_raw = df["code_a"].to_list()
    codes_b_raw = df["code_b"].to_list()

    tensors: list[np.ndarray] = []
    perp_a_list: list[float] = []
    perp_b_list: list[float] = []
    norm_a_list: list[str] = []
    norm_b_list: list[str] = []

    for code_a, code_b in zip(codes_a_raw, codes_b_raw):
        norm_a = normalize_source(code_a, language=language, strip_boilerplate=strip_boilerplate)
        norm_b = normalize_source(code_b, language=language, strip_boilerplate=strip_boilerplate)
        norm_a_list.append(norm_a)
        norm_b_list.append(norm_b)

        res_a = parse_to_state_sequence(
            code_a,
            language=language,
            strip_boilerplate=strip_boilerplate,
        )
        res_b = parse_to_state_sequence(
            code_b,
            language=language,
            strip_boilerplate=strip_boilerplate,
        )
        tensor = pair_to_tensor(
            res_a.states,
            res_b.states,
            num_states=NUM_STATES,
            smoothing_alpha=smoothing_alpha,
        )
        tensors.append(tensor)
        perp_a_list.append(
            transition_perplexity(res_a.states, tensor[0], smoothing_alpha=smoothing_alpha)
        )
        perp_b_list.append(
            transition_perplexity(res_b.states, tensor[1], smoothing_alpha=smoothing_alpha)
        )

    X = np.stack(tensors, axis=0).astype(np.float32)
    y = df["label"].to_numpy().astype(np.uint8, copy=False)
    fold_ids = df["fold_id"].to_numpy().astype(np.int32, copy=False)
    group_ids = np.asarray(df["group_id"].to_list(), dtype=object)
    levels = [str(lvl) for lvl in df["level"].to_list()]

    return PreparedDataset(
        X=X,
        y=y,
        perp_a=np.asarray(perp_a_list, dtype=np.float64),
        perp_b=np.asarray(perp_b_list, dtype=np.float64),
        codes_a=norm_a_list,
        codes_b=norm_b_list,
        fold_ids=fold_ids,
        group_ids=group_ids,
        levels=levels,
    )


def find_f1_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold maximizing binary F1 score."""
    y_arr = np.asarray(y_true, dtype=np.uint8).ravel()
    s_arr = np.asarray(scores, dtype=np.float64).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_arr, s_arr)
    if thresholds.size == 0:
        return 0.5
    p_vals = precisions[:-1]
    r_vals = recalls[:-1]
    denom = p_vals + r_vals
    f1_vals = np.divide(2.0 * p_vals * r_vals, denom, out=np.zeros_like(denom), where=denom > 0.0)
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx])


def find_low_fpr_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Return the smallest threshold achieving FPR <= max_fpr."""
    y_arr = np.asarray(y_true, dtype=np.uint8).ravel()
    s_arr = np.asarray(scores, dtype=np.float64).ravel()
    neg_scores = s_arr[y_arr == 0]
    if neg_scores.size == 0:
        return 0.5

    candidates = np.unique(s_arr)
    n_neg = float(neg_scores.size)
    for thresh in candidates:
        fpr = float(np.sum(neg_scores >= thresh)) / n_neg
        if fpr <= max_fpr:
            return float(thresh)
    return float(np.max(s_arr) + 1e-6)


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Compute precision, recall, F1, ROC AUC, FPR, and confusion matrix at threshold."""
    y_arr = np.asarray(y_true, dtype=np.uint8).ravel()
    s_arr = np.asarray(scores, dtype=np.float64).ravel()
    y_pred = (s_arr >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_arr, s_arr)) if np.unique(y_arr).size == 2 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": fpr,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_oof_cv(
    dataset: PreparedDataset,
    classifier_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run 5-fold problem-disjoint CV using manifest fold_id and return OOF probabilities."""
    kwargs = dict(classifier_kwargs or {})
    unique_folds = sorted(int(f) for f in np.unique(dataset.fold_ids))
    oof_probs = np.zeros(dataset.y.shape[0], dtype=np.float64)
    fold_records: list[dict[str, float]] = []

    for fold in unique_folds:
        val_mask = dataset.fold_ids == fold
        train_mask = ~val_mask

        clf = PairwisePlagiarismClassifier(**kwargs)
        clf.fit(
            dataset.X[train_mask],
            dataset.y[train_mask],
            perplexities=(dataset.perp_a[train_mask], dataset.perp_b[train_mask]),
            groups=dataset.group_ids[train_mask],
        )
        val_probs = clf.predict_proba(
            dataset.X[val_mask],
            perplexities=(dataset.perp_a[val_mask], dataset.perp_b[val_mask]),
        )[:, 1]
        oof_probs[val_mask] = val_probs

        # Evaluate per-fold ROC AUC and thresholded metrics calibrated on train fold.
        train_probs = clf.predict_proba(
            dataset.X[train_mask],
            perplexities=(dataset.perp_a[train_mask], dataset.perp_b[train_mask]),
        )[:, 1]
        fold_thresh = find_f1_optimal_threshold(dataset.y[train_mask], train_probs)
        m = compute_binary_metrics(dataset.y[val_mask], val_probs, threshold=fold_thresh)
        fold_records.append(
            {
                "fold": float(fold),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "roc_auc": float(m["roc_auc"]),
                "fpr": float(m["fpr"]),
            }
        )

    opt_threshold = find_f1_optimal_threshold(dataset.y, oof_probs)
    low_fpr_threshold = find_low_fpr_threshold(dataset.y, oof_probs, max_fpr=0.05)

    micro_opt = compute_binary_metrics(dataset.y, oof_probs, threshold=opt_threshold)
    micro_low_fpr = compute_binary_metrics(dataset.y, oof_probs, threshold=low_fpr_threshold)

    macro_summary: dict[str, dict[str, float]] = {}
    for key in ("precision", "recall", "f1", "roc_auc", "fpr"):
        vals = np.asarray([r[key] for r in fold_records], dtype=np.float64)
        macro_summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
        }

    cv_metrics: dict[str, object] = {
        "n_splits": len(unique_folds),
        "n_samples": int(dataset.y.shape[0]),
        "micro_oof": micro_opt,
        "micro_oof_low_fpr": micro_low_fpr,
        "macro_cv": macro_summary,
        "per_fold": fold_records,
    }
    return oof_probs, cv_metrics


def train_and_save_bundle(
    manifest_path: Path = DEFAULT_CONPLAG_V2_MANIFEST,
    models_dir: Path = DEFAULT_MODELS_DIR,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    strip_boilerplate: bool = True,
) -> dict[str, object]:
    """Train final models on ConPlag v2, calibrate thresholds via 5-fold CV, and save bundle."""
    dataset = prepare_manifest_dataset(
        manifest_path=manifest_path,
        language="java",
        strip_boilerplate=strip_boilerplate,
        smoothing_alpha=smoothing_alpha,
    )

    _, cv_metrics = run_oof_cv(dataset)
    threshold = float(cv_metrics["micro_oof"]["threshold"])  # type: ignore[index]
    threshold_low_fpr = float(cv_metrics["micro_oof_low_fpr"]["threshold"])  # type: ignore[index]

    final_clf = PairwisePlagiarismClassifier()
    final_clf.fit(
        dataset.X,
        dataset.y,
        perplexities=(dataset.perp_a, dataset.perp_b),
        groups=dataset.group_ids,
    )

    final_tfidf = TfidfCosineBaseline(
        sublinear_tf=True,
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=20000,
    )
    final_tfidf.fit(dataset.codes_a + dataset.codes_b)

    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = models_dir / "plagiarism_lgbm.joblib"
    tfidf_path = models_dir / "tfidf_baseline.joblib"
    meta_path = models_dir / "model_metadata.json"

    joblib.dump(final_clf, lgbm_path, compress=3)
    joblib.dump(final_tfidf, tfidf_path, compress=3)

    metadata: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "threshold": threshold,
        "threshold_low_fpr": threshold_low_fpr,
        "smoothing_alpha": float(smoothing_alpha),
        "num_states": int(NUM_STATES),
        "strip_boilerplate": bool(strip_boilerplate),
        "supported_languages": ["java"],
        "cv_metrics": cv_metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    sizes = {
        lgbm_path.name: lgbm_path.stat().st_size,
        tfidf_path.name: tfidf_path.stat().st_size,
        meta_path.name: meta_path.stat().st_size,
    }
    total_bytes = sum(sizes.values())
    if total_bytes >= MAX_BUNDLE_BYTES:
        raise RuntimeError(
            f"Serialized model bundle exceeds 1 MB limit: {total_bytes} bytes ({sizes})"
        )

    return {
        "metadata": metadata,
        "bundle_sizes_bytes": sizes,
        "total_bundle_bytes": total_bytes,
    }


def main() -> None:
    """CLI entrypoint to train and serialize the production model bundle."""
    result = train_and_save_bundle()
    meta = result["metadata"]
    cv = meta["cv_metrics"]  # type: ignore[index]
    micro = cv["micro_oof"]  # type: ignore[index]
    low_fpr = cv["micro_oof_low_fpr"]  # type: ignore[index]
    macro = cv["macro_cv"]  # type: ignore[index]

    print(f"Saved model bundle ({result['total_bundle_bytes']} bytes): {result['bundle_sizes_bytes']}")
    print(
        f"ConPlag v2 5-fold OOF (threshold={meta['threshold']:.4f}): "  # type: ignore[index]
        f"F1={micro['f1']:.4f}, P={micro['precision']:.4f}, R={micro['recall']:.4f}, "
        f"AUC={micro['roc_auc']:.4f}, FPR={micro['fpr']:.4f}"
    )
    print(
        f"ConPlag v2 5-fold OOF Low-FPR (threshold={meta['threshold_low_fpr']:.4f}): "  # type: ignore[index]
        f"F1={low_fpr['f1']:.4f}, P={low_fpr['precision']:.4f}, R={low_fpr['recall']:.4f}, "
        f"FPR={low_fpr['fpr']:.4f}"
    )
    print(
        f"ConPlag v2 Macro CV: "
        f"F1={macro['f1']['mean']:.4f} +/- {macro['f1']['std']:.4f}, "
        f"AUC={macro['roc_auc']['mean']:.4f} +/- {macro['roc_auc']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
