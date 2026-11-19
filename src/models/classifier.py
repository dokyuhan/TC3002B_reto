"""Permutation-symmetric calibrated LightGBM classifier for pairwise Markov tensors."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _row_wise_jsd_batch(
    p_a: np.ndarray,
    p_b: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Vectorized mean row-wise base-2 JSD for batches of (N, K, K) matrices."""
    mat_a = np.clip(p_a, 0.0, None)
    mat_b = np.clip(p_b, 0.0, None)

    both_zero = (mat_a.sum(axis=-1) <= 0.0) & (mat_b.sum(axis=-1) <= 0.0)

    norm_a = mat_a + eps
    norm_b = mat_b + eps
    norm_a = norm_a / norm_a.sum(axis=-1, keepdims=True)
    norm_b = norm_b / norm_b.sum(axis=-1, keepdims=True)

    mid = 0.5 * (norm_a + norm_b)
    kl_a = np.sum(norm_a * np.log2(norm_a / mid), axis=-1)
    kl_b = np.sum(norm_b * np.log2(norm_b / mid), axis=-1)
    row_jsd = np.where(both_zero, 0.0, 0.5 * (kl_a + kl_b))
    return np.maximum(0.0, np.mean(row_jsd, axis=-1))


def _mean_row_entropy_batch(p_mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute mean base-2 Shannon entropy across rows for (N, K, K) matrices."""
    mat = np.clip(p_mat, 0.0, None)
    row_sums = mat.sum(axis=-1, keepdims=True)
    zero_rows = (row_sums <= 0.0).squeeze(-1)

    norm = mat + eps
    norm = norm / norm.sum(axis=-1, keepdims=True)
    row_entropy = -np.sum(norm * np.log2(norm), axis=-1)
    row_entropy = np.where(zero_rows, 0.0, row_entropy)
    return np.maximum(0.0, np.mean(row_entropy, axis=-1))


class PairwisePlagiarismClassifier:
    """Permutation-symmetric calibrated LightGBM classifier on (N, 3, K, K) Markov tensors.

    Feature extraction produces a fixed-width symmetric vector of size 3 * K^2 + 8:
    - 0.5 * (P_A + P_B) flattened (K^2)
    - |P_A - P_B| flattened (K^2)
    - (P_A - P_B)^2 flattened (K^2)
    - Frobenius distance ||P_A - P_B||_F (1)
    - Mean row-wise base-2 Jensen-Shannon divergence (1)
    - Active state support sum and |diff| (2)
    - Mean row entropy average and |diff| (2)
    - Sequence transition perplexity average and |diff| (2; falls back to 2**entropy when omitted)

    Inference enforces exact permutation symmetry by averaging p(A, B) and p(B, A).
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 5,
        num_leaves: int = 23,
        min_child_samples: int = 15,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: str | dict[int, float] | None = "balanced",
        calibration_method: str = "sigmoid",
        calibration_cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.random_state = random_state

        self._model: CalibratedClassifierCV | None = None
        self._num_states: int | None = None
        self._n_features: int | None = None

    def extract_features(
        self,
        X: np.ndarray,
        perplexities: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Transform (N, 3, K, K) tensors and optional perplexities into symmetric tabular features."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 4 or arr.shape[1] != 3 or arr.shape[2] != arr.shape[3]:
            raise ValueError(f"Expected X with shape (N, 3, K, K), got {arr.shape}")

        n_samples, _, num_states, _ = arr.shape
        if n_samples == 0:
            return np.empty((0, 3 * num_states * num_states + 8), dtype=np.float32)

        p_a = arr[:, 0, :, :]
        p_b = arr[:, 1, :, :]
        diff = p_a - p_b

        mean_ab = (0.5 * (p_a + p_b)).reshape(n_samples, -1)
        abs_diff = np.abs(diff).reshape(n_samples, -1)
        sq_diff = (diff * diff).reshape(n_samples, -1)

        frob = np.sqrt(np.sum(diff * diff, axis=(1, 2)))[:, None]
        jsd = _row_wise_jsd_batch(p_a, p_b)[:, None]

        uniform_val = 1.0 / num_states
        active_a = np.sum(
            (p_a.sum(axis=2) > 1e-6) & (np.max(np.abs(p_a - uniform_val), axis=2) > 1e-5),
            axis=1,
        ).astype(np.float64)
        active_b = np.sum(
            (p_b.sum(axis=2) > 1e-6) & (np.max(np.abs(p_b - uniform_val), axis=2) > 1e-5),
            axis=1,
        ).astype(np.float64)
        support_sum = (active_a + active_b)[:, None]
        support_diff = np.abs(active_a - active_b)[:, None]

        ent_a = _mean_row_entropy_batch(p_a)
        ent_b = _mean_row_entropy_batch(p_b)
        ent_mean = (0.5 * (ent_a + ent_b))[:, None]
        ent_diff = np.abs(ent_a - ent_b)[:, None]

        if perplexities is not None:
            perp_a = np.asarray(perplexities[0], dtype=np.float64).ravel()
            perp_b = np.asarray(perplexities[1], dtype=np.float64).ravel()
            if perp_a.shape[0] != n_samples or perp_b.shape[0] != n_samples:
                raise ValueError(
                    f"Perplexity lengths ({perp_a.shape[0]}, {perp_b.shape[0]}) "
                    f"must match N={n_samples}"
                )
        else:
            perp_a = np.power(2.0, ent_a)
            perp_b = np.power(2.0, ent_b)

        perp_mean = (0.5 * (perp_a + perp_b))[:, None]
        perp_diff = np.abs(perp_a - perp_b)[:, None]

        features = np.concatenate(
            [
                mean_ab,
                abs_diff,
                sq_diff,
                frob,
                jsd,
                support_sum,
                support_diff,
                ent_mean,
                ent_diff,
                perp_mean,
                perp_diff,
            ],
            axis=1,
        )
        return features.astype(np.float32, copy=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        perplexities: tuple[np.ndarray, np.ndarray] | None = None,
        groups: np.ndarray | None = None,
    ) -> PairwisePlagiarismClassifier:
        """Fit calibrated LightGBM classifier on symmetric pairwise Markov features."""
        feats = self.extract_features(X, perplexities=perplexities)
        y_arr = np.asarray(y, dtype=np.int32).ravel()
        if feats.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Mismatched sample counts: X={feats.shape[0]}, y={y_arr.shape[0]}")

        _, class_counts = np.unique(y_arr, return_counts=True)
        if class_counts.size < 2:
            raise ValueError("Training labels must contain at least 2 classes.")

        min_class_count = int(class_counts.min())
        n_splits = max(2, min(self.calibration_cv, min_class_count))

        base_estimator = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=min(self.min_child_samples, max(2, feats.shape[0] // 4)),
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=1,
        )

        cv_splits: list[tuple[np.ndarray, np.ndarray]]
        if groups is not None and len(set(groups)) >= n_splits:
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            cv_splits = list(sgkf.split(feats, y_arr, groups=groups))
        else:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            cv_splits = list(skf.split(feats, y_arr))

        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.calibration_method,
            cv=cv_splits,
        )
        calibrated.fit(feats, y_arr)

        self._model = calibrated
        self._num_states = int(np.asarray(X).shape[2])
        self._n_features = int(feats.shape[1])
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        perplexities: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Predict permutation-symmetric class probabilities of shape (N, 2)."""
        if self._model is None:
            raise RuntimeError("PairwisePlagiarismClassifier must be fitted before prediction.")

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1] != 3 or arr.shape[2] != arr.shape[3]:
            raise ValueError(f"Expected X with shape (N, 3, K, K), got {arr.shape}")
        if arr.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        arr_swapped = arr[:, [1, 0, 2], :, :]
        perp_swapped = (perplexities[1], perplexities[0]) if perplexities is not None else None

        feats_ab = self.extract_features(arr, perplexities=perplexities)
        feats_ba = self.extract_features(arr_swapped, perplexities=perp_swapped)

        proba_ab = self._model.predict_proba(feats_ab)
        proba_ba = self._model.predict_proba(feats_ba)
        return np.clip(0.5 * (proba_ab + proba_ba), 0.0, 1.0)

    def predict(
        self,
        X: np.ndarray,
        perplexities: tuple[np.ndarray, np.ndarray] | None = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary plagiarism labels using the positive-class probability threshold."""
        proba = self.predict_proba(X, perplexities=perplexities)
        return (proba[:, 1] >= threshold).astype(np.uint8)
