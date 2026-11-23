import numpy as np

INPUT_SPEC = {
    "normalization": "filas_normalizadas",
    "state_order": None,   # lista de nombres de categorías AST cuando se defina
}

N_MATRIX_CHANNELS = 2   # canales 0 y 1 = matrices de transición (programa A y B); el resto son derivados (A - B)

def validate_transition_matrices(P: np.ndarray, atol: float = 1e-4) -> None:
    """Verifica que `P` (..., K, K) tenga valores en [0, 1] y filas que sumen 1 (o 0 si no hay transiciones)."""
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise ValueError("Las matrices contienen NaN o infinitos")
    if P.min() < -atol or P.max() > 1 + atol:
        raise ValueError(f"Valores fuera de [0, 1]: min={P.min():.4f}, max={P.max():.4f}")
    sums = P.sum(axis=-1)
    ok = (np.abs(sums - 1.0) <= atol) | (np.abs(sums) <= atol)
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} filas no suman 0 ni 1 (suma fuera de tolerancia)")

def validate_features(X: np.ndarray, atol: float = 1e-4) -> None:
    """Valida un tensor (N, C, K, K): 4 dimensiones, matrices cuadradas y cada canal normalizado por filas."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"X debe tener 4 dimensiones (N, C, K, K), se recibió {X.shape}")
    if X.shape[2] != X.shape[3]:
        raise ValueError(f"Las matrices deben ser cuadradas, se recibió {X.shape[2:]}")
    for c in range(min(X.shape[1], N_MATRIX_CHANNELS)):
        validate_transition_matrices(X[:, c], atol=atol)
    if X.shape[1] > N_MATRIX_CHANNELS:   # canales derivados: solo deben ser finitos y estar en [-1, 1]
        extra = np.asarray(X[:, N_MATRIX_CHANNELS:], dtype=np.float64)
        if not np.isfinite(extra).all() or extra.min() < -1 - atol or extra.max() > 1 + atol:
            raise ValueError("Los canales derivados deben ser finitos y estar en [-1, 1]")