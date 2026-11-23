"""La CNN de torch (entrenamiento) y la de numpy (app) deben dar la misma salida con los mismos pesos.

Requiere torch; se omite si no está instalado (el CI de la app no lo instala).
Ejecutar desde la raíz:  python -m pytest model_training/tests
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model_training.export_numpy import export_weights  # noqa: E402
from model_training.network.cnn_torch import PlagiarismCNN  # noqa: E402
from src.models.cnn_numpy import NumpyCnnClassifier  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numpy_forward_matches_torch(tmp_path, seed):
    """Con pesos aleatorios y entradas aleatorias, los logits de numpy coinciden con los de torch."""
    torch.manual_seed(seed)
    model = PlagiarismCNN(in_channels=3, n_states=12).eval()
    path = tmp_path / "cnn_weights.npz"
    export_weights(model.state_dict(), model.config, path)

    x = np.random.default_rng(seed).random((16, 3, 12, 12), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = NumpyCnnClassifier.load(path).predict_logits(x)
    assert np.allclose(actual, expected, atol=1e-4)
