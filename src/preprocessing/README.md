# Data Preprocessing (Step 1)

Converts raw Java source pairs in `dataset/raw/` into problem disjoint `.parquet` manifests and offset packed `.npz` 12 state AST sequence archives under `dataset/preprocessed/`.

## Run

From the repository root:

```bash
uv run python -m src.preprocessing.build_datasets
```

## Outputs (`dataset/preprocessed/`)

* `conplag_v2_manifest.parquet` and `conplag_v2_sequences.npz`: 911 Java pairs across 21 Codeforces problems (template free), split into `train` (562), `val` (168), and `test` (181) with 0 problem or submission overlap across splits.
* `conplag_v1_manifest.parquet` and `conplag_v1_sequences.npz`: 911 raw Java pairs for boilerplate robustness evaluation.
* `ir_plag_manifest.parquet` and `ir_plag_sequences.npz`: 460 Java pairs across 7 cases (`L1` through `L6` and `NON_PLAG`) under the `diagnostic` split.

## Downstream Usage (Training And Inference)

Load precomputed AST sequences from `.npz`:

```python
import numpy as np

with np.load("dataset/preprocessed/conplag_v2_sequences.npz", allow_pickle=False) as data:
    tokens_a, offsets_a = data["train_tokens_a"], data["train_offsets_a"]
    tokens_b, offsets_b = data["train_tokens_b"], data["train_offsets_b"]
    y_train = data["train_y"]
    seq_a_0 = tokens_a[offsets_a[0] : offsets_a[1]]
    seq_b_0 = tokens_b[offsets_b[0] : offsets_b[1]]
```

Parse new source code directly at inference time:

```python
from src.preprocessing import DEFAULT_AST_STATES, NUM_STATES, normalize_source, parse_to_state_sequence

result = parse_to_state_sequence(source_code, language="java", strip_boilerplate=True)
states = result.states  # 1D int32 array in [0, 11]
```

## Schema Reference

* `.parquet` columns: `pair_id`, `group_id`, `fold_id`, `split`, `level`, `label`, `code_a`, `code_b`.
* `.npz` keys per split (`train`, `val`, `test` for ConPlag; `diagnostic` for IR Plag): `{split}_tokens_a`, `{split}_offsets_a`, `{split}_tokens_b`, `{split}_offsets_b`, `{split}_y`, `{split}_pair_ids`, `{split}_group_ids`, `{split}_fold_ids`, `{split}_levels`, plus `num_states` (`12`) and `state_names`.
