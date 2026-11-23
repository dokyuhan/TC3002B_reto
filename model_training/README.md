# model_training: how the CNN is built, trained, tested and exported

This folder holds everything needed to **train and test** the plagiarism CNN with PyTorch. The app does not use this code at runtime: it only uses the weights that `export_numpy.py` copies into `../models/`.

Run every command from the **repository root**, with `uv run --group train` (this installs PyTorch the first time).

```bash
uv run --group train python -m model_training.main build-data     # 1. AST sequences -> matrices
uv run --group train python -m model_training.main train          # 2. train, keep the best epoch
uv run --group train python -m model_training.main evaluate       # 3. test on ConPlag test
uv run --group train python -m model_training.main evaluate --ir-plag   # 3b. test on unseen IR Plag
uv run --group train python -m model_training.export_numpy        # 4. copy the weights into the app
uv run --group train python -m model_training.baseline            # (optional) logistic baseline
uv run --group train python -m pytest model_training/tests        # (optional) torch vs numpy check
```

## 1. The idea

Two Java programs are compared through the **structure** of their code, not their text.

* Each program becomes a sequence of 12 AST states (`IMPORT`, `FUNCTION`, `CLASS`, `ASSIGN`, `IF`, `FOR`, `WHILE`, `CALL`, `RETURN`, `EXCEPTION`, `OP`, `OTHER`). This step is done by `src/preprocessing` and `scripts/build_datasets.py`.
* The sequence becomes a **12 x 12 Markov transition matrix**: cell `(i, j)` is the probability that state `j` follows state `i`. Each row sums to 1, or is all zeros if the state never appears.
* A pair of programs becomes a tensor with **3 channels**: `P_A`, `P_B` and `P_A - P_B` (signed difference). Shape `(3, 12, 12)`.
* The CNN reads that tensor and outputs one number, the probability that the pair is plagiarized.

**Labels.** `y = 1` means the pair is plagiarized, `y = 0` means it is not. The labels come from ConPlag verdicts and from the IR Plag folders (`plagiarized/` and `non-plagiarized/`). They do not say whether code was written by an AI.

**Why a CNN on a transition matrix?** It is a hypothesis, not a fact. Matrix positions are AST categories, not neighbouring pixels, so convolution is not a natural fit. The logistic regression baseline (`baseline.py`) exists to show whether the CNN learns something a linear model cannot.

## 2. The process, step by step

```
dataset/preprocessed/conplag_v2_sequences.npz     AST state sequences (from scripts/build_datasets.py)
dataset/preprocessed/ir_plag_sequences.npz
        |  main.py build-data   (data/build_matrices.py)
        v
dataset/processed/{train,val,test,ir_plag}.npz    X (N,3,12,12), y, group   [generated, not committed]
        |  main.py train        (training/trainer.py)
        v
model_training/checkpoints/best.pt                best weights + saved threshold
dataset/results/history.json                      loss and metrics per epoch
        |  main.py evaluate     (training/metrics.py)
        v
dataset/results/test_metrics.json, ir_plag_metrics.json
        |  export_numpy.py
        v
models/cnn_weights.npz + models/model_metadata.json   what the app loads
```

**Data splits.**

| File | Source | Used for |
|---|---|---|
| `train.npz` (562 pairs) | ConPlag v2 | Learning the weights |
| `val.npz` (168 pairs) | ConPlag v2 | Choosing the best epoch (early stopping) and the thresholds |
| `test.npz` (181 pairs) | ConPlag v2 | Final score on held-out data |
| `ir_plag.npz` (460 pairs) | IR Plag | Final score on a dataset the model never saw; never used for training |

The ConPlag splits are separated **by programming problem**, so the same problem never appears in two splits. Splitting by pair would leak information and inflate the scores.

**Training (`main.py train`).**

1. Fix the random seed and choose the device (`cuda`, then `mps`, then `cpu`).
2. Load `train.npz` and `val.npz`, and read the number of channels and states from the data.
3. Build the CNN and train it with Adam and a binary cross-entropy loss.
4. After each epoch, compute loss, accuracy and F1 on train and validation. Whenever the validation loss improves, save a checkpoint.
5. Stop when the validation loss has not improved for `PATIENCE` epochs, and restore the best weights.
6. Pick a threshold on the validation set (the one with the best F1, ties broken toward 0.5) and save it in the checkpoint.
7. Write the per-epoch history to `dataset/results/history.json`.

**Evaluation (`main.py evaluate`).** Loads the checkpoint, predicts on the chosen `.npz`, and computes metrics at the threshold in `config.THRESHOLD` (0.5). For IR Plag it also prints accuracy per level (`L1` to `L6` are plagiarism levels, `NON_PLAG` is non-plagiarized code).

**Export (`export_numpy.py`).** The app runs the network with numpy, so it does not need PyTorch. This script copies the weights, checks that numpy and PyTorch give the same output, evaluates the version the app uses, and writes the thresholds and metrics to the metadata file.

## 3. The network

Defined in `network/cnn_torch.py`. With the default configuration (`CONV_CHANNELS = (16, 32)`), the shapes for one pair are:

```
input                    (3, 12, 12)
Conv 3x3, pad 1 + ReLU   (16, 12, 12)
MaxPool 2x2              (16, 6, 6)
Conv 3x3, pad 1 + ReLU   (32, 6, 6)
AdaptiveAvgPool 4x4      (32, 4, 4)
Flatten                  512
Dropout + Linear + ReLU  64
Dropout + Linear         1      <- logit; sigmoid gives P(plagiarism)
```

That is about 38,000 trainable parameters. Training uses 562 pairs, so the model is deliberately small.

## 4. Configuration (`config.py`)

Everything you tune is in this one file. Paths are computed from the repository root, so the commands work from any folder.

| Setting | Value | Meaning |
|---|---|---|
| `EPOCHS` | 100 | Maximum number of epochs |
| `BATCH_SIZE` | 32 | Samples per update |
| `LEARNING_RATE` | 1e-3 | Adam step size |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `PATIENCE` | 15 | Epochs without improvement before stopping (`None` disables early stopping) |
| `MIN_DELTA` | 1e-3 | Minimum validation-loss drop that counts as an improvement |
| `SEED` | 42 | Seed for reproducible runs |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `CONV_CHANNELS` | (16, 32) | Filters in the two convolution layers |
| `HIDDEN_DIM` | 64 | Units in the hidden linear layer |
| `DROPOUT` | 0.4 | Dropout probability |
| `POOL_SIZE` | 4 | Output size of the adaptive pooling |
| `THRESHOLD` | 0.5 | Decision threshold at evaluation (`None` uses the one saved in the checkpoint) |

If you change the architecture settings, retrain and run `export_numpy.py` again. The saved `best.pt` only works with the architecture it was trained with.

## 5. Function reference

### `main.py`: command line entry point

| Function | What it does |
|---|---|
| `build_data()` | Converts the AST sequences in `dataset/preprocessed` into the matrices in `dataset/processed` (train, val, test and IR Plag). |
| `train()` | Runs the full training described above and saves `best.pt` and `history.json`. |
| `evaluate(data_path, out_path)` | Evaluates the checkpoint on a `.npz`, prints the metrics, adds the per-level breakdown when the file has `levels`, and saves a JSON. Options: `--data`, `--out`, `--ir-plag`. |
| `predict(index)` | Prints the probability and label for the examples of `PREDICT_INPUT_PATH`, or for one example with `--index`. |
| `main()` | Parses the command (`build-data`, `train`, `evaluate`, `predict`) and calls the right function. |

### `config.py`

Constants only (paths and hyperparameters, see section 4). `ROOT` is the repository root.

### `data/build_matrices.py`: sequences to matrices

| Function | What it does |
|---|---|
| `_matrices(tokens, offsets, n_states)` | The sequences are stored as one long array plus offsets. This cuts the array into programs and builds one transition matrix per program, using `src.features.markov` with no smoothing. Returns `(N, K, K)`. |
| `build_split(npz, split)` | Builds the three channels (A, B, A - B) for one split and returns `X`, `y` and `group`. |
| `convert(sequences_path, out_dir)` | Writes `train.npz`, `val.npz` and `test.npz` from a sequences archive. |
| `convert_diagnostic(sequences_path, out_path)` | Writes the single IR Plag file, including the `levels` column used for the per-level report. |

### `data/features.py`: input checks

| Name | What it does |
|---|---|
| `INPUT_SPEC` | Records how the inputs were built (rows normalized). It is saved inside the checkpoint. |
| `N_MATRIX_CHANNELS` | Number of true transition-matrix channels (2). Extra channels are derived. |
| `validate_transition_matrices(P)` | Checks values are finite, in [0, 1], and each row sums to 1 (or 0 if empty). |
| `validate_features(X)` | Checks `X` is `(N, C, K, K)` with square matrices. The first two channels must be transition matrices. Derived channels (A - B) only need to be finite and within [-1, 1]. |

### `data/data_loader.py`: loading

| Function | What it does |
|---|---|
| `_load_npz(path)` | Reads a `.npz` and requires the keys `X` and `y` (`group` is optional). |
| `load_split(path)` | Loads and validates one split: checks the file, the shape of `X`, that `y` has only 0 and 1, and converts to `float32` and `int64`. |
| `get_data_info(X)` | Returns the number of channels and of states, read from the shape of `X`. |
| `make_dataloader(data, batch_size, shuffle, seed)` | Wraps the data in a PyTorch `DataLoader`. With a seed, the shuffling is reproducible. |

### `data/dataset.py`

| Name | What it does |
|---|---|
| `PlagiarismDataset` | A PyTorch `Dataset` holding `X` and `y` as `float32` tensors. `__len__` returns the number of samples and `__getitem__` returns one `(matrices, label)` pair. |

### `network/cnn_torch.py`: the model

| Name | What it does |
|---|---|
| `AdaptiveAvgPool2dMPS` | An adaptive average pooling that falls back to the CPU on Apple GPUs when the size is not divisible, a case MPS does not support. |
| `PlagiarismCNN` | The network of section 3. `forward(x)` takes `(B, C, K, K)` and returns one logit per sample. `from_config(config)` rebuilds a model from the saved configuration. |

### `network/checkpoints.py`: saving and loading

| Function | What it does |
|---|---|
| `_plain(obj)` | Forces plain Python types (through JSON), so checkpoints never contain numpy objects. |
| `save_checkpoint(path, model, threshold, extra)` | Saves the weights, the model configuration, the threshold and extra data (best epoch, validation loss, input spec). |
| `load_checkpoint(path, device)` | Rebuilds the model in evaluation mode and returns it with the checkpoint dictionary. |

### `network/predictor.py`: using a trained model

| Name | What it does |
|---|---|
| `Predictor(checkpoint_path, device, threshold)` | Loads a checkpoint. Uses the saved threshold unless you pass one (must be between 0 and 1). |
| `Predictor._prepare(X)` | Accepts one example `(C, K, K)` or a batch, validates it, and checks it matches the model. |
| `Predictor.predict_proba(X)` | Returns the probability of plagiarism for each sample. |
| `Predictor.predict(X)` | Returns the probabilities and the 0/1 labels at the threshold. |
| `predict(X, checkpoint_path, ...)` | A one-line shortcut that loads the model and predicts. |

### `training/trainer.py`: the training loop

| Name | What it does |
|---|---|
| `TrainConfig` | The training hyperparameters in one object. |
| `EarlyStopping` | `step(value)` records the validation loss and returns `True` if it improved. `should_stop` becomes `True` after `patience` epochs without improvement. |
| `train_one_epoch(...)` | One pass over the training data: forward, loss, backpropagation, weight update. Returns the average loss. |
| `run_inference(model, loader, device, criterion)` | Evaluates without updating weights. Returns the loss (if a criterion is given), the probabilities and the labels. |
| `fit(...)` | The full loop: trains epoch by epoch, records the history, saves a checkpoint on every improvement, stops early, and restores the best weights. Returns the history and the best epoch. |

### `training/metrics.py`: measuring results

| Function | What it does |
|---|---|
| `compute_metrics(y_true, probs, threshold)` | Accuracy, precision, recall, F1, false positive rate, PR-AUC and the confusion matrix `[[TN, FP], [FN, TP]]`. |
| `select_threshold(y_true, probs)` | Finds the threshold with the best F1 (ties go to the one closest to 0.5). Use it on validation only. |
| `format_summary(m, title)` | Turns the metrics dictionary into readable text for the terminal. |
| `save_metrics(path, metrics)` | Writes the metrics as JSON, creating the folder if needed. |

### `utils.py`

| Function | What it does |
|---|---|
| `set_seed(seed)` | Fixes the random seed of Python, numpy and PyTorch, so runs are reproducible. |
| `get_device(name)` | Chooses the device. `auto` prefers CUDA, then MPS, then CPU. |

### `baseline.py`: reference model

| Function | What it does |
|---|---|
| `flat(split)` | Flattens each `(3, 12, 12)` sample into a vector of 432 numbers. |
| `main()` | Trains a logistic regression on those vectors, picks its threshold on validation, evaluates on val, test and IR Plag, and saves `dataset/results/baseline_metrics.json`. |

The baseline uses the default settings and no tuning. It shows how much of the result is possible without a neural network.

### `export_numpy.py`: from training to the app

| Function | What it does |
|---|---|
| `export_weights(state, model_config, path)` | Saves the layer weights from the PyTorch model into `cnn_weights.npz`. |
| `low_fpr_threshold(y, probs)` | The lowest threshold, at or above 0.5, with a validation false positive rate of 5% or less. |
| `main()` | Exports the weights, checks that numpy and PyTorch match (stops if they differ by more than 1e-4), evaluates the version the app uses (probabilities averaged over `(A, B)` and `(B, A)`), and writes the thresholds and metrics to `models/model_metadata.json`. |

### `tests/test_numpy_parity.py`

Builds random PyTorch networks, exports them, and checks that the numpy version in `src/models/cnn_numpy.py` gives the same logits. It is skipped when PyTorch is not installed.

## 6. Why these choices

* **Early stopping on validation loss.** With 562 training pairs the model overfits quickly. Stopping at the best epoch is the cheapest protection.
* **The threshold is not chosen on test.** Choosing it on the test set would make the reported score too optimistic. Validation selects the saved threshold. The fixed `THRESHOLD = 0.5` in `config.py` is used for the reported results because the validation-chosen threshold changed a lot between runs.
* **A signed difference channel.** It gives the network the comparison between the two programs directly, instead of making it learn that comparison from two separate matrices.
* **Averaging over both orders in the app.** The network sees `(A, B)` differently from `(B, A)`, so the app averages both. A pair should score the same whichever program is called A.
* **Evaluation on IR Plag.** It is a different dataset, so its score estimates how the model behaves on data unlike the training set.
* **Same data, same seed.** `SEED` fixes the run, so the same configuration gives the same results.

## 7. Known limits

* The 12 x 12 matrix discards names, nesting and statement order, so hard plagiarism levels (`L5`, `L6`) are close to a coin flip.
* The model flags many non-plagiarized pairs in IR Plag. These programs solve the same problem, so their structure is similar.
* The validation set is small (168 pairs, 47 plagiarized), so validation metrics move a lot between runs and seeds.
* Only Java is supported.
