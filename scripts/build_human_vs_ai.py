"""Build the single-program human-vs-AI dataset: raw HumanVsAICode JSONL -> AST state sequences.

Every raw row is one Java function with one human version and up to three AI versions
(ChatGPT, DeepSeek-Coder, Qwen2.5-Coder). Each version becomes one labeled sample:
label 0 = human, label 1 = AI, and `sources` records who wrote it.

Run from the repository root:
    uv run python -m scripts.build_human_vs_ai [--n-functions 20000] [--seed 42]

Output: dataset/preprocessed/human_vs_ai_java_sequences.npz with, for each split
(`train`, `val`, `test`): `{split}_tokens`, `{split}_offsets` (all programs packed in one array),
`{split}_y`, `{split}_group_ids` (block of consecutive functions, never shared between splits),
`{split}_sources` and `{split}_function_ids`.

Fairness rules, applied identically to human and AI code:
* Comments and whitespace never reach the model: the AST states skip comment nodes.
* Samples that declare their own top-level class/interface/enum are dropped (3% of ChatGPT output
  does this; human methods never do, so it would be a shortcut).
* Methods whose body has no statements (only comments or nothing) are dropped, because AI stubs such
  as `// Implementation goes here` would otherwise be an easy tell.
* Samples with fewer than `--min-states` AST states are dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.preprocessing.filters import declares_top_level_type, has_empty_body
from src.preprocessing.parser import parse_to_state_sequence
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES

RAW_PATH = Path("dataset/raw/human_vs_ai_code/java_dataset.jsonl")
OUTPUT_PATH = Path("dataset/preprocessed/human_vs_ai_java_sequences.npz")
# column name -> (label, source name). The raw file calls DeepSeek's column `dsc_code`.
SOURCES = {"human_code": (0, "human"), "chatgpt_code": (1, "chatgpt"), "dsc_code": (1, "deepseek"), "qwen_code": (1, "qwen")}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def count_rows(path: Path) -> int:
    """Count the JSONL rows (one function each) without parsing them."""
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def choose_blocks(n_rows: int, n_functions: int, block_size: int, seed: int) -> dict[int, str]:
    """Pick random blocks of consecutive rows and assign each whole block to one split.

    The raw file has no repository column, but consecutive rows tend to come from the same
    repository. Keeping a block inside one split limits leakage between train and test.
    """
    n_blocks = -(-n_rows // block_size)
    wanted = min(n_blocks, -(-n_functions // block_size))
    order = np.random.default_rng(seed).permutation(n_blocks)[:wanted]
    n_train = int(round(wanted * SPLIT_FRACTIONS["train"]))
    n_val = int(round(wanted * SPLIT_FRACTIONS["val"]))
    split_of = {}
    for i, block in enumerate(order.tolist()):
        split_of[block] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return split_of


def program_states(code: str, min_states: int) -> tuple[np.ndarray | None, str]:
    """Parse one function into AST states; return (states, "") or (None, reason it was dropped)."""
    if not code or not code.strip():
        return None, "empty"
    if declares_top_level_type(code):
        return None, "declares_a_class"
    if has_empty_body(code):
        return None, "empty_body"
    # Raw rows are bare methods, so wrap them in a class to give tree-sitter a valid compilation unit.
    states = parse_to_state_sequence("class W {\n" + code + "\n}", language="java", strip_boilerplate=False).states
    if len(states) < min_states:
        return None, "too_few_states"
    return states.astype(np.int8), ""


def build(raw_path: Path, output_path: Path, n_functions: int, block_size: int, min_states: int, seed: int) -> None:
    """Parse the sampled functions and write the offset-packed archive."""
    n_rows = count_rows(raw_path)
    split_of = choose_blocks(n_rows, n_functions, block_size, seed)
    data = {s: {"seqs": [], "y": [], "group": [], "source": [], "function": []} for s in SPLIT_FRACTIONS}
    dropped: dict[str, int] = {}
    with raw_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            block = idx // block_size
            split = split_of.get(block)
            if split is None:
                continue
            row = json.loads(line)
            for column, (label, source) in SOURCES.items():
                states, reason = program_states(row.get(column) or "", min_states)
                if states is None:
                    key = f"{source}:{reason}"
                    dropped[key] = dropped.get(key, 0) + 1
                    continue
                d = data[split]
                d["seqs"].append(states)
                d["y"].append(label)
                d["group"].append(f"block{block}")
                d["source"].append(source)
                d["function"].append(str(row.get("hm_index", idx)))

    payload: dict[str, np.ndarray] = {"num_states": np.asarray(NUM_STATES, dtype=np.int32), "state_names": np.asarray(DEFAULT_AST_STATES)}
    for split, d in data.items():
        lengths = np.asarray([len(s) for s in d["seqs"]], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        payload[f"{split}_tokens"] = np.concatenate(d["seqs"]) if d["seqs"] else np.empty(0, dtype=np.int8)
        payload[f"{split}_offsets"] = offsets
        payload[f"{split}_y"] = np.asarray(d["y"], dtype=np.int64)
        payload[f"{split}_group_ids"] = np.asarray(d["group"])
        payload[f"{split}_sources"] = np.asarray(d["source"])
        payload[f"{split}_function_ids"] = np.asarray(d["function"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    print(f"raw rows: {n_rows} | sampled functions: ~{min(n_rows, len(split_of) * block_size)} in {len(split_of)} blocks of {block_size}")
    for split, d in data.items():
        y = np.asarray(d["y"])
        by_source = {s: int(np.sum(np.asarray(d["source"]) == s)) for s in ("human", "chatgpt", "deepseek", "qwen")}
        print(f"{split:5s}: {len(y):6d} programs | AI share {y.mean():.2f} | {by_source}")
    print("dropped (same rules for every source):", dict(sorted(dropped.items())))
    print(f"saved {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, default=RAW_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--n-functions", type=int, default=20000, help="approximate number of functions to sample")
    p.add_argument("--block-size", type=int, default=50, help="consecutive rows kept together in one split")
    p.add_argument("--min-states", type=int, default=5, help="drop programs with fewer AST states")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.raw, a.output, a.n_functions, a.block_size, a.min_states, a.seed)


if __name__ == "__main__":
    main()
