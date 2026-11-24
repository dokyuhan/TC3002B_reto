"""Dataset manifest builders and offset-packed AST sequence archive generators."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing.parser import parse_to_state_sequence
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES

DEFAULT_CONPLAG_ROOT = Path("dataset/raw/conplag")
DEFAULT_IR_PLAG_ROOT = Path("dataset/raw/ir_plag")
DEFAULT_OUTPUT_DIR = Path("dataset/preprocessed")


@dataclass(frozen=True)
class ArchiveStats:
    """Summary metrics produced while building a dataset manifest and .npz archive."""

    dataset_name: str
    num_pairs: int
    split_counts: dict[str, int]
    positive_count: int
    negative_count: int
    total_parsed_files: int
    clean_parsed_files: int
    parse_success_rate: float
    mean_other_ratio: float
    median_other_ratio: float
    max_other_ratio: float
    group_overlap_count: int
    submission_overlap_count: int
    manifest_path: Path
    archive_path: Path


def _read_text_safe(path: Path) -> str:
    """Read source file as UTF-8 with Latin-1 fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _assign_conplag_folds_and_splits(
    labels: np.ndarray,
    group_ids: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Assign StratifiedGroupKFold fold IDs (0..4) and problem-disjoint train/val/test splits."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_ids = np.full(len(labels), -1, dtype=np.int32)
    dummy_x = np.zeros(len(labels), dtype=np.int8)

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(dummy_x, labels, groups=group_ids)):
        fold_ids[val_idx] = fold_idx

    if (fold_ids < 0).any():
        raise ValueError("StratifiedGroupKFold left unassigned samples.")

    splits = [
        "train" if int(f) in (0, 1, 2) else ("val" if int(f) == 3 else "test")
        for f in fold_ids
    ]
    return fold_ids, splits


def _verify_conplag_disjoint(
    splits: list[str],
    group_ids: list[str],
    sub1_list: list[str],
    sub2_list: list[str],
) -> tuple[int, int]:
    """Verify zero problem-group overlap and zero submission-ID overlap across splits."""
    groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    subs_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for split, gid, s1, s2 in zip(splits, group_ids, sub1_list, sub2_list):
        groups_by_split[split].add(gid)
        subs_by_split[split].add(s1)
        subs_by_split[split].add(s2)

    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    group_overlaps = sum(len(groups_by_split[a] & groups_by_split[b]) for a, b in split_pairs)
    sub_overlaps = sum(len(subs_by_split[a] & subs_by_split[b]) for a, b in split_pairs)

    if group_overlaps > 0 or sub_overlaps > 0:
        raise ValueError(
            f"Leakage detected across ConPlag splits: "
            f"group_overlaps={group_overlaps}, submission_overlaps={sub_overlaps}"
        )

    return group_overlaps, sub_overlaps


def build_conplag_manifest(
    version: str,
    conplag_root: Path = DEFAULT_CONPLAG_ROOT,
) -> tuple[pl.DataFrame, int, int]:
    """Load ConPlag version_1 or version_2 into a standardized Polars manifest."""
    labels_path = conplag_root / "labels.csv"
    version_dir = conplag_root / version
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing ConPlag labels file: {labels_path}")
    if not version_dir.exists():
        raise FileNotFoundError(f"Missing ConPlag version directory: {version_dir}")

    with labels_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    pair_ids: list[str] = []
    group_ids: list[str] = []
    labels_list: list[int] = []
    sub1_list: list[str] = []
    sub2_list: list[str] = []
    code_a_list: list[str] = []
    code_b_list: list[str] = []

    for row in rows:
        sub1 = row["sub1"].strip()
        sub2 = row["sub2"].strip()
        problem = row["problem"].strip()
        verdict = int(row["verdict"].strip())
        pair_id = f"{sub1}_{sub2}"

        pair_dir = version_dir / pair_id
        path_a = pair_dir / f"{sub1}.java"
        path_b = pair_dir / f"{sub2}.java"
        if not (path_a.exists() and path_b.exists()):
            raise FileNotFoundError(f"Missing ConPlag pair files in {pair_dir}")

        pair_ids.append(pair_id)
        group_ids.append(problem)
        labels_list.append(verdict)
        sub1_list.append(sub1)
        sub2_list.append(sub2)
        code_a_list.append(_read_text_safe(path_a))
        code_b_list.append(_read_text_safe(path_b))

    labels_arr = np.asarray(labels_list, dtype=np.uint8)
    fold_ids, splits = _assign_conplag_folds_and_splits(labels_arr, group_ids)
    group_overlaps, sub_overlaps = _verify_conplag_disjoint(
        splits=splits,
        group_ids=group_ids,
        sub1_list=sub1_list,
        sub2_list=sub2_list,
    )

    df = pl.DataFrame(
        {
            "pair_id": pl.Series(pair_ids, dtype=pl.String),
            "group_id": pl.Series(group_ids, dtype=pl.String),
            "fold_id": pl.Series(fold_ids, dtype=pl.Int32),
            "split": pl.Series(splits, dtype=pl.String),
            "level": pl.Series([""] * len(pair_ids), dtype=pl.String),
            "label": pl.Series(labels_arr, dtype=pl.UInt8),
            "code_a": pl.Series(code_a_list, dtype=pl.String),
            "code_b": pl.Series(code_b_list, dtype=pl.String),
        }
    )
    return df, group_overlaps, sub_overlaps


def build_ir_plag_manifest(
    ir_plag_root: Path = DEFAULT_IR_PLAG_ROOT,
) -> pl.DataFrame:
    """Load IR-Plag-Dataset cases (case-01..case-07) into a diagnostic Polars manifest."""
    if not ir_plag_root.exists():
        raise FileNotFoundError(f"Missing IR-Plag root directory: {ir_plag_root}")

    case_dirs = sorted(p for p in ir_plag_root.iterdir() if p.is_dir() and p.name.startswith("case-"))
    if not case_dirs:
        raise FileNotFoundError(f"No case-* directories found in {ir_plag_root}")

    pair_ids: list[str] = []
    group_ids: list[str] = []
    fold_ids: list[int] = []
    splits: list[str] = []
    levels: list[str] = []
    labels_list: list[int] = []
    code_a_list: list[str] = []
    code_b_list: list[str] = []

    for case_idx, case_dir in enumerate(case_dirs):
        orig_files = sorted((case_dir / "original").rglob("*.java"))
        if len(orig_files) != 1:
            raise ValueError(f"Expected exactly 1 original .java in {case_dir}, found {len(orig_files)}")
        orig_code = _read_text_safe(orig_files[0])

        plag_root = case_dir / "plagiarized"
        for cand_path in sorted(plag_root.rglob("*.java")):
            rel_parts = cand_path.relative_to(plag_root).parts
            level = rel_parts[0]
            sub_tag = "_".join(rel_parts[1:-1]) if len(rel_parts) > 2 else "root"
            pair_id = f"{case_dir.name}_{level}_{sub_tag}_{cand_path.stem}"

            pair_ids.append(pair_id)
            group_ids.append(case_dir.name)
            fold_ids.append(case_idx)
            splits.append("diagnostic")
            levels.append(level)
            labels_list.append(1)
            code_a_list.append(orig_code)
            code_b_list.append(_read_text_safe(cand_path))

        non_plag_root = case_dir / "non-plagiarized"
        for cand_path in sorted(non_plag_root.rglob("*.java")):
            rel_parts = cand_path.relative_to(non_plag_root).parts
            sub_tag = "_".join(rel_parts[:-1]) if len(rel_parts) > 1 else "root"
            pair_id = f"{case_dir.name}_NON_PLAG_{sub_tag}_{cand_path.stem}"

            pair_ids.append(pair_id)
            group_ids.append(case_dir.name)
            fold_ids.append(case_idx)
            splits.append("diagnostic")
            levels.append("NON_PLAG")
            labels_list.append(0)
            code_a_list.append(orig_code)
            code_b_list.append(_read_text_safe(cand_path))

    return pl.DataFrame(
        {
            "pair_id": pl.Series(pair_ids, dtype=pl.String),
            "group_id": pl.Series(group_ids, dtype=pl.String),
            "fold_id": pl.Series(fold_ids, dtype=pl.Int32),
            "split": pl.Series(splits, dtype=pl.String),
            "level": pl.Series(levels, dtype=pl.String),
            "label": pl.Series(labels_list, dtype=pl.UInt8),
            "code_a": pl.Series(code_a_list, dtype=pl.String),
            "code_b": pl.Series(code_b_list, dtype=pl.String),
        }
    )


def _pack_offset_sequences(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length 1D int32 sequences into (tokens, offsets) arrays."""
    lengths = np.asarray([int(s.size) for s in seqs], dtype=np.int64)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    if offsets[-1] == 0:
        tokens = np.empty((0,), dtype=np.int32)
    else:
        tokens = np.concatenate(seqs).astype(np.int32, copy=False)
    return tokens, offsets


def write_sequence_archive(
    manifest: pl.DataFrame,
    output_path: Path,
    language: str = "java",
    strip_boilerplate: bool = True,
    split_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, int], int, int, float, float, float]:
    """Parse manifest pairs and save an offset-packed .npz sequence archive."""
    if split_order is None:
        unique_splits = manifest["split"].unique(maintain_order=True).to_list()
        split_order = tuple(str(s) for s in unique_splits)

    payload: dict[str, np.ndarray] = {
        "num_states": np.asarray(NUM_STATES, dtype=np.int32),
        "state_names": np.asarray(DEFAULT_AST_STATES),
    }

    split_counts: dict[str, int] = {}
    other_ratios: list[float] = []
    clean_files = 0
    total_files = 0

    for split in split_order:
        sub = manifest.filter(pl.col("split") == split)
        split_counts[split] = sub.height

        seqs_a: list[np.ndarray] = []
        seqs_b: list[np.ndarray] = []

        for code_a, code_b in zip(sub["code_a"].to_list(), sub["code_b"].to_list()):
            res_a = parse_to_state_sequence(
                code_a, language=language, strip_boilerplate=strip_boilerplate
            )
            res_b = parse_to_state_sequence(
                code_b, language=language, strip_boilerplate=strip_boilerplate
            )
            seqs_a.append(res_a.states)
            seqs_b.append(res_b.states)

            total_files += 2
            clean_files += int(not res_a.has_error) + int(not res_b.has_error)
            other_ratios.append(res_a.other_ratio)
            other_ratios.append(res_b.other_ratio)

        tok_a, off_a = _pack_offset_sequences(seqs_a)
        tok_b, off_b = _pack_offset_sequences(seqs_b)

        payload[f"{split}_tokens_a"] = tok_a
        payload[f"{split}_offsets_a"] = off_a
        payload[f"{split}_tokens_b"] = tok_b
        payload[f"{split}_offsets_b"] = off_b
        payload[f"{split}_y"] = sub["label"].to_numpy().astype(np.uint8, copy=False)
        payload[f"{split}_pair_ids"] = np.asarray(sub["pair_id"].to_list())
        payload[f"{split}_group_ids"] = np.asarray(sub["group_id"].to_list())
        payload[f"{split}_fold_ids"] = sub["fold_id"].to_numpy().astype(np.int32, copy=False)
        payload[f"{split}_levels"] = np.asarray(sub["level"].to_list())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    ratios_arr = np.asarray(other_ratios, dtype=np.float64)
    mean_other = float(ratios_arr.mean()) if ratios_arr.size else 0.0
    median_other = float(np.median(ratios_arr)) if ratios_arr.size else 0.0
    max_other = float(ratios_arr.max()) if ratios_arr.size else 0.0
    return split_counts, total_files, clean_files, mean_other, median_other, max_other


def build_all_datasets(
    conplag_root: Path = DEFAULT_CONPLAG_ROOT,
    ir_plag_root: Path = DEFAULT_IR_PLAG_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[ArchiveStats]:
    """Generate manifests and offset-packed .npz archives for ConPlag v2, v1, and IR-Plag."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ArchiveStats] = []

    for version, prefix in (("version_2", "conplag_v2"), ("version_1", "conplag_v1")):
        manifest, group_ov, sub_ov = build_conplag_manifest(version=version, conplag_root=conplag_root)
        manifest_path = output_dir / f"{prefix}_manifest.parquet"
        archive_path = output_dir / f"{prefix}_sequences.npz"
        manifest.write_parquet(manifest_path)

        split_counts, total_f, clean_f, mean_o, med_o, max_o = write_sequence_archive(
            manifest=manifest,
            output_path=archive_path,
            language="java",
            strip_boilerplate=True,
            split_order=("train", "val", "test"),
        )
        pos = int(manifest["label"].sum())
        neg = manifest.height - pos
        results.append(
            ArchiveStats(
                dataset_name=prefix,
                num_pairs=manifest.height,
                split_counts=split_counts,
                positive_count=pos,
                negative_count=neg,
                total_parsed_files=total_f,
                clean_parsed_files=clean_f,
                parse_success_rate=(clean_f / total_f) if total_f else 0.0,
                mean_other_ratio=mean_o,
                median_other_ratio=med_o,
                max_other_ratio=max_o,
                group_overlap_count=group_ov,
                submission_overlap_count=sub_ov,
                manifest_path=manifest_path,
                archive_path=archive_path,
            )
        )

    ir_manifest = build_ir_plag_manifest(ir_plag_root=ir_plag_root)
    ir_manifest_path = output_dir / "ir_plag_manifest.parquet"
    ir_archive_path = output_dir / "ir_plag_sequences.npz"
    ir_manifest.write_parquet(ir_manifest_path)

    split_counts, total_f, clean_f, mean_o, med_o, max_o = write_sequence_archive(
        manifest=ir_manifest,
        output_path=ir_archive_path,
        language="java",
        strip_boilerplate=True,
        split_order=("diagnostic",),
    )
    pos = int(ir_manifest["label"].sum())
    neg = ir_manifest.height - pos
    results.append(
        ArchiveStats(
            dataset_name="ir_plag",
            num_pairs=ir_manifest.height,
            split_counts=split_counts,
            positive_count=pos,
            negative_count=neg,
            total_parsed_files=total_f,
            clean_parsed_files=clean_f,
            parse_success_rate=(clean_f / total_f) if total_f else 0.0,
            mean_other_ratio=mean_o,
            median_other_ratio=med_o,
            max_other_ratio=max_o,
            group_overlap_count=0,
            submission_overlap_count=0,
            manifest_path=ir_manifest_path,
            archive_path=ir_archive_path,
        )
    )

    return results


def main() -> None:
    """CLI entrypoint to build all manifests and sequence archives and print metrics."""
    stats_list = build_all_datasets()
    for st in stats_list:
        print(
            f"[{st.dataset_name}] pairs={st.num_pairs} (pos={st.positive_count}, neg={st.negative_count}) "
            f"splits={st.split_counts} "
            f"parse_clean={st.clean_parsed_files}/{st.total_parsed_files} ({st.parse_success_rate * 100:.2f}%) "
            f"other_ratio(mean={st.mean_other_ratio:.4f}, median={st.median_other_ratio:.4f}, max={st.max_other_ratio:.4f}) "
            f"overlaps(group={st.group_overlap_count}, submission={st.submission_overlap_count})"
        )
        print(f"  manifest: {st.manifest_path.resolve()}")
        print(f"  archive:  {st.archive_path.resolve()}")


if __name__ == "__main__":
    main()
