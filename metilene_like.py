#!/usr/bin/env python3
"""
metilene_like.py
=================

Reimplement the core ideas of metilene's de novo DMR detection in Python.

This script takes methylation beta values from per-sample bedGraph files,
groups samples into two cohorts, segments the genome to find regions with
consistent methylation differences, performs a Mann-Whitney U test for
each region, and reports differentially methylated regions (DMRs) with
Benjamini-Hochberg FDR correction.

bedGraph expectation:
    chrom   start   end     beta

Sample sheet expectation (TSV/CSV with header):
    sample  group   path

Usage example:
    python metilene_like.py design.tsv --output dmr.tsv
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("metilene_like")


@dataclass(frozen=True)
class Sample:
    """Container for a single sample description."""

    name: str
    group: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate metilene-style de novo DMR detection from bedGraph files."
    )
    parser.add_argument(
        "design",
        type=Path,
        help="Sample sheet with columns sample/group/path (TSV/CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dmr_calls.tsv"),
        help="Path to write detected DMRs (TSV). Default: dmr_calls.tsv",
    )
    parser.add_argument(
        "--min-cpgs",
        type=int,
        default=5,
        help="Minimum number of CpGs per region. Default: 5.",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=0.2,
        help="Minimum absolute methylation difference (beta) required to keep CpGs in a region. Default: 0.2.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=500,
        help="Maximum distance (bp) between consecutive CpGs to stay in the same region. Default: 500.",
    )
    parser.add_argument(
        "--max-qvalue",
        type=float,
        default=None,
        help="Optional q-value (FDR) threshold to filter reported regions.",
    )
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop CpGs with missing values across samples before segmentation (default: keep and ignore during tests).",
    )
    parser.add_argument(
        "--group-order",
        nargs=2,
        metavar=("GROUP_A", "GROUP_B"),
        default=None,
        help="Optional explicit order for the two groups (affects sign of differences).",
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Do not raise an error if no regions pass the filters; write an empty file instead.",
    )
    return parser.parse_args()


def read_design_sheet(path: Path) -> List[Sample]:
    if not path.exists():
        raise FileNotFoundError(f"Sample sheet {path} does not exist.")

    LOGGER.info("Reading sample sheet from %s", path)
    table = pd.read_csv(path, sep=None, engine="python")
    lowered = {col.lower(): col for col in table.columns}

    sample_col = _lookup_column(lowered, {"sample", "sample_id", "id"})
    group_col = _lookup_column(lowered, {"group", "condition", "cohort"})
    path_col = _lookup_column(lowered, {"path", "bedgraph", "file", "filepath"})

    parent = path.parent
    samples: List[Sample] = []

    for row in table.itertuples(index=False):
        sample_name = getattr(row, sample_col)
        group_name = getattr(row, group_col)
        file_path = getattr(row, path_col)
        resolved = Path(file_path)
        if not resolved.is_absolute():
            resolved = (parent / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"BedGraph for sample '{sample_name}' not found: {resolved}")
        samples.append(Sample(str(sample_name), str(group_name), resolved))

    if not samples:
        raise ValueError("No samples found in the design sheet.")
    unique_samples = {s.name for s in samples}
    if len(unique_samples) != len(samples):
        duplicates = len(samples) - len(unique_samples)
        raise ValueError(f"Duplicate sample names detected ({duplicates} duplicates). Sample names must be unique.")

    return samples


def _lookup_column(column_map: Dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]
    raise KeyError(f"Missing required column. Expected one of: {', '.join(candidates)}")


def group_samples(samples: Sequence[Sample], group_order: Sequence[str] | None = None) -> Tuple[List[Sample], List[Sample]]:
    groups: Dict[str, List[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group, []).append(sample)

    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, found: {list(groups)}")

    if group_order:
        missing = [g for g in group_order if g not in groups]
        if missing:
            raise ValueError(f"Group order references groups not in design sheet: {missing}")
        ordered_groups = [groups[group_order[0]], groups[group_order[1]]]
    else:
        ordered_labels = sorted(groups)
        ordered_groups = [groups[ordered_labels[0]], groups[ordered_labels[1]]]

    return ordered_groups[0], ordered_groups[1]


def load_bedgraph_matrix(samples: Sequence[Sample], drop_na: bool) -> pd.DataFrame:
    frames = []
    for sample in samples:
        LOGGER.info("Loading bedGraph for %s from %s", sample.name, sample.path)
        df = pd.read_csv(
            sample.path,
            sep="\t",
            comment="#",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["chrom", "start", "end", "beta"],
            dtype={"chrom": str, "start": np.int64, "end": np.int64, "beta": np.float64},
        )
        if df.empty:
            LOGGER.warning("BedGraph for sample %s is empty.", sample.name)
            continue
        frames.append(df.assign(sample=sample.name))

    if not frames:
        raise ValueError("No bedGraph data loaded; check sample sheet and file contents.")

    merged = pd.concat(frames, ignore_index=True)
    if merged.isnull().values.any():
        LOGGER.warning("NaN values detected in bedGraphs; they will be handled according to --drop-na.")

    matrix = (
        merged.pivot_table(
            index=["chrom", "start", "end"],
            columns="sample",
            values="beta",
        )
        .sort_index()
    )

    if drop_na:
        before = len(matrix)
        matrix = matrix.dropna(axis=0, how="any")
        LOGGER.info("Dropped %d CpGs with missing values.", before - len(matrix))

    return matrix


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvalues, dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    sorted_pvals = pvals[order]
    qvals = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = ranks[i]
        val = sorted_pvals[i] * n / rank
        val = min(val, prev, 1.0)
        qvals[order[i]] = val
        prev = val
    return qvals


def segment_cpgs(
    matrix: pd.DataFrame,
    group_a: Sequence[str],
    group_b: Sequence[str],
    min_cpgs: int,
    min_diff: float,
    max_gap: int,
) -> List[Dict[str, object]]:
    diffs = matrix[group_a].mean(axis=1) - matrix[group_b].mean(axis=1)
    index_tuples = list(matrix.index)

    regions: List[Dict[str, object]] = []
    current: List[int] = []
    current_sign: int | None = None
    prev_chrom: str | None = None
    prev_start: int | None = None

    def flush():
        nonlocal current, current_sign, prev_chrom, prev_start
        if len(current) >= min_cpgs:
            group_a_means: np.ndarray | None = None
            group_b_means: np.ndarray | None = None
            u_stat = math.nan
            p_value = math.nan

            region_indices = [index_tuples[i] for i in current]
            region_df = matrix.iloc[current]

            valid_group_a = [col for col in group_a if not region_df[col].isna().all()]
            valid_group_b = [col for col in group_b if not region_df[col].isna().all()]

            if not valid_group_a or not valid_group_b:
                LOGGER.debug("Skipping region due to samples without coverage in one group.")
            else:
                group_a_means = region_df[valid_group_a].mean(axis=0, skipna=True).to_numpy(dtype=float)
                group_b_means = region_df[valid_group_b].mean(axis=0, skipna=True).to_numpy(dtype=float)

                group_a_means = group_a_means[~np.isnan(group_a_means)]
                group_b_means = group_b_means[~np.isnan(group_b_means)]

                if len(group_a_means) == 0 or len(group_b_means) == 0:
                    LOGGER.debug("Skipping region with insufficient per-sample means after NaN filtering.")
                    group_a_means = group_b_means = None
                else:
                    try:
                        u_stat, p_value = mannwhitneyu(group_a_means, group_b_means, alternative="two-sided")
                    except ValueError as exc:
                        LOGGER.debug(
                            "Mann-Whitney failed for region %s:%d-%d: %s",
                            region_indices[0][0],
                            region_indices[0][1],
                            region_indices[-1][2],
                            exc,
                        )
                        p_value = math.nan
                        u_stat = math.nan

            if group_a_means is not None:
                chrom = region_indices[0][0]
                start = min(idx[1] for idx in region_indices)
                end = max(idx[2] for idx in region_indices)
                mean_a = float(np.mean(group_a_means))
                mean_b = float(np.mean(group_b_means))
                mean_diff = mean_a - mean_b
                abs_diff = abs(mean_diff)
                direction = "hyper" if mean_diff > 0 else "hypo"

                regions.append(
                    {
                        "chrom": chrom,
                        "start": int(start),
                        "end": int(end),
                        "num_cpgs": len(current),
                        "mean_beta_group_a": mean_a,
                        "mean_beta_group_b": mean_b,
                        "mean_diff": mean_diff,
                        "abs_diff": abs_diff,
                        "direction": direction,
                        "mannwhitney_u": u_stat,
                        "p_value": p_value,
                    }
                )
        current = []
        current_sign = None
        prev_chrom = None
        prev_start = None

    for idx, diff in enumerate(diffs):
        chrom, start, end = index_tuples[idx]
        if pd.isna(diff) or abs(diff) < min_diff:
            flush()
            continue

        sign = 1 if diff > 0 else -1
        is_contiguous = (
            current
            and prev_chrom == chrom
            and prev_start is not None
            and (start - prev_start) <= max_gap
            and sign == current_sign
        )

        if not current or is_contiguous:
            if not current:
                current_sign = sign
            current.append(idx)
            prev_chrom = chrom
            prev_start = start
        else:
            flush()
            current.append(idx)
            current_sign = sign
            prev_chrom = chrom
            prev_start = start

    flush()
    LOGGER.info("Identified %d candidate regions before statistical filtering.", len(regions))
    return regions


def main() -> None:
    args = parse_args()

    samples = read_design_sheet(args.design)
    group_a_samples, group_b_samples = group_samples(samples, args.group_order)
    LOGGER.info(
        "Detected groups: %s (%d samples) vs %s (%d samples)",
        group_a_samples[0].group,
        len(group_a_samples),
        group_b_samples[0].group,
        len(group_b_samples),
    )

    matrix = load_bedgraph_matrix(samples, drop_na=args.drop_na)
    if matrix.empty:
        raise ValueError("Combined methylation matrix is empty after preprocessing.")

    group_a_names = [sample.name for sample in group_a_samples if sample.name in matrix.columns]
    group_b_names = [sample.name for sample in group_b_samples if sample.name in matrix.columns]

    if not group_a_names or not group_b_names:
        raise ValueError("Samples missing from combined matrix; verify bedGraph loading.")

    regions = segment_cpgs(
        matrix=matrix,
        group_a=group_a_names,
        group_b=group_b_names,
        min_cpgs=args.min_cpgs,
        min_diff=args.min_diff,
        max_gap=args.max_gap,
    )

    if not regions:
        msg = "No candidate regions detected; consider relaxing thresholds."
        if args.allow_empty_output:
            LOGGER.warning(msg)
            args.output.write_text("")
            return
        raise RuntimeError(msg)

    region_df = pd.DataFrame(regions)
    non_nan_mask = ~region_df["p_value"].isna()
    region_df.loc[non_nan_mask, "q_value"] = benjamini_hochberg(region_df.loc[non_nan_mask, "p_value"].to_numpy())
    region_df.loc[~non_nan_mask, "q_value"] = np.nan

    if args.max_qvalue is not None:
        before = len(region_df)
        region_df = region_df[region_df["q_value"] <= args.max_qvalue]
        LOGGER.info("Filtered regions by q-value (<= %.3g): %d -> %d", args.max_qvalue, before, len(region_df))

    region_df = region_df.sort_values(["q_value", "abs_diff"], ascending=[True, False])

    if region_df.empty:
        msg = "No regions passed q-value filtering."
        if args.allow_empty_output:
            LOGGER.warning(msg)
            region_df.to_csv(args.output, sep="\t", index=False)
            return
        raise RuntimeError(msg)

    region_df.to_csv(args.output, sep="\t", index=False)
    LOGGER.info("Wrote %d regions to %s", len(region_df), args.output)


if __name__ == "__main__":
    main()
