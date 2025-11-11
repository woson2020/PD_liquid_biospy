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

bedGraph expectation (per sample):
    Required columns: chrom   start   end   beta
    Optional columns (recommended for metilene parity):
        - coverage  (total reads covering the CpG)
        - meth_count and unmeth_count (or combined coverage inferred from them)

If coverage columns are supplied（覆盖度信息）,统计检验和分段都会使用加权逻辑，更贴近 metilene。

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("metilene_like")


@dataclass(frozen=True)
class Sample:
    """单一样本的描述信息容器，包含名称、分组和bedGraph路径。"""

    name: str
    group: str
    path: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回包含所有选项的 Namespace 对象。"""
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
    parser.add_argument(
        "--require-coverage",
        action="store_true",
        help="Enforce presence of coverage columns in bedGraph files; error if缺失。",
    )
    return parser.parse_args()


def read_design_sheet(path: Path) -> List[Sample]:
    """读取设计表，解析样本信息并返回 Sample 对象列表。"""
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
    """在列名映射中查找候选列，找到则返回原始列名。"""
    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]
    raise KeyError(f"Missing required column. Expected one of: {', '.join(candidates)}")


def group_samples(samples: Sequence[Sample], group_order: Sequence[str] | None = None) -> Tuple[List[Sample], List[Sample]]:
    """按照分组信息将样本划分为两组，可选指定组顺序。"""
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


def load_bedgraph_tables(
    samples: Sequence[Sample],
    drop_na: bool,
    require_coverage: bool,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    """加载所有样本的 bedGraph 数据并构建 CpG × Sample 的甲基化和覆盖度矩阵。"""
    frames: List[pd.DataFrame] = []
    coverage_available = True

    for sample in samples:
        LOGGER.info("Loading bedGraph for %s from %s", sample.name, sample.path)
        df = pd.read_csv(
            sample.path,
            sep="\t",
            comment="#",
            header=None,
            dtype={0: str, 1: np.int64, 2: np.int64},
        )
        if df.empty:
            LOGGER.warning("BedGraph for sample %s is empty.", sample.name)
            continue

        num_cols = df.shape[1]
        if num_cols < 4:
            raise ValueError(f"BedGraph for sample {sample.name} has fewer than 4 columns.")

        base = df.iloc[:, :4].copy()
        base.columns = ["chrom", "start", "end", "beta"]

        if num_cols >= 6:
            meth = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            unmeth = pd.to_numeric(df.iloc[:, 5], errors="coerce")
            coverage = meth + unmeth
        elif num_cols == 5:
            coverage = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            meth = coverage * pd.to_numeric(base["beta"], errors="coerce")
            unmeth = coverage - meth
        else:
            coverage = pd.Series(np.nan, index=base.index, dtype=np.float64)
            meth = pd.Series(np.nan, index=base.index, dtype=np.float64)
            unmeth = pd.Series(np.nan, index=base.index, dtype=np.float64)
            coverage_available = False

        base["coverage"] = coverage.astype(float)
        base["meth_count"] = meth.astype(float)
        base["unmeth_count"] = unmeth.astype(float)

        beta = pd.to_numeric(base["beta"], errors="coerce")
        try:
            max_beta = np.nanmax(beta.to_numpy(dtype=float))
        except ValueError:
            max_beta = 0.0
        needs_scaling = max_beta > 1.0
        if needs_scaling:
            beta = beta / 100.0
        base["beta"] = beta

        base["sample"] = sample.name
        frames.append(base)

    if not frames:
        raise ValueError("No bedGraph data loaded; check sample sheet and file contents.")

    if require_coverage and not coverage_available:
        raise ValueError("Coverage columns were required but not found in all bedGraph files.")

    merged = pd.concat(frames, ignore_index=True)
    if merged.isnull().values.any():
        LOGGER.warning("NaN values detected in bedGraphs; they will be handled according to --drop-na.")

    beta_matrix = (
        merged.pivot_table(
            index=["chrom", "start", "end"],
            columns="sample",
            values="beta",
        )
        .sort_index()
    )

    coverage_matrix: Optional[pd.DataFrame]
    if coverage_available:
        coverage_matrix = (
            merged.pivot_table(
                index=["chrom", "start", "end"],
                columns="sample",
                values="coverage",
            )
            .sort_index()
        )
    else:
        coverage_matrix = None

    if drop_na:
        before = len(beta_matrix)
        beta_matrix = beta_matrix.dropna(axis=0, how="any")
        if coverage_matrix is not None:
            coverage_matrix = coverage_matrix.loc[beta_matrix.index]
        LOGGER.info("Dropped %d CpGs with missing values.", before - len(beta_matrix))

    return beta_matrix, coverage_matrix, coverage_available


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """对 p 值序列执行 Benjamini-Hochberg 多重检验校正，返回 q 值数组。"""
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
    beta_matrix: pd.DataFrame,
    coverage_matrix: Optional[pd.DataFrame],
    group_a: Sequence[str],
    group_b: Sequence[str],
    min_cpgs: int,
    min_diff: float,
    max_gap: int,
) -> List[Dict[str, object]]:
    """根据甲基化差异对 CpG 进行递归二分分段，生成候选 DMR 区域并计算统计量。"""

    def compute_group_stats(
        beta_block: pd.DataFrame,
        coverage_block: Optional[pd.DataFrame],
        columns: Sequence[str],
    ) -> Dict[str, object]:
        if not columns:
            return {
                "per_sample_mean": np.array([], dtype=float),
                "per_sample_cov": np.array([], dtype=float),
                "total_cov": 0.0,
                "total_meth": 0.0,
                "mean": math.nan,
            }

        beta_subset = beta_block[columns]
        if coverage_block is not None:
            coverage_subset = coverage_block[columns].fillna(0.0)
            beta_filled = beta_subset.fillna(0.0)
            meth_per_sample = (beta_filled * coverage_subset).sum(axis=0, skipna=True).astype(float)
            cov_per_sample = coverage_subset.sum(axis=0, skipna=True).astype(float)
            per_sample_mean = np.divide(
                meth_per_sample.to_numpy(),
                cov_per_sample.to_numpy(),
                out=np.full(cov_per_sample.shape, np.nan, dtype=float),
                where=cov_per_sample.to_numpy() > 0,
            )
            total_cov = float(np.nansum(cov_per_sample.to_numpy()))
            total_meth = float(np.nansum(meth_per_sample.to_numpy()))
            mean_value = total_meth / total_cov if total_cov > 0 else math.nan
            return {
                "per_sample_mean": per_sample_mean,
                "per_sample_cov": cov_per_sample.to_numpy(),
                "total_cov": total_cov,
                "total_meth": total_meth,
                "mean": mean_value,
            }

        per_sample_mean_series = beta_subset.mean(axis=0, skipna=True)
        per_sample_counts = beta_subset.notna().sum(axis=0).astype(float)
        return {
            "per_sample_mean": per_sample_mean_series.to_numpy(dtype=float),
            "per_sample_cov": per_sample_counts.to_numpy(dtype=float),
            "total_cov": math.nan,
            "total_meth": math.nan,
            "mean": float(per_sample_mean_series.mean(skipna=True)),
        }

    def weighted_group_mean(
        beta_df: pd.DataFrame,
        coverage_df: Optional[pd.DataFrame],
        columns: Sequence[str],
    ) -> Tuple[pd.Series, pd.Series]:
        beta_subset = beta_df[columns]
        if coverage_df is not None:
            coverage_subset = coverage_df[columns].fillna(0.0)
            weighted_sum = (beta_subset.fillna(0.0) * coverage_subset).sum(axis=1, min_count=1)
            weight = coverage_subset.sum(axis=1, min_count=1)
            mean_series = weighted_sum / weight
        else:
            mean_series = beta_subset.mean(axis=1, skipna=True)
            weight = beta_subset.notna().sum(axis=1).astype(float)
        return mean_series, weight

    beta_group_a = [col for col in group_a if col in beta_matrix.columns]
    beta_group_b = [col for col in group_b if col in beta_matrix.columns]

    coverage_df_a = coverage_matrix[beta_group_a] if coverage_matrix is not None else None
    coverage_df_b = coverage_matrix[beta_group_b] if coverage_matrix is not None else None

    mean_a_series, weight_a_series = weighted_group_mean(beta_matrix, coverage_df_a, beta_group_a)
    mean_b_series, weight_b_series = weighted_group_mean(beta_matrix, coverage_df_b, beta_group_b)

    diff_series = mean_a_series - mean_b_series
    weight_series = weight_a_series + weight_b_series

    index_tuples = list(beta_matrix.index)
    regions: List[Dict[str, object]] = []

    def binary_segment(block_indices: List[int]) -> List[Tuple[int, int, float]]:
        if len(block_indices) < min_cpgs:
            return []
        diffs_block = diff_series.iloc[block_indices].to_numpy()
        weights_block = weight_series.iloc[block_indices].to_numpy()

        prefix_w = np.concatenate(([0.0], np.cumsum(weights_block)))
        prefix_s = np.concatenate(([0.0], np.cumsum(weights_block * diffs_block)))

        results: List[Tuple[int, int, float]] = []

        def recurse(left: int, right: int) -> None:
            length = right - left
            if length < min_cpgs:
                return

            best_span: Optional[Tuple[int, int]] = None
            best_score = 0.0

            for start in range(left, right - min_cpgs + 1):
                min_end = start + min_cpgs
                candidate_ends = np.arange(min_end, right + 1)
                weights = prefix_w[candidate_ends] - prefix_w[start]
                sums = prefix_s[candidate_ends] - prefix_s[start]

                valid_mask = weights > 0
                if not np.any(valid_mask):
                    continue

                weights = weights[valid_mask]
                sums = sums[valid_mask]
                candidate_ends = candidate_ends[valid_mask]

                mean_diffs = sums / weights
                diff_mask = np.abs(mean_diffs) >= min_diff
                if not np.any(diff_mask):
                    continue

                weights = weights[diff_mask]
                sums = sums[diff_mask]
                candidate_ends = candidate_ends[diff_mask]

                scores = np.abs(sums) / np.sqrt(weights)
                local_idx = int(np.argmax(scores))
                score = float(scores[local_idx])
                if score > best_score:
                    best_score = score
                    best_span = (start, int(candidate_ends[local_idx]))

            if best_span is None:
                return

            results.append((best_span[0], best_span[1], best_score))
            recurse(left, best_span[0])
            recurse(best_span[1], right)

        recurse(0, len(block_indices))
        return results

    blocks: List[List[int]] = []
    current_block: List[int] = []
    prev_chrom: Optional[str] = None
    prev_start: Optional[int] = None

    for idx, (chrom, start, end) in enumerate(index_tuples):
        diff_value = diff_series.iloc[idx]
        weight_value = weight_series.iloc[idx]
        if pd.isna(diff_value) or pd.isna(weight_value) or weight_value <= 0:
            if current_block:
                blocks.append(current_block)
                current_block = []
            prev_chrom = None
            prev_start = None
            continue

        if current_block:
            max_gap_exceeded = max_gap is not None and (chrom != prev_chrom or (start - prev_start) > max_gap)
            if max_gap_exceeded:
                blocks.append(current_block)
                current_block = []

        if not current_block:
            prev_chrom = chrom

        current_block.append(idx)
        prev_start = start

    if current_block:
        blocks.append(current_block)

    for block_indices in blocks:
        segments = binary_segment(block_indices)
        for local_start, local_end, score in segments:
            global_indices = block_indices[local_start:local_end]
            region_beta = beta_matrix.iloc[global_indices]
            region_cov = coverage_matrix.iloc[global_indices] if coverage_matrix is not None else None

            chr_name = index_tuples[global_indices[0]][0]
            region_start = min(index_tuples[i][1] for i in global_indices)
            region_end = max(index_tuples[i][2] for i in global_indices)

            stats_a = compute_group_stats(region_beta, region_cov, beta_group_a)
            stats_b = compute_group_stats(region_beta, region_cov, beta_group_b)

            mean_a = stats_a["mean"]
            mean_b = stats_b["mean"]
            mean_diff = mean_a - mean_b if not (math.isnan(mean_a) or math.isnan(mean_b)) else math.nan
            abs_diff = abs(mean_diff) if not math.isnan(mean_diff) else math.nan
            if math.isnan(mean_diff):
                direction = "undetermined"
            else:
                direction = "hyper" if mean_diff > 0 else "hypo"

            wald_z = math.nan
            p_value = math.nan
            u_stat = math.nan

            if (
                coverage_matrix is not None
                and stats_a["total_cov"] > 0
                and stats_b["total_cov"] > 0
            ):
                p1 = stats_a["total_meth"] / stats_a["total_cov"]
                p2 = stats_b["total_meth"] / stats_b["total_cov"]
                pooled_cov = stats_a["total_cov"] + stats_b["total_cov"]
                pooled_meth = stats_a["total_meth"] + stats_b["total_meth"]
                pooled = pooled_meth / pooled_cov if pooled_cov > 0 else math.nan
                variance = pooled * (1.0 - pooled) * (1.0 / stats_a["total_cov"] + 1.0 / stats_b["total_cov"])
                if variance > 0 and not math.isnan(variance):
                    wald_z = (p1 - p2) / math.sqrt(variance)
                    p_value = 2 * norm.sf(abs(wald_z))

            if math.isnan(p_value):
                group_a_vals = stats_a["per_sample_mean"]
                group_b_vals = stats_b["per_sample_mean"]
                group_a_vals = group_a_vals[~np.isnan(group_a_vals)]
                group_b_vals = group_b_vals[~np.isnan(group_b_vals)]
                if len(group_a_vals) > 0 and len(group_b_vals) > 0:
                    try:
                        u_stat, p_value = mannwhitneyu(group_a_vals, group_b_vals, alternative="two-sided")
                    except ValueError as exc:
                        LOGGER.debug(
                            "Mann-Whitney failed for region %s:%d-%d: %s",
                            chr_name,
                            region_start,
                            region_end,
                            exc,
                        )
                        p_value = math.nan
                        u_stat = math.nan

            regions.append(
                {
                    "chrom": chr_name,
                    "start": int(region_start),
                    "end": int(region_end),
                    "num_cpgs": len(global_indices),
                    "mean_beta_group_a": float(mean_a),
                    "mean_beta_group_b": float(mean_b),
                    "mean_diff": float(mean_diff),
                    "abs_diff": float(abs_diff),
                    "direction": direction,
                    "segment_score": float(score),
                    "total_cov_group_a": float(stats_a["total_cov"]),
                    "total_cov_group_b": float(stats_b["total_cov"]),
                    "wald_z": float(wald_z),
                    "mannwhitney_u": float(u_stat),
                    "p_value": float(p_value),
                }
            )

    LOGGER.info("Identified %d candidate regions before statistical filtering.", len(regions))
    return regions


def main() -> None:
    """主入口：解析参数、加载数据、执行分段检测并输出结果。"""
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

    beta_matrix, coverage_matrix, coverage_available = load_bedgraph_tables(
        samples=samples,
        drop_na=args.drop_na,
        require_coverage=args.require_coverage,
    )
    if beta_matrix.empty:
        raise ValueError("Combined methylation matrix is empty after preprocessing.")

    if coverage_matrix is not None:
        LOGGER.info("Coverage-aware weighting enabled for segmentation and statistical testing.")
    else:
        LOGGER.warning(
            "No coverage columns detected; falling back to unweighted averages and Mann-Whitney tests."
        )

    group_a_names = [sample.name for sample in group_a_samples if sample.name in beta_matrix.columns]
    group_b_names = [sample.name for sample in group_b_samples if sample.name in beta_matrix.columns]

    if not group_a_names or not group_b_names:
        raise ValueError("Samples missing from combined matrix; verify bedGraph loading.")

    regions = segment_cpgs(
        beta_matrix=beta_matrix,
        coverage_matrix=coverage_matrix,
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
