#!/usr/bin/env python3
"""
dmr_methylation_summary.py
===========================

根据已知的 DMR 区域（例如 metilene 输出）和每个样本的 CpG bedGraph，
计算每个 DMR 在每个样本中的加权平均甲基化水平。

bedGraph 要求至少包含以下列（无表头）：
    chrom  start  end  coverage  meth_read

覆盖度加权方式：
    区域 methylation = sum(meth_read_i) / sum(coverage_i)

若某个 DMR 在某个样本的覆盖度总和为 0，则输出 NaN。

示例：
    python dmr_methylation_summary.py \
        dmrs.tsv design.tsv \
        --output dmr_beta.tsv \
        --dmr-id-column id \
        --add-coverage
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("dmr_methylation_summary")


@dataclass(frozen=True)
class SampleInfo:
    """样本配置，包括名称与 bedGraph 路径。"""

    name: str
    path: Path


@dataclass
class SampleBedGraph:
    """按染色体存储的 bedGraph 数据（numpy 数组形式）。"""

    chrom_arrays: Dict[str, Dict[str, np.ndarray]]

    def summarize(self, chrom: str, start: int, end: int) -> tuple[float, float]:
        """对指定区域返回 (meth_sum, coverage_sum)。"""
        arrays = self.chrom_arrays.get(chrom)
        if arrays is None:
            return (0.0, 0.0)

        starts = arrays["start"]
        ends = arrays["end"]
        coverage = arrays["coverage"]
        meth = arrays["meth"]

        left = np.searchsorted(starts, start, side="left")
        right = np.searchsorted(starts, end, side="right")
        if right <= left:
            return (0.0, 0.0)

        slice_starts = starts[left:right]
        slice_ends = ends[left:right]
        mask = slice_ends <= end
        if not np.any(mask):
            return (0.0, 0.0)

        idx = np.nonzero(mask)[0]
        cov_sum = float(np.sum(coverage[left + idx]))
        meth_sum = float(np.sum(meth[left + idx]))
        return (meth_sum, cov_sum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sample methylation levels for DMRs using bedGraph coverage-weighted averages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dmr_file",
        type=Path,
        help="DMR 列表（TSV/CSV）。需包含染色体和区间信息。",
    )
    parser.add_argument(
        "design",
        type=Path,
        help="样本设计表（TSV/CSV），包含 sample 和 path 列。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dmr_methylation_matrix.tsv"),
        help="输出文件路径（TSV）。",
    )
    parser.add_argument(
        "--chrom-column",
        default="chrom",
        help="DMR 文件中表示染色体的列名。",
    )
    parser.add_argument(
        "--start-column",
        default="start",
        help="DMR 文件中表示起始坐标的列名（0-based, inclusive）。",
    )
    parser.add_argument(
        "--end-column",
        default="end",
        help="DMR 文件中表示终止坐标的列名（0-based, exclusive/inclusive 均可；脚本按 end >= CpG end 处理）。",
    )
    parser.add_argument(
        "--dmr-id-column",
        default=None,
        help="DMR 唯一标识列；若缺失则使用行号作为 ID。",
    )
    parser.add_argument(
        "--add-coverage",
        action="store_true",
        help="在输出中追加每个样本的覆盖度总和列（coverage）。",
    )
    parser.add_argument(
        "--float-format",
        default="%.6f",
        help="输出浮点数格式控制。",
    )
    return parser.parse_args()


def read_design(design_path: Path) -> List[SampleInfo]:
    if not design_path.exists():
        raise FileNotFoundError(f"Design file not found: {design_path}")

    table = pd.read_csv(design_path, sep=None, engine="python")
    lower_map = {col.lower(): col for col in table.columns}
    sample_col = _lookup_column(lower_map, {"sample", "sample_id", "id"})
    path_col = _lookup_column(lower_map, {"path", "bedgraph", "file"})

    samples: List[SampleInfo] = []
    for row in table.itertuples(index=False):
        name = str(getattr(row, sample_col))
        bedgraph_path = Path(getattr(row, path_col))
        if not bedgraph_path.is_absolute():
            bedgraph_path = (design_path.parent / bedgraph_path).resolve()
        if not bedgraph_path.exists():
            raise FileNotFoundError(f"BedGraph for sample '{name}' not found: {bedgraph_path}")
        samples.append(SampleInfo(name=name, path=bedgraph_path))

    if not samples:
        raise ValueError("Design file contains no samples.")
    return samples


def _lookup_column(column_map: Dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]
    raise KeyError(f"Expected one of columns: {', '.join(candidates)}")


def load_bedgraph(sample: SampleInfo) -> SampleBedGraph:
    LOGGER.info("Loading bedGraph for %s from %s", sample.name, sample.path)
    df = pd.read_csv(
        sample.path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["chrom", "start", "end", "coverage", "meth_read"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "coverage": np.float64, "meth_read": np.float64},
    )
    if df.empty:
        LOGGER.warning("BedGraph for sample %s is empty.", sample.name)

    df = df.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)

    chrom_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for chrom, sub in df.groupby("chrom", sort=False):
        chrom_arrays[chrom] = {
            "start": sub["start"].to_numpy(dtype=np.int64),
            "end": sub["end"].to_numpy(dtype=np.int64),
            "coverage": sub["coverage"].to_numpy(dtype=np.float64),
            "meth": sub["meth_read"].to_numpy(dtype=np.float64),
        }
    return SampleBedGraph(chrom_arrays=chrom_arrays)


def summarize_dmrs(
    dmrs: pd.DataFrame,
    dmrs_id: pd.Series,
    samples: List[SampleInfo],
    add_coverage: bool,
) -> pd.DataFrame:
    summary: Dict[str, np.ndarray] = {}
    coverage_summary: Dict[str, np.ndarray] = {}

    cached_bedgraphs: Dict[str, SampleBedGraph] = {}

    for sample in samples:
        bedgraph = load_bedgraph(sample)
        cached_bedgraphs[sample.name] = bedgraph

    n_dmrs = len(dmrs)
    for sample in samples:
        bedgraph = cached_bedgraphs[sample.name]
        meth_array = np.zeros(n_dmrs, dtype=np.float64)
        cov_array = np.zeros(n_dmrs, dtype=np.float64)

        for idx, row in enumerate(dmrs.itertuples(index=False)):
            meth_sum, cov_sum = bedgraph.summarize(
                chrom=row.chrom,
                start=row.start,
                end=row.end,
            )
            cov_array[idx] = cov_sum
            if cov_sum > 0:
                meth_array[idx] = meth_sum / cov_sum
            else:
                meth_array[idx] = np.nan

        summary[sample.name] = meth_array
        if add_coverage:
            coverage_summary[f"{sample.name}_coverage"] = cov_array

    result = pd.DataFrame(
        {
            "dmr_id": dmrs_id,
            "chrom": dmrs["chrom"],
            "start": dmrs["start"],
            "end": dmrs["end"],
        }
    )

    for sample in samples:
        result[sample.name] = summary[sample.name]
        if add_coverage:
            result[f"{sample.name}_coverage"] = coverage_summary[f"{sample.name}_coverage"]

    return result


def main() -> None:
    args = parse_args()

    dmrs = pd.read_csv(args.dmr_file, sep=None, engine="python")
    required_cols = {args.chrom_column, args.start_column, args.end_column}
    missing_cols = required_cols - set(dmrs.columns)
    if missing_cols:
        raise KeyError(f"DMR file missing required columns: {', '.join(sorted(missing_cols))}")

    dmrs = dmrs.rename(
        columns={
            args.chrom_column: "chrom",
            args.start_column: "start",
            args.end_column: "end",
        }
    )
    dmrs[["start", "end"]] = dmrs[["start", "end"]].astype(np.int64)

    if args.dmr_id_column:
        if args.dmr_id_column not in dmrs.columns:
            raise KeyError(f"DMR ID column '{args.dmr_id_column}' not found in {args.dmr_file}")
        dmr_id = dmrs[args.dmr_id_column].astype(str)
    else:
        dmr_id = dmrs.index.map(lambda i: f"DMR_{i+1}")

    samples = read_design(args.design)

    summary_df = summarize_dmrs(
        dmrs=dmrs[["chrom", "start", "end"]],
        dmrs_id=dmr_id,
        samples=samples,
        add_coverage=args.add_coverage,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, sep="\t", index=False, float_format=args.float_format)
    LOGGER.info("Wrote methylation summary for %d DMRs and %d samples to %s", len(summary_df), len(samples), args.output)


if __name__ == "__main__":
    main()
