#!/usr/bin/env python3
"""
Compute cfDNA fragment length distribution from paired-end BAM files.

This script focuses on fragments whose read1/read2 both map to the same
reference and start coordinate (i.e., properly paired fragments coming from the
same genomic locus).  It is intended to be "%run" inside Jupyter or executed as
standalone CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pysam
from tqdm import tqdm


plt.style.use("seaborn-v0_8-whitegrid")


def compute_fragment_length_distribution(
    bam_path: str | Path,
    *,
    min_length: int = 0,
    max_length: Optional[int] = None,
    require_same_start: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with fragment length counts/fractions.

    Only read1 entries are inspected.  Pairs are accepted when:
      * both mates are mapped
      * the alignment is a proper pair
      * neither read is marked secondary/supplementary/duplicate
      * (optional) R1/R2 share the same reference id and start coordinate
    """

    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    counter = Counter()

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        iterator = bam.fetch(until_eof=True)
        if verbose:
            iterator = tqdm(iterator, desc="Scanning reads")

        for read in iterator:
            if not read.is_paired or not read.is_read1:
                continue
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary or read.is_duplicate:
                continue
            if not read.is_proper_pair:
                continue

            if require_same_start:
                same_chr = read.reference_id == read.next_reference_id
                same_pos = read.reference_start == read.next_reference_start
                if not (same_chr and same_pos):
                    continue

            frag_len = abs(read.template_length)
            if frag_len == 0:
                continue
            if frag_len < min_length:
                continue
            if max_length is not None and frag_len > max_length:
                continue

            counter[frag_len] += 1

    data = (
        pd.Series(counter, name="count")
        .rename_axis("fragment_length")
        .sort_index()
        .reset_index()
    )
    data["fraction"] = data["count"] / data["count"].sum()
    return data


def plot_fragment_distribution(
    distribution: pd.DataFrame,
    *,
    bin_width: int = 5,
    ax: Optional[plt.Axes] = None,
    color: str = "#1f77b4",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot binned fragment length fractions.
    """

    if {"fragment_length", "fraction"} - set(distribution.columns):
        raise ValueError("distribution must include 'fragment_length' and 'fraction'")

    binned = (
        distribution.assign(bin=lambda df: (df["fragment_length"] // bin_width) * bin_width)
        .groupby("bin", as_index=False)["fraction"]
        .sum()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    ax.bar(binned["bin"], binned["fraction"], width=bin_width * 0.9, color=color)
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Fraction of read pairs")
    ax.set_xlim(left=0)
    ax.set_title("cfDNA fragment length distribution")
    return fig, ax


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute cfDNA fragment length distribution from BAM."
    )
    parser.add_argument("--bam", required=True, help="Input paired-end BAM file.")
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Minimum fragment length to keep (default: 0).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum fragment length to keep (default: None).",
    )
    parser.add_argument(
        "--no-same-start",
        action="store_true",
        help="Disable enforcing same reference/start between R1/R2.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=5,
        help="Bin width (bp) for the bar plot (default: 5).",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=None,
        help="Optional CSV path to save the distribution table.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional image path (e.g., PNG/PDF) to save the plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting entirely.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    distribution = compute_fragment_length_distribution(
        args.bam,
        min_length=args.min_length,
        max_length=args.max_length,
        require_same_start=not args.no_same_start,
        verbose=not args.no_progress,
    )

    if args.output_table:
        distribution.to_csv(args.output_table, index=False)
        print(f"Saved table -> {args.output_table}")

    if not args.no_plot or args.output_plot:
        fig, ax = plot_fragment_distribution(distribution, bin_width=args.bin_width)
        if args.output_plot:
            fig.savefig(args.output_plot, dpi=300, bbox_inches="tight")
            print(f"Saved plot -> {args.output_plot}")
        if args.no_plot:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
