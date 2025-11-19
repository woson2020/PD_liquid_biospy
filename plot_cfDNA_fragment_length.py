#!/usr/bin/env python3
"""
Minimal script for plotting cfDNA fragment length distribution in Jupyter.
Replace `bam_path` with your BAM location and run the file (e.g. `%run plot_cfDNA_fragment_length.py`).
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pysam
from tqdm import tqdm


# === User-configurable section ===
bam_path = Path("xxx")  # TODO: replace with the real BAM path
min_length = 0          # optional lower bound for fragment length (bp)
max_length = None       # optional upper bound; set to an integer or keep None
bin_width = 5           # bin width for the histogram-style bar plot
require_same_start = True  # enforce read1/read2 map to same reference/start
show_progress = True
# ===============================


if not bam_path.exists():
    raise FileNotFoundError(f"BAM file not found: {bam_path}")

counter = Counter()

with pysam.AlignmentFile(bam_path, "rb") as bam:
    iterator = bam.fetch(until_eof=True)
    if show_progress:
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

distribution = (
    pd.Series(counter, name="count")
    .rename_axis("fragment_length")
    .sort_index()
    .reset_index()
)
distribution["fraction"] = distribution["count"] / distribution["count"].sum()

# Plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 4))

binned = (
    distribution.assign(bin=lambda df: (df["fragment_length"] // bin_width) * bin_width)
    .groupby("bin", as_index=False)["fraction"]
    .sum()
)

ax.bar(binned["bin"], binned["fraction"], width=bin_width * 0.9, color="#1f77b4")
ax.set_xlabel("Fragment length (bp)")
ax.set_ylabel("Fraction of read pairs")
ax.set_title("cfDNA fragment length distribution")
ax.set_xlim(left=0)
plt.show()
