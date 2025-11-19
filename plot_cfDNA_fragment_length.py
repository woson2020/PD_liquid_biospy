#!/usr/bin/env python3
"""
超直观版本：只需要把 bam_path 改成自己的 BAM 文件，直接在 Jupyter 里 `%run plot_cfDNA_fragment_length.py` 就能画出片段长度分布。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pysam
from tqdm import tqdm

# 1. 把这里改成你的 BAM 路径，例如 Path("/data/sample.bam")
bam_path = Path("xxx")

# 2. 是否要求 read1/read2 完全从同一位置开始（一般 cfDNA 需要）
require_same_start = True

# 3. 只看特定长度区间，可按需调节
min_length = 0
max_length = 600  # 如果不想限制，改成 None

# 4. 如果 BAM 很大，可以打开进度条方便观察
show_progress = True

# --- 以下代码保持不动即可 ---
if not bam_path.exists():
    raise FileNotFoundError(f"找不到 BAM 文件：{bam_path}")

fragment_lengths = []

with pysam.AlignmentFile(bam_path, "rb") as bam:
    iterator = bam.fetch(until_eof=True)
    if show_progress:
        iterator = tqdm(iterator, desc="扫描 BAM", unit="read")

    for read in iterator:
        # 只检查 read1，避免重复计算
        if not read.is_paired or not read.is_read1:
            continue
        # R1 或 R2 没有比对成功就跳过
        if read.is_unmapped or read.mate_is_unmapped:
            continue
        # 去掉次要/补充/重复的比对
        if read.is_secondary or read.is_supplementary or read.is_duplicate:
            continue
        # 不是 proper pair 的不要
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

        fragment_lengths.append(frag_len)

if not fragment_lengths:
    raise RuntimeError("没有找到符合条件的片段，请检查 BAM 或过滤条件。")

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 4))
plt.hist(fragment_lengths, bins=range(0, max(fragment_lengths) + 5, 5), color="#1f77b4")
plt.xlabel("Fragment length (bp)")
plt.ylabel("Read count")
plt.title("cfDNA fragment length distribution")
plt.xlim(left=0)
plt.show()
