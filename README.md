# PD_liquid_biospy

## metilene_like.py 使用说明

基于 metilene 的原理，我们提供了 `metilene_like.py`，用于从每个样本的 bedGraph CpG 甲基化文件中检测去 novo 差异甲基化区域（DMR）。

### 依赖

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`

可以使用 `pip install numpy pandas scipy` 安装依赖。

### 输入数据

1. **bedGraph 文件**：每个样本一个文件（至少包含 `chrom`, `start`, `end`, `beta` 四列）。`beta` 为 0~1 之间的甲基化比例。
2. **设计表（design.tsv）**：包含样本分组信息，支持 TSV 或 CSV。例如：

```
sample	group	path
sample01	case	data/sample01.bedgraph
sample02	case	data/sample02.bedgraph
sample03	control	data/sample03.bedgraph
sample04	control	data/sample04.bedgraph
```

路径可以是相对设计表的相对路径或绝对路径。

### 运行示例

```
python metilene_like.py design.tsv \
  --output dmr.tsv \
  --min-cpgs 5 \
  --min-diff 0.2 \
  --max-gap 500 \
  --max-qvalue 0.1
```

主要参数说明：

- `--min-cpgs`：候选区域至少包含的 CpG 数量。
- `--min-diff`：纳入区域的单个 CpG 组间甲基化均值差阈值。
- `--max-gap`：连续 CpG 之间允许的最大距离（bp）。
- `--max-qvalue`：可选的 FDR 阈值；未指定时输出全部候选区域。
- `--drop-na`：若指定，将在分段前删除任意样本缺失值的 CpG。
- `--group-order`：显式指定两组名称，决定差值的正负方向。
  
输出 `dmr.tsv` 包含以下列：

| 列名 | 含义 |
| ---- | ---- |
| chrom | 染色体 |
| start/end | 区域起止坐标 |
| num_cpgs | 区域内 CpG 数量 |
| mean_beta_group_a / mean_beta_group_b | 区域内两组样本平均甲基化水平（按样本均值） |
| mean_diff / abs_diff | 组间甲基化水平差 |
| direction | `hyper` 表示组 A 相对组 B 超甲基化，`hypo` 反之 |
| mannwhitney_u | Mann-Whitney U 统计量 |
| p_value / q_value | p 值与 BH 校正后的 q 值 |

若没有区域满足阈值，可添加 `--allow-empty-output` 以输出空文件并避免报错。