# PD_liquid_biospy

## metilene_like.py 使用说明

基于 metilene 的原理，我们提供了 `metilene_like.py`，用于从每个样本的 bedGraph CpG 甲基化文件中检测去 novo 差异甲基化区域（DMR）。脚本引入覆盖度加权、递归二分分段及 Wald z 检验，力求与 metilene 保持一致。

### 依赖

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`

可以使用 `pip install numpy pandas scipy` 安装依赖。

### 输入数据

1. **bedGraph 文件**：每个样本一个文件（至少包含 `chrom`, `start`, `end`, `beta` 四列）。`beta` 为 0~1 之间的甲基化比例；若为 0~100 会自动换算。为得到 metilene 类似的结果，建议额外提供覆盖度列：
   - 5 列格式：`chrom start end beta coverage`
   - 6 列格式：`chrom start end beta methylated_count unmethylated_count`
   缺失覆盖度时将退化为简单平均，不再使用覆盖度加权或 Wald 检验。
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
  --max-qvalue 0.1 \
  --require-coverage
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
| segment_score | 递归分段时的加权得分（对应 metilene 的评分） |
| total_cov_group_a / total_cov_group_b | 两组在该区域的覆盖度总和（无覆盖度时为 NaN） |
| wald_z | 基于覆盖度的 Wald z 统计量（无覆盖度时为 NaN） |
| mannwhitney_u | Mann-Whitney U 统计量 |
| p_value / q_value | p 值与 BH 校正后的 q 值 |

若没有区域满足阈值，可添加 `--allow-empty-output` 以输出空文件并避免报错。

## dmr_methylation_summary.py 使用说明

给定 metilene 等工具输出的 DMR 列表以及各样本的 CpG bedGraph（含 coverage 与 meth_read），`dmr_methylation_summary.py` 用覆盖度加权方式计算每个 DMR 在每个样本中的平均甲基化水平。

### 额外依赖

- 与前述脚本相同的 Python 环境即可（numpy / pandas）。

### 输入

1. **DMR 文件**：至少包含 `chrom`, `start`, `end` 列，`start/end` 推荐使用 0-based 坐标。可通过 `--chrom-column` 等参数调整列名；若有 ID 列，可用 `--dmr-id-column` 指定。
2. **设计表 design.tsv**：列示样本名称和对应 bedGraph 路径，与 `metilene_like.py` 的设计表相同格式。
3. **bedGraph**：必须包含 `chrom start end coverage meth_read` 五列（无表头）。覆盖度为总 reads，`meth_read` 为甲基化 reads。

### 使用示例

```
python dmr_methylation_summary.py dmrs.tsv design.tsv \
  --output dmr_beta.tsv \
  --dmr-id-column id \
  --add-coverage
```

### 输出

TSV 文件包含以下列：

| 列名 | 含义 |
| ---- | ---- |
| dmr_id | DMR 标识（若未提供则为 `DMR_#`） |
| chrom / start / end | DMR 坐标 |
| `<sample>` | 对应样本的覆盖度加权甲基化水平（meth / coverage） |
| `<sample>_coverage` | （可选）该样本在此 DMR 的覆盖度总和，需 `--add-coverage` |

若某个样本在 DMR 中 coverage 总和为 0，则对应甲基化水平为 `NaN`。可直接将输出用于后续绘图或差异可视化。

### Notebook 调用示例

脚本同时提供无需命令行的函数接口，可直接在 Jupyter 中使用：

```python
from pathlib import Path
import pandas as pd
from dmr_methylation_summary import compute_dmr_methylation_matrix

dmr_df = pd.read_csv("dmrs.tsv", sep="\t")
sample_paths = {
    "sample01": Path("data/sample01.bedgraph"),
    "sample02": Path("data/sample02.bedgraph"),
}

matrix = compute_dmr_methylation_matrix(
    dmr_df,
    sample_paths,
    dmr_id_column="id",  # 若 DMR 表有 ID 列
    add_coverage=True,
)
matrix.head()
```

生成的 `matrix` 即为 DMR × 样本的加权甲基化矩阵。