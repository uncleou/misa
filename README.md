# MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
Code for the [ACM MM 2020](https://2020.acmmm.org) paper [MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2005.03545.pdf)

Research context for this fork: [研究方案与当前状态](RESEARCH_CONTEXT.md). AI assistants should first read [AGENTS.md](AGENTS.md).

### CMD 私有信息诊断：`cmd_private_probe_v1`

独立入口 `src/probe_private.py`，冻结已有 CMD 模型，缓存融合前的六个原始
Sigmoid 表示，分别从头训练五组小预测头。只读取已有的 train/dev 数据缓存，
不读取 test.pkl、不重建数据、不调用原 Solver 的训练或测试流程。
原模型、主训练目标及默认配置保持不变。

此版本固定为 **BERT + MISA，hidden_size=128、LSTM、ReLU、原模型 dropout=0.5**，
适用于当前默认结构的 MOSI/MOSEI 回归 checkpoint。模型使用 `eval()` 且全部参数冻结。
预测头为 `Linear(D, H) → ReLU → Linear(H, 1)`，不标准化、不加 dropout。
共享输入始终拼接三个共享表示，不取质心。

| 组名 | 输入 | D | H | 参数量 |
| --- | --- | ---: | ---: | ---: |
| S | 共享 T/A/V | 384 | 128 | 49,409 |
| S_T | 共享 T/A/V + 私有 T | 512 | 96 | 49,345 |
| S_A | 共享 T/A/V + 私有 A | 512 | 96 | 49,345 |
| S_V | 共享 T/A/V + 私有 V | 512 | 96 | 49,345 |
| S_T_A_V | 全部共享和私有 | 768 | 64 | 49,281 |

固定预测头种子 `336,337,338,339,340`；Adam、学习率 `1e-3`、weight decay=0、
batch size=64、最多 100 轮，验证 MSE 连续 10 轮未改善即停止，恢复验证 MSE
最低的预测头后报告指标。五组相同种子的样本打乱顺序一致。
训练损失为 MSE，验证误差按全部样本计算，不平均不同大小 batch 的均值。
固定参数用于初筛，不代表已调优；v1 不提供更改预测头训练协议的 CLI 参数。

从仓库根目录运行（先准备原始 CMD state_dict 和项目原有数据/BERT 环境）：

```bash
python src/probe_private.py \
  --version cmd_private_probe_v1 \
  --data mosi \
  --checkpoint checkpoints/model_mosi_cmd_s336.std \
  --output results/cmd_private_probe_v1/mosi_cmd_s336
```

MOSEI 使用 `--data mosei` 并更换权重及输出目录。`--device auto` 默认有 CUDA 则使用
CUDA，否则 CPU；可指定 `--device cpu` 或 `--device cuda`。单进程运行，不用 torchrun。
特征提取 batch size 默认为 64，可通过 `--extract_batch_size` 调整显存占用。
不传 `--output` 则创建带时间戳的新目录；已有目录会报错，防止覆盖结果。

数据必须已存在于 `datasets/MOSI/` 或 `datasets/MOSEI/`：`train.pkl`、`dev.pkl`、
`embedding_and_mapping.pt`。诊断读取沿用原始预处理和 collate，同时保存经过排序的
样本 ID，保证特征、标签、预测一一对应。数据缓存须为可信的项目文件。
加载权重使用严格参数匹配。旧权重仅包含 state_dict，无法据此验证训练时使用的是
CMD 还是 ATP，需自行核对训练记录；脚本不会把指定 `cmd` 当作历史来源已验证。
非默认架构或激活函数的权重不适用本固定版本（部分差异未必能通过形状检测）。

输出包括：

- `manifest.json`：固定协议、模型配置、checkpoint/数据/特征缓存 SHA-256、运行状态。
- `features.pt`：train/dev 六组特征、标签、样本 ID 和提取元数据；每次新运行重新提取，
  同一次运行的 25 个预测头复用该缓存，不自动复用历史缓存。
- `per_seed.csv`、`summary.csv`、`summary.json`：逐种子结果、均值及样本标准差。
- 每组每种子的 `.pt` 最佳预测头、`_history.json` 训练曲线和 `_dev_predictions.csv`。

`delta_mae_vs_S = MAE(S) - MAE(当前组)`，按相同种子配对，正值表示改善。
MAE/MSE/Corr 使用全部验证样本；`acc2_nonzero`、`f1_nonzero` 排除标签 0 并用 `>0`
二分；`acc2_with_zero`、`f1_with_zero` 保留标签 0 并用 `>=0` 二分。
F1 使用真实标签在前的 weighted F1。常数预测/标签导致相关系数未定义或没有非零标签时，
相应指标记为 JSON null / CSV 空白，汇总同时报告有效种子数，不能当作 0 分。

这是固定 CMD 编码器上的探索性验证集诊断，衡量小预测头可读出的补充预测价值。
参数量接近但隐藏宽度不同；多预测头种子不代表多编码器种子。
验证集参与早停，结果不构成独立测试结论，也不能证明信息论独立或原 Transformer 的
因果分支贡献。先做这五组；打乱私有样本对应关系等进一步对照尚未加入 v1。

离线测试（无真实数据或 BERT 下载，开发验证环境为 PyTorch 2.4.1）：

```bash
python -m unittest discover -s tests -p 'test_private_probe.py' -v
```


<p align="center">
  <img width="600" src="misa-pic.png">
</p>



### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yml
conda activate misa-code
```

### Data Download

- Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Ensure, you can perform ```from mmsdk import mmdatasdk```.    
- Option 1: Download [pre-computed splits](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside ```datasets``` folder.     
- Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.

### Running the code

1. ```cd src```
2. Set ```word_emb_path``` in ```config.py``` to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip).
3. Set ```sdk_dir``` to the path of CMU-MultimodalSDK.
2. ```python train.py --data mosi```. Replace ```mosi``` with ```mosei``` or ```ur_funny``` for other datasets.

### Shared Alignment Loss (extension)

The shared (modality-invariant) branch can be aligned with one of two losses, selected via `--shared_align_loss`:

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--shared_align_loss` | `cmd`, `atp` | `cmd` | Shared alignment loss. `cmd` is the original MISA Central Moment Discrepancy; `atp` aligns unit-normalized shared representations. |
| `--atp_variant` | `text`, `all` | `all` | ATP variant: `text` anchors audio/video to text; `all` aligns all three pairs. |
| `--sim_weight` | float | `1.0` | Weight of the selected shared alignment loss. |
| `--cu_weight` | non-negative float | `0.0` | Sample-centroid uniformity weight. Zero disables CU and preserves existing versions. |
| `--cu_tau` | positive float | `2.0` | Distance scale in the uniformity exponential. |
| `--sent_weight` | non-negative float | `0.0` | Sentiment-conditioned centroid KL weight; exclusive with positive `cu_weight` or `rnc_weight`. |
| `--sent_sigma` | positive float | `1.0` | Continuous-label neighborhood width (in the original label units). |
| `--sent_tau` | positive float | `1.0` | Representation neighborhood temperature; divides squared centroid distance. |

**ATP** (`--shared_align_loss atp`) minimizes the mean squared distance between unit-normalized shared representations. The `--atp_variant` switch selects the alignment target:

- `text` — anchors audio and video to text, aligning only the text-audio and text-video pairs.
- `all` (default) — symmetrically aligns all three pairs (text-audio, text-video, audio-video), averaged with equal weight.

Notes:

- `atp` requires `--use_cmd_sim true` (the default); combining ATP with `--use_cmd_sim false` raises a configuration error. Domain-adversarial alignment remains available with `--shared_align_loss cmd --use_cmd_sim false`.
- The task loss, DiffLoss, reconstruction, model structure and prediction path are unchanged. CU is optional and can also be used with CMD for an independent control.

### ATP-all + centroid uniformity (CU)

For each sample, normalize the three shared representations, average them with
equal weights, then normalize the resulting centroid:

```text
z_m = normalize(h_shared_m), for m in {text, audio, video}
u_i = normalize((z_text_i + z_audio_i + z_video_i) / 3)
L_ATP-all = mean_i (||z_text_i-z_audio_i||² + ||z_text_i-z_video_i||²
                   + ||z_audio_i-z_video_i||²) / 3
L_CU = log(mean_{i<j} exp(-cu_tau * ||u_i-u_j||²))
L_total = L_task + diff_weight * L_diff + sim_weight * L_alignment
          + recon_weight * L_recon + cu_weight * L_CU
```

CU excludes self-pairs and uses a stable log-sum-exp calculation. Gradients flow
through the centroid into all three modalities; the normalization does not
modify the states used by fusion, reconstruction or DiffLoss. A batch with fewer
than two total samples has zero CU loss. With `torchrun`, centroids are gathered
across ranks with autograd, so CU uses the global batch; its backward scaling
accounts for the solver's existing parameter-gradient averaging. No extra
division by the number of GPUs is needed. The existing distributed sampler can
pad an epoch with repeated examples, which remain in the sampled global batch.

CU is normally non-positive: a lower total training loss alone is not evidence
of better prediction. Logs report task, alignment, reconstruction, DiffLoss and
CU separately, plus centroid cosine/variance diagnostics (averaged across
batches). Validation/test `eval_loss` remains task loss only. Shared Sigmoid
outputs still restrict normalized centroids to the positive region of the sphere.

From `src`, select each version explicitly. Distinct names keep checkpoints
separate (choose a new name when repeating a run):

```bash
# Original MISA / CMD
python train.py --data mosi --seed 336 --shared_align_loss cmd --cu_weight 0 --name mosi_cmd_s336

# Earlier text-anchored ATP
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant text --cu_weight 0 --name mosi_atp_text_s336

# ATP-all, without CU
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --cu_weight 0 --name mosi_atp_all_s336

# ATP-all + CU (0.01 is a starting weight, not a tuned optimum)
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --cu_weight 0.01 --cu_tau 2 --name mosi_atp_all_cu001_s336

# Same new variant on four GPUs
torchrun --standalone --nproc_per_node=4 train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --cu_weight 0.01 --cu_tau 2 --name mosi_atp_all_cu001_4gpu_s336
```

The default remains CMD with CU disabled; the existing ATP variant default
remains `all`. For a validation-only CU weight search, compare `0`, `0.01`,
`0.05`, and `0.1` with the same seeds and training setup, then evaluate the
selected setting on the test set. Checkpoints are stored in the repository's
`checkpoints/` directory.

Loss and distributed-gradient checks (CPU, no dataset/model download), from
the repository root:

```bash
python -m unittest discover -s tests -v
```

### ATP-all + sentiment-conditioned centroid relations

This variant replaces CU with the research plan's label-conditioned neighborhood
KL loss, while retaining the existing MISA six-representation Transformer fusion,
task loss, DiffLoss and reconstruction. It does **not** implement the document's
separate centroid-plus-private fusion architecture. No model parameters or
checkpoint tensor shapes are added or changed.

```text
z_m_i = normalize(h_shared_m_i)
c_i = (z_text_i + z_audio_i + z_video_i) / 3  # NO second normalization
q_ij = exp(-abs(y_i-y_j)/sent_sigma) / sum_{k!=i} exp(-abs(y_i-y_k)/sent_sigma)
p_ij = exp(-||c_i-c_j||²/sent_tau) / sum_{k!=i} exp(-||c_i-c_k||²/sent_tau)
L_sent = mean_i sum_{j!=i} q_ij * (log(q_ij) - log(p_ij))
L_total = L_task + diff_weight * L_diff + sim_weight * L_alignment
          + recon_weight * L_recon + sent_weight * L_sent
```

Both distributions exclude self-pairs and normalize over each row's neighbors.
The implementation removes the diagonal before log-softmax to avoid undefined
diagonal KL terms. It sums KL over neighbors and averages over samples, not over
all pair entries. A global batch of fewer than three samples has zero loss:
with two samples each row has just one neighbor and both distributions are `[1]`.

Use real continuous MOSI/MOSEI training labels, without rounding, binning or
rescaling. Labels are detached targets; gradients flow through all three shared
branches. Under `torchrun`, centroids and labels are gathered together in matching
rank/sample order, with centroid gradients preserved and the existing distributed
gradient averaging respected. The sampler's padding behavior described above
also applies here. No labels enter model inference; dev/test prediction and
`eval_loss` remain unchanged. This loss is disabled for UR_FUNNY classification.

Old CU retains its unit-length centroid and its original exponential multiplier
`cu_tau`. The new `sent_tau` is a **divisor** and is not interchangeable with
`cu_tau`. Comparing these variants changes both the neighborhood objective and
the final centroid normalization; a separate controlled normalization ablation
is needed to attribute improvements to the objective alone.

Enable exactly one sample-level constraint through its weight:

| Version | Flags in addition to `--shared_align_loss atp --atp_variant all` |
|---------|----------------------------------------------------------------|
| ATP-all only | `--cu_weight 0 --sent_weight 0` |
| ATP-all + original CU | `--cu_weight 0.01 --sent_weight 0 --cu_tau 2` |
| ATP-all + sentiment relations | `--cu_weight 0 --sent_weight 0.01 --sent_sigma 1 --sent_tau 1` |

At most one of CU, sentiment KL, and RNC weights may be positive; conflicting weights raise a configuration error. Existing
CMD/ATP/CU commands continue to work because `sent_weight` defaults to zero.
The example weight and temperatures below are starting settings, not validated
optima; select them using validation data, keeping the label scale fixed.

From `src`:

```bash
# New variant on one GPU (or CPU)
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --sim_weight 1 --cu_weight 0 --sent_weight 0.01 --sent_sigma 1 --sent_tau 1 --name mosi_atp_all_sent001_sigma1_tau1_s336

# New variant on four GPUs
torchrun --standalone --nproc_per_node=4 train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --sim_weight 1 --cu_weight 0 --sent_weight 0.01 --sent_sigma 1 --sent_tau 1 --name mosi_atp_all_sent001_sigma1_tau1_4gpu_s336
```

For MOSEI, change `--data mosi` to `--data mosei` and use a distinct `--name`.
Training logs show the raw and weighted sentiment KL, nearest-centroid neighbor's
mean absolute label gap, raw centroid variance and mean centroid norm. These are
global-training-batch diagnostics averaged across steps, not held-out quality
estimates. Lower KL alone does not prove prediction improvement or rule out
collapse; compare validation task metrics and representation statistics.

### Citation

If this paper is useful for your research, please cite us at:

```
@article{hazarika2020misa,
  title={MISA: Modality-Invariant and-Specific Representations for Multimodal Sentiment Analysis},
  author={Hazarika, Devamanyu and Zimmermann, Roger and Poria, Soujanya},
  journal={arXiv preprint arXiv:2005.03545},
  year={2020}
}
```

### Contact

For any questions, please email at [hazarika@comp.nus.edu.sg](mailto:hazarika@comp.nus.edu.sg)

### ATP-all + Rank-N-Contrast (RNC)

This independent control replaces the **sample-level** CU/KL term with RNC,
while retaining ATP-all, task supervision, DiffLoss, reconstruction and the
six-representation Transformer fusion. It adds no model parameters.

This is a **single-view centroid adaptation of the RNC loss**, not a reproduction
of the original augmented two-view / representation-pretraining pipeline.
Reference: [Rank-N-Contrast, NeurIPS 2023](https://arxiv.org/abs/2210.01189)
and [official loss implementation](https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py).
No duplicate features are manufactured as a second view.

```text
c_i = (normalize(shared_t_i) + normalize(shared_a_i) + normalize(shared_v_i)) / 3
# NO second normalization; same centroid as the sentiment KL variant.
dy_ij = abs(y_i - y_j)
s_ij = -||c_i-c_j||_2          # rnc_distance=l2 (official metric)
s_ij = -||c_i-c_j||_2^2        # rnc_distance=l2sq (controlled variant)
S_ij = {k != i : dy_ik >= dy_ij}
L_RNC = mean_{i,j!=i} [logsumexp_{k in S_ij}(s_ik/rnc_tau) - s_ij/rnc_tau]
L_total = L_task + diff_weight * L_diff + recon_weight * L_recon
          + sim_weight * L_ATP-all + rnc_weight * L_RNC
```

The candidate itself and all equally distant label neighbors remain in the
denominator. Self-pairs are excluded. The mean is over `N*(N-1)` ordered pairs.
A global batch with fewer than three samples returns a differentiable zero.
All-equal labels do not generally imply zero RNC: if off-diagonal feature
distances are also equal, the loss is `log(N-1)` for `N >= 3`.

| Argument | Default | Description |
| --- | --- | --- |
| `--rnc_weight` | `0.0` | Non-negative finite loss weight; zero disables RNC. |
| `--rnc_tau` | `2.0` | Positive finite temperature dividing feature distance. |
| `--rnc_distance` | `l2` | `l2` for the official Euclidean metric; `l2sq` for squared distance, matching the KL metric. |

At most one of `cu_weight`, `sent_weight`, `rnc_weight` may be positive.
This first version intentionally excludes RNC+KL joint training. RNC uses raw
continuous MOSI/MOSEI training labels and is rejected for UR_FUNNY. Existing
commands remain compatible because `rnc_weight` defaults to zero.

Like sentiment KL, RNC uses differentiably gathered **global-batch** centroids
and correspondingly ordered detached labels under `torchrun`. The loss needs no
extra `world_size` division with the solver's parameter-gradient averaging.
Existing distributed-sampler padding still applies. Neither RNC nor labels are
added to inference; dev/test loss and model selection remain task-based.

Implementation sorts label gaps per row, computes suffix `logcumsumexp`, and
uses the first index of each tie group to obtain exact denominators. It avoids
an `N x N x N` mask and uses quadratic pairwise storage. Logs identify the
metric/temperature/weight and report raw/weighted RNC, nearest-neighbor label
gap, centroid variance and centroid norm (diagnostics averaged across batches).

From `src`, using distinct names for each run:

```bash
# ATP-all without sample-level relations
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --cu_weight 0 --sent_weight 0 --rnc_weight 0 --name mosi_atp_all_only_s336

# Independent RNC control: official Euclidean distance
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --sim_weight 1 --cu_weight 0 --sent_weight 0 --rnc_weight 0.01 --rnc_tau 2 --rnc_distance l2 --name mosi_atp_all_rnc_l2_s336

# Squared-distance control for comparison with sentiment KL
python train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --sim_weight 1 --cu_weight 0 --sent_weight 0 --rnc_weight 0.01 --rnc_tau 2 --rnc_distance l2sq --name mosi_atp_all_rnc_l2sq_s336

# Same RNC version on four GPUs
torchrun --standalone --nproc_per_node=4 train.py --data mosi --seed 336 --shared_align_loss atp --atp_variant all --sim_weight 1 --cu_weight 0 --sent_weight 0 --rnc_weight 0.01 --rnc_tau 2 --rnc_distance l2 --name mosi_atp_all_rnc_l2_4gpu_s336
```

The example weight and temperature are starting settings, not tuned results.
Give RNC and KL comparable validation tuning budgets; equal numerical weights
or temperatures do not imply equal constraint strength. Keep representation,
normalization, seeds, effective batch size and evaluation protocol fixed.

Run `python -m unittest discover -s tests -v` from the repository root to check
formula values and gradients against an explicit reference, label-distance ties,
permutation invariance, both distances, degenerate batches, existing routes,
real solver training/evaluation with a tiny model, and two-process CPU Gloo
value/parameter-gradient equivalence for unequal local batch sizes. The Gloo
tests require permission to create local processes and communication resources.
