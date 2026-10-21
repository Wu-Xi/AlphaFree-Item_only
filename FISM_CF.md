# FISM-BCE：独立的 item-based CF 基线

实现位于 `cf/fism.py`，接入 `train_cf.py` 和 `search_cf.py`。
它的打分、BCE 损失和正则与本地 `MixGCF/modules/FISM.py` 的设定一致；
改用紧凑 CSR/embedding_bag 历史聚合，避免按最长历史填充。
不依赖 MixGCF、ItemGCN 或 AlphaFree 文本编码器。
这是 FISM-BCE，不是 FISM-BPR，也不声称复现论文基线的完整训练配方。

## 当前运行流程（2026-10-21）

Baby / Steam 的 60 个任务统一使用 `fism_reg=1e-6`，保留重复消费协议。
专用配置 `configs/fism_baby_steam.json` 已启用训练后测试、保存 TXT、删除 checkpoint；
直接通过该配置运行搜索和通过启动脚本运行，行为相同。

每个任务独立执行：训练 → 按验证 Recall@20 选最佳 epoch → 加载该最佳权重 →
立即完整评估 test → 保存 `result.json` 与 `training.txt` 并刷新到磁盘 → 删除本次 `best.pt`。
测试不参与早停或最佳 epoch 选择。测试或结果保存失败时不删除权重。
正式运行的 `eval_users=0`，覆盖所有有测试真值的用户；`--smoke` 仍是截断诊断，不能作为正式结果。

每个模型的完整文本记录位于：

```text
<输出目录>/<数据集>/<任务ID>/attempt_<时间>/run/training.txt
```

TXT 包含配置、每个 epoch 的汇总损失、验证指标、最佳 epoch、最终测试 Recall/NDCG/HR@20、
测试用户数和权重删除记录。取消 `epoch 2 step 150 loss ...` 这类 step 日志，仍保留 epoch 汇总。
`config.json`、`history.jsonl`、`result.json`、`status.json` 继续保留，便于之后汇总结果。
结构化结果还保留删除前 checkpoint 的 SHA256 和数据哈希。
已成功测试并删除权重的任务仍计为 completed，重新启动会跳过，不会因此重新训练。

只删除新流程当前任务的 `best.pt`，不会清理旧实验目录。已删除的 checkpoint 无法再补跑其他 K 值；
后面的 `eval_fism.py` 重载测试说明仅适用于仍保留权重的旧批次。

## 与当前 item-only 研究的关系

FISM 自身就用历史物品生成用户向量，无可学习 user embedding table。
当前提供 `original` 和 `item-only` 两个实验版本；后者是替换历史聚合方式的 FISM-B，
不是再次删除用户向量表。旧 `native` 命令保留为 `original` 的兼容别名。
它学习两个 item 表（source P、target Q），不进行 LightGCN 传播，不显式构建 item-item 图。
因此参数量不能简单等同于只有一个 item 表的 item-only LightGCN。

默认保留一个用户标量偏置 b_u，这不是用户向量表，但确实是用户专属可学习参数。
如果实验要求完全没有用户专属参数，使用 `--fism-user-bias 0`，并单独标注这一变体。
默认参数量为 `2 * n_items * dim + n_items + n_users`；禁用用户 bias 后减去 n_users。

## 打分与损失

令 H_u 为用户在 train.txt 中的历史多重集（重复记录保留），候选物品为 i：

```text
H_ui = remove ALL occurrences of i from H_u
p_ui = sum(P_j for j in H_ui) / max(1, |H_ui|)^alpha
s_ui = dot(p_ui, Q_i) + b_i + b_u

L_data = mean BCEWithLogits(s_ui, y_ui)
L_reg  = lambda * (||P||_F + ||Q||_F)
L      = L_data + L_reg
```

正则是两个表的全局、未平方范数，不是常见的平方 L2，也不乘通用 `--l2`。
优化器仍接受 `--weight-decay`，默认及搜索中为 0，避免额外施加正则。
空历史、移除目标后的空历史均返回零向量；item 0 是真实物品，不是 padding。
训练和全目录打分均进行候选特定的目标排除。

每轮打乱所有训练正交互；每条正例随机抽取 4 个负例（有放回，只排除训练正例）。
每个 batch 包含 `--batch-size` 条正交互，BCE 对其 5 倍数量的带标签样本求均值。
例如 batch-size=2048，对应 10,240 个带标签样本；这与本地旧 runner 按带标签样本计 batch 的约定不同。
本次对齐的是同一输入样本的打分/损失/梯度，不是旧 runner 的整个优化轨迹。
original 不读取 `--b-*`；item-only 通过 `--b-mode/--b-a/--b-b` 替换上述归一化，
不再叠加 `--fism-alpha`。两个版本都不使用 `--context-hops`、`--n-negs` 或 StabCF 的 `--alpha`。

## 29 种用户构造

令 d_j 为物品在完整训练集中的交互次数（含重复），H_ui 仍先排除当前目标的所有重复项：

```text
degree:        p_ui = |H_ui|^(-a) * sum(d_j^(-b) * P_j for j in H_ui)
weighted_mean: p_ui = sum(d_j^(-b) * P_j for j in H_ui) / sum(d_j^(-b) for j in H_ui)
```

空历史返回零向量。degree 的 a、b 均取 `{0, 0.25, 0.5, 0.75, 1}`，共 25 组；
weighted_mean 的 b 取 `{0.25, 0.5, 0.75, 1}`，共 4 组。
sum/mean/sqrt/sym 是 25 组中的命名点，不额外计数。
用户度/权重和在目标排除后计算，物品度始终来自完整训练集。
因此 sqrt（a=.5,b=0）与 original alpha=.5 数学等价，保留作一致性对照。
这是对 B 的候选特定实现，不是给 FISM 加 GNN 或 item-item 图。

## Steam 重复记录：保留次数

用户选择保留重复记录，按次数建模。专用启动器已加入 `--duplicate-items count`：

- 原始文件不改写；正例列表保留重复，所以重复 m 次的 user-item 每轮出现 m 次。
- 稀疏交互矩阵 R_ui=m，历史向量按 m 加权，用户/物品度均累计事件次数。
- 打分候选 i 时，移除历史中 i 的全部 m 次记录，再计算剩余长度/权重和。
- original 与 29 个 item-only 完全共享上述规则。
- 验证/测试文件也保留原始列表，但 Recall/NDCG 的真值按唯一物品计数，
  与 AlphaFree **C++** 评估器的 `unordered_set` 语义一致；
  AlphaFree 的 Python 后端在重复真值的分母处理上不同，不混称两者完全相同。

已核查本地 Steam 重复记录额外出现次数：train=84,734、valid=53,179、test=53,648。
这些次数仅记录为元数据，不删除。无该选项的其他实验仍默认遇到重复记录即报错。
对有重复项的数据，这是显式的计数加权 FISM 扩展；不声称旧 MixGCF 的重复项处理已完全对齐。

## 搜索空间

| 参数 | 取值 |
|---|---|
| fism-alpha | original 固定 0.5；item-only 用 B 替换，不叠加 |
| fism-reg | 固定 0.000001（1e-6） |
| fism-negatives | 固定 4 |
| fism-user-bias | 固定 1 |
| dim / lr / batch-size | 64 / 0.001 / 2048 |
| weight-decay | 0 |

每数据集、每 seed 为 1 original + 29 item-only，共 30 次独立训练。
Baby + Steam 默认 seed=2025，共 60 次；无需先选 original 配方或继承其权重。
最多 1000 轮、每 5 轮验证、连续 20 次验证无提升早停。
前四类模型的 750 组保持原样，默认五类合计 780 组。
只用 validation Recall@20 早停与选参；Baby/Steam 专用配置对每个最佳 checkpoint 立即计算测试指标。
通用 `cf_joint_search.json` 中 FISM 的正则同样改为 1e-6，但通用搜索默认仍只验证并保留权重；
可通过 `--test-after-train --delete-checkpoint-after-test` 显式启用相同的测试与清理流程。
评估仍使用当前框架的 AlphaFree 全目录候选屏蔽协议。

## GPU 服务器命令

在服务器 AlphaFree 目录、已安装 CUDA 版 PyTorch 的环境中执行。

```bash
# 预览 Baby + Steam 的 60 个任务，不训练
python search_cf.py --config configs/fism_baby_steam.json --datasets Baby Steam --models fism --duplicate-items count

# 先验证一组：1 个优化步、前 32 位验证用户；不得用作正式结果
python search_cf.py --config configs/fism_baby_steam.json --datasets Baby Steam --models fism --duplicate-items count --device cuda:0 --smoke --output weights/fism_baby_steam_reg1e-6_smoke_v2 --execute

# 正式多 GPU 搜索：4 张卡，每张卡同时 3 个进程，共 12 个后台 worker
bash start_fism_baby_steam.sh 0,1,2,3 weights/fism_baby_steam_reg1e-6_v2 3

# 例如只使用 GPU 2、5
# bash start_fism_baby_steam.sh 2,5 weights/fism_baby_steam_reg1e-6_v2 3

# 查看进度及验证选参结果
python search_cf.py --output weights/fism_baby_steam_reg1e-6_v2 --summarize

# 单次训练：训练后测试、保存 TXT，然后删除本次 checkpoint
python train_cf.py --dataset Baby --model fism --variant original --duplicate-items count --device cuda:0 --fism-alpha 0.5 --fism-reg 1e-6 --log-every 0 --delete-checkpoint-after-test
```

先将本次代码和配置同步到服务器，再在已激活的 CUDA 环境运行。
`CF_PYTHON=/absolute/path/to/python bash start_fism_baby_steam.sh ...` 可指定解释器。
启动器第三个参数是每卡并行进程数，默认 3；卡号列表不重复填写。
启动器先检查每张卡、冻结清单，再把 60 个任务分配到所有 worker；不是单模型 DDP。
四卡 × 每卡 3 进程 = 12 个 worker，每个 worker 顺序处理 5 个任务，每卡共 15 个任务。
每个 worker 对当前模型完成训练、测试、保存结果、删除权重后，再处理下一个模型。
第三个参数设为 1 可恢复每卡一个进程；总 worker 数不能超过 60，各任务耗时可能不同。
日志在输出目录的 `scheduler_logs/`；启动成功不代表任务已完成。
再次执行相同启动命令会跳过完成任务。失败任务需检查日志后用
`search_cf.py` 的 `--retry-failed` 显式重试；不要在已有进程运行时重复启动。
默认短跑只执行一个任务，不能作为正式结果。专用配置的正式任务均在训练结束后测试；
summary 仍只根据验证 Recall@20 选择配置，并展示对应任务已经保存的测试指标。

`--datasets all` 可搜索七个数据集；必须提前准备每个数据集的 cf_data 三份划分文件。
本次代码和配置发生变化，不要向旧 750 组搜索目录强行续写，也不要修改旧 manifest。
仅补 FISM 时单独指定 `--models fism` 和新的输出目录，不必重跑其他模型。
同一配置再次执行会跳过完成任务；这不是 epoch 级恢复训练。

## 旧批次：对保留权重的已选模型进行最终测试

新增 `eval_fism.py`，直接加载搜索保存的 `best.pt`，不重新训练。
先按验证集 Recall@20 在每个 dataset / variant / seed 组内选定配置，
再对选定模型计算完整测试集指标。默认 original 与 item-only 分别选取，
Baby + Steam、单 seed 的完整搜索通常得到 4 行测试结果。
每组必须全部完成；smoke 结果不参与最终测试。

```bash
# 在 AlphaFree 目录中，先查看将测试哪些 checkpoint（不计算测试指标）
python eval_fism.py --search-output weights/fism_baby_steam_v1 --dry-run

# 加载最佳权重，计算测试集 Recall/NDCG/HR@20
CUDA_VISIBLE_DEVICES=0 python eval_fism.py \
  --search-output weights/fism_baby_steam_v1 --device cuda:0

# 可选：同时报告 @10、@20；显存不足时减小 --eval-batch-size
# python eval_fism.py --search-output weights/fism_baby_steam_v1 \
#   --device cuda:0 --top-k 10 20 --eval-batch-size 32
```

默认保存为搜索目录下 `final_test.json` 和 `final_test.csv`；
包含选定 B、seed、最佳 epoch、验证选参分数、测试指标与 checkpoint 路径。
不修改训练的 `result.json`、`best.pt` 或搜索清单。
若数据移动了位置，使用 `--data-root /path/to/data`；脚本核对三份划分的哈希，
并从原训练配置恢复聚合方式、重复计数策略、用户偏置等参数。
测试沿用当前 AlphaFree 协议：屏蔽 train+valid，并保留当前 test 真值。
`--datasets Baby` 或 `--variants original` 可只评估指定组；
不会根据测试集效果重新选择 B、epoch 或 seed。
先同步此脚本到保存正式搜索结果的服务器；本机已有 smoke 权重不能替代正式结果。

## 旧批次：测试全部保留权重的配置并导出 Excel

使用 `--all-models` 时，按 manifest 中每个已完成正式任务当前有效的 `best.pt` 逐个测试，
包括未被验证集选中的配置；不是仅测试 4 个优胜模型，也不会重复测试废弃的重试 attempt。
Baby + Steam、seed=2025 的完整清单为 60 个模型，Excel 对应 60 行。
清单中存在未完成任务或 smoke 结果时会在测试前报错，避免静默遗漏。

同步 `eval_fism.py` 和 `export_fism_xlsx.mjs` 到服务器 AlphaFree 目录后执行：

```bash
# 服务器没有 Artifact Tool 运行时时，Excel 自动使用 openpyxl
python -m pip install openpyxl

# 预览全部待测试模型，正式清单应显示 Checkpoints to test: 60
python eval_fism.py --search-output weights/fism_baby_steam_v1 --all-models --dry-run

# 依次评估全部模型并生成一个 Excel 文件；不重新训练
CUDA_VISIBLE_DEVICES=0 python -u eval_fism.py \
  --search-output weights/fism_baby_steam_v1 --all-models --device cuda:0
```

默认生成同目录的 `all_test.xlsx`、`all_test.csv`、`all_test.json`。
Excel 每行包含 dataset、variant、B、seed、最佳 epoch、测试 Recall/NDCG/HR、
验证指标、是否为验证集选中配置、聚合参数、训练超参数及 checkpoint 路径。
指标为数值单元格，带筛选和冻结表头，按清单顺序呈现全部配置。
`--top-k 10 20` 可同时输出两个截断位置；`--eval-batch-size 32` 可降低显存占用。

每完成一个模型就更新 JSON；若个别模型评估失败，会继续其余模型，
并在 Excel 的 status/error 列记录失败，指标留空，最后返回非零退出状态。
训练的 result.json、权重和 manifest 不会改写。
若只有 Excel 导出依赖缺失或需要重新生成表格，安装依赖后运行以下命令，**不重复测试**：

```bash
python eval_fism.py --search-output weights/fism_baby_steam_v1 --all-models --export-only
```

JSON 若来自中断运行，仅包含已尝试的模型；导出前核对行数和 status 列，
不能把不足 60 行或存在 failed 的文件当成完整测试报告。

## 验证范围

自动化测试覆盖与本地旧实现的打分、损失、梯度一致性，全目录与逐对打分一致性，
目标排除、空历史、单物品历史、真实 item 0、可关闭用户 bias、checkpoint、验证评估和搜索计数。
CUDA 测试仅在可用 GPU 环境执行。本机 CPU 短跑不代表 GPU 已验证或模型已收敛。

当前 1+29 / 保留重复次数版本（2026-10-20）：
50 项自动化测试，46 项通过，4 项因无 CUDA 跳过。
覆盖全部 29 种聚合的显式重复历史参考打分、反向传播、sqrt/original 损失与梯度一致性、
重复交互采样/计数矩阵、搜索协议冻结及恢复检查。
Baby、Steam 各完成 original、sqrt、weighted_mean(b=.25) 的 1 步训练及 32 用户验证，
共 6 个短跑；两项重复调用正确跳过。记录在
`weights/fism_baby_steam_count_verify_20261020/`。
多 GPU 启动脚本通过 Bash 语法检查，尚未在本机执行 CUDA 训练。

历史验证记录（旧 9 配方 native 网格，已由当前 1+29 方案替代）：30 项自动化测试，28 项通过、2 项因无 CUDA 跳过。
真实 Movie 数据上 9 组 FISM 调度短跑全部通过，再次运行全部正确跳过；
记录在 `weights/fism_verify_20261020/`。
另完成关闭用户 bias 的训练、checkpoint 重载及验证/测试评估短跑，
记录在 `weights/fism_no_user_bias_verify_20261020/`。尚未启动正式 GPU 搜索。
