# FISM-BCE：独立的 item-based CF 基线

实现位于 `cf/fism.py`，接入 `train_cf.py` 和 `search_cf.py`。
它的打分、BCE 损失和正则与本地 `MixGCF/modules/FISM.py` 的设定一致；
改用紧凑 CSR/embedding_bag 历史聚合，避免按最长历史填充。
不依赖 MixGCF、ItemGCN 或 AlphaFree 文本编码器。
这是 FISM-BCE，不是 FISM-BPR，也不声称复现论文基线的完整训练配方。

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
| fism-reg | 固定 0.001 |
| fism-negatives | 固定 4 |
| fism-user-bias | 固定 1 |
| dim / lr / batch-size | 64 / 0.001 / 2048 |
| weight-decay | 0 |

每数据集、每 seed 为 1 original + 29 item-only，共 30 次独立训练。
Baby + Steam 默认 seed=2025，共 60 次；无需先选 original 配方或继承其权重。
最多 1000 轮、每 5 轮验证、连续 20 次验证无提升早停。
前四类模型的 750 组保持原样，默认五类合计 780 组。
只用 validation Recall@20 早停与选参；搜索不计算测试指标。
评估仍使用当前框架的 AlphaFree 全目录候选屏蔽协议。

## GPU 服务器命令

在服务器 AlphaFree 目录、已安装 CUDA 版 PyTorch 的环境中执行。

```bash
# 预览 Baby + Steam 的 60 个任务，不训练
python search_cf.py --config configs/fism_baby_steam.json --datasets Baby Steam --models fism --duplicate-items count

# 先验证一组：1 个优化步、前 32 位验证用户；不得用作正式结果
python search_cf.py --config configs/fism_baby_steam.json --datasets Baby Steam --models fism --duplicate-items count --device cuda:0 --smoke --output weights/fism_baby_steam_smoke --execute

# 正式多 GPU 搜索：可自由指定卡号，一个进程/卡；后台运行
bash start_fism_baby_steam.sh 0,1,2,3 weights/fism_baby_steam_v1

# 例如只使用 GPU 2、5
# bash start_fism_baby_steam.sh 2,5 weights/fism_baby_steam_v1

# 查看进度及验证选参结果
python search_cf.py --output weights/fism_baby_steam_v1 --summarize

# 单次训练（不做搜索）；此示例只评估验证集
python train_cf.py --dataset Baby --model fism --variant original --duplicate-items count --device cuda:0 --fism-alpha 0.5 --fism-reg 0.001 --validation-only
```

先将本次代码和配置同步到服务器，再在已激活的 CUDA 环境运行。
`CF_PYTHON=/absolute/path/to/python bash start_fism_baby_steam.sh ...` 可指定解释器。
启动器先检查每张卡、冻结清单，再把 60 个任务轮流分配到各 GPU；不是单模型 DDP。
四卡各 15 个任务，两卡各 30 个任务；各任务耗时可能不同。
日志在输出目录的 `scheduler_logs/`；启动成功不代表任务已完成。
再次执行相同启动命令会跳过完成任务。失败任务需检查日志后用
`search_cf.py` 的 `--retry-failed` 显式重试；不要在已有进程运行时重复启动。
默认短跑只执行一个任务，不能作为正式结果。调参报告只含验证指标，最终测试另行进行。

`--datasets all` 可搜索七个数据集；必须提前准备每个数据集的 cf_data 三份划分文件。
本次代码和配置发生变化，不要向旧 750 组搜索目录强行续写，也不要修改旧 manifest。
仅补 FISM 时单独指定 `--models fism` 和新的输出目录，不必重跑其他模型。
同一配置再次执行会跳过完成任务；这不是 epoch 级恢复训练。

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
