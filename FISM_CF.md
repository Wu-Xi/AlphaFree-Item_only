# FISM-BCE：独立的 item-based CF 基线

实现位于 `cf/fism.py`，接入 `train_cf.py` 和 `search_cf.py`。
它的打分、BCE 损失和正则与本地 `MixGCF/modules/FISM.py` 的设定一致；
改用紧凑 CSR/embedding_bag 历史聚合，避免按最长历史填充。
不依赖 MixGCF、ItemGCN 或 AlphaFree 文本编码器。
这是 FISM-BCE，不是 FISM-BPR，也不声称复现论文基线的完整训练配方。

## 与当前 item-only 研究的关系

FISM 自身就用历史物品生成用户向量，无可学习 user embedding table。
它作为独立 `native` 基线加入，用于比较“传统 item-based CF”与
“把已有高性能模型改成 item-only”的结果，而非把它复制成两份 original/item-only。
它学习两个 item 表（source P、target Q），不进行 LightGCN 传播，不显式构建 item-item 图。
因此参数量不能简单等同于只有一个 item 表的 item-only LightGCN。

默认保留一个用户标量偏置 b_u，这不是用户向量表，但确实是用户专属可学习参数。
如果实验要求完全没有用户专属参数，使用 `--fism-user-bias 0`，并单独标注这一变体。
默认参数量为 `2 * n_items * dim + n_items + n_users`；禁用用户 bias 后减去 n_users。

## 打分与损失

令 H_u 为用户在 train.txt 中的历史，候选物品为 i：

```text
H_ui = H_u \ {i}
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
FISM 不读取 `--b-mode/--b-a/--b-b`、`--context-hops`、`--n-negs` 或 StabCF 的 `--alpha`。
使用独立的 `--fism-*` 参数，避免混淆。

## 搜索空间

| 参数 | 取值 |
|---|---|
| fism-alpha | 0、0.5、1 |
| fism-reg | 0.0001、0.001、0.01 |
| fism-negatives | 固定 4 |
| fism-user-bias | 固定 1 |
| dim / lr / batch-size | 64 / 0.001 / 2048 |
| weight-decay | 0 |

每数据集、每 seed 共 9 组，属于新增起始网格，不是已验证的最优范围。
不与 29 种 B 交叉。前四类模型的 750 组保持原样，默认五类合计 759 组。
只用 validation Recall@20 早停与选参；搜索不计算测试指标。
评估仍使用当前框架的 AlphaFree 全目录候选屏蔽协议。

## GPU 服务器命令

在服务器 AlphaFree 目录、已安装 CUDA 版 PyTorch 的环境中执行。

```bash
# 预览 FISM 的 9 组，不训练
python search_cf.py --datasets Movie --models fism

# 先验证一组：1 个优化步、前 32 位验证用户；不得用作正式结果
python search_cf.py --datasets Movie --models fism --device cuda:0 --smoke --output weights/fism_movie_smoke --execute

# 正式 FISM 搜索：使用新的输出目录
python -u search_cf.py --datasets Movie --models fism --device cuda:0 --output weights/fism_movie_v1 --execute

# 查看进度及验证选参结果
python search_cf.py --output weights/fism_movie_v1 --summarize

# 五类模型完整搜索，4 张 GPU 后台分片
bash start_cf_joint.sh Movie 0,1,2,3 weights/joint_movie_with_fism

# 单次训练（不做搜索）；此示例只评估验证集
python train_cf.py --dataset Movie --model fism --variant native --device cuda:0 --fism-alpha 0.5 --fism-reg 0.001 --validation-only
```

`--datasets all` 可搜索七个数据集；必须提前准备每个数据集的 cf_data 三份划分文件。
本次代码和配置发生变化，不要向旧 750 组搜索目录强行续写，也不要修改旧 manifest。
仅补 FISM 时单独指定 `--models fism` 和新的输出目录，不必重跑其他模型。
同一配置再次执行会跳过完成任务；这不是 epoch 级恢复训练。

## 验证范围

自动化测试覆盖与本地旧实现的打分、损失、梯度一致性，全目录与逐对打分一致性，
目标排除、空历史、单物品历史、真实 item 0、可关闭用户 bias、checkpoint、验证评估和搜索计数。
CUDA 测试仅在可用 GPU 环境执行。本机 CPU 短跑不代表 GPU 已验证或模型已收敛。

2026-10-20 验证记录：30 项自动化测试，28 项通过、2 项因无 CUDA 跳过。
真实 Movie 数据上 9 组 FISM 调度短跑全部通过，再次运行全部正确跳过；
记录在 `weights/fism_verify_20261020/`。
另完成关闭用户 bias 的训练、checkpoint 重载及验证/测试评估短跑，
记录在 `weights/fism_no_user_bias_verify_20261020/`。尚未启动正式 GPU 搜索。
