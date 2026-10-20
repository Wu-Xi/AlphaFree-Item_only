# 四类模型的调参组合

> 此文件是先前的分阶段搜索草案，不再作为当前实验方案。
> 当前采用模型参数网格 × 29 种 B 的完整联合搜索，见 [CF_JOINT_SEARCH.md](CF_JOINT_SEARCH.md)。

本方案依据当前本地实现设计，是待验证的搜索空间，不是已找到的最优参数，
也不是声称与原论文完全相同的搜索范围。不要直接把其他数据集的最优配置当成 Movie 最优。
搜索空间的机读说明在 `configs/cf_search_space.json`，当前训练脚本不直接读取此 JSON；
运行时使用下列 CLI 参数。本文不自动启动任何搜索。

## 1. 公平比较的设置

四类模型均运行 original / item-only。先固定 item-only 的 B 为 sym，
对同一模型的两个版本提供相同搜索空间、训练预算、seed 和评估频率，各自在验证集选最优。
再单独报告 B 的消融，避免把 item-only 额外调参收益混入“只去用户表”的结论。

同时保留一组受控比较：采用 original 的验证最优参数，原封不动用于 item-only（B=sym）。
这回答“仅替换用户表的影响”；双方独立调参后的结果回答“各自充分调优后的性能”。
不要用 original 的固定参数限制 item-only 后，宣称后者已经充分调优。

公共固定设置：dim=64、batch-size=2048、pool=mean、保留第 0 层（SimGCL 按自身实现排除）、
epochs 上限=1000、每 5 轮验证一次、patience=20 次验证、top-k=20、seed=2025、weight-decay=0。
DirectAU 固定 encoder=lightgcn；MF 编码器应另列实验，不混进本轮层数搜索。
所有训练使用完整训练交互；调参时不要使用 max-steps 或 eval-users。

## 2. 第一轮：每个模型、每个版本 36 组

每行内取笛卡尔积，各行之间不是笛卡尔积。

| 模型 | lr | context-hops | 第三个搜索参数 | 组数/版本 |
|---|---|---|---|---:|
| LightGCN-RNS | 1e-4、5e-4、1e-3 | 1、2、3 | l2：0、1e-5、1e-4、1e-3 | 36 |
| StabCF | 1e-4、5e-4、1e-3 | 1、2、3 | l2：0、1e-5、1e-4、1e-3 | 36 |
| SimGCL | 1e-4、5e-4、1e-3 | 1、2、3 | l2：0、1e-5、1e-4、1e-3 | 36 |
| DirectAU | 1e-4、5e-4、1e-3 | 1、2、3 | gamma：0.1、0.5、1、2 | 36 |

第一轮其他参数：

- RNS：始终一个随机负例，不搜 n-negs、window-length、alpha。
- StabCF：n-negs=64、window-length=5、alpha=1。
- SimGCL：eps=0.1、cl-rate=0.5、ssl-temp=0.2；始终一个随机负例。
- DirectAU：不使用负例，也不使用模型 l2；设 l2=0 便于记录，weight-decay 固定 0。

总预算：36 × 4 类 × 2 版本 = 288 次训练/数据集/调参 seed。
七个数据集全部执行是 2,016 次，不建议未经 Movie 验证直接全量启动。
显存不足时先降低 eval-batch-size；若必须降低训练 batch-size，应在成对实验中统一。
SimGCL 的 batch-size 会改变对比负样本集合，DirectAU 的 batch-size 会改变 uniformity 样本对，
因此不能仅把它视作显存参数。

## 3. 第二轮：在各自第一轮验证最优配置上细搜

固定该模型/版本的最优 lr、层数、l2，分别搜索：

| 模型 | 参数一 | 参数二 | 参数三 | 组数/版本 |
|---|---|---|---|---:|
| StabCF | n-negs：16、32、64 | window-length：1、3、5 | alpha：0.1、1、10 | 27 |
| SimGCL | eps：0.05、0.1、0.2 | cl-rate：0.01、0.05、0.1、0.5 | ssl-temp：0.1、0.2、0.5 | 36 |

StabCF 的 alpha 必须严格大于 0，因为模型计算 log(alpha)；不要加入 0。
SimGCL 的温度必须大于 0。eps=0 或 cl-rate=0 可以另作机制消融，不与完整模型最优混称。
模型间损失归约不同，不能通过比较训练 loss 大小来选模型或比较正则强度。

第二轮两个版本最多额外 126 次训练/数据集，重复第一轮的配置可以跳过。
分阶段搜索节省预算，但不等价于全联合网格最优；若第二轮改变了较多损失权重，
可在所得最优点附近再复查 lr、l2，且对两种版本使用同等复查预算。
最优值落在边界时，再沿该方向扩展，不要预先把所有范围扩大。

## 4. item-only 的 B 消融单独做

各模型选定的最佳配置上，只替换 `--b-mode`：

| b-mode | 初始用户表示 |
|---|---|
| sym | D_u^(-1/2) R D_i^(-1/2) Q |
| mean | D_u^(-1) R Q |
| sqrt | D_u^(-1/2) R Q |

四类模型共 12 个配置，其中 4 个 sym 可复用，因此新增 8 次训练/数据集/seed。
这是固定其他参数的消融，不是对每种 B 都充分调参后的上限比较。
degree 模式的 a、b 连续指数更适合后续机制研究；第一轮不加入以控制预算。

## 5. 算力有限时的起步预算

把第一轮层数固定为 2，仍保留三种 lr 和四种 l2/gamma：
12 组 × 4 类 × 2 版本 = 96 次训练/数据集。
先在 Movie 执行，确认最佳点和耗时，再扩展层数或数据集。
每轮都使用验证集 Recall@20 选择，不能从短跑指标判断最终赢家。

## 6. 命令示例

以下命令在已配置 GPU 的 Python 环境运行；未验证 GPU 性能。
完整路径避免工作目录混淆（服务器上替换为实际路径）：

```powershell
$cfScript = 'D:\吴熙\实验结果\WWW2027\AlphaFree\train_cf.py'
python $cfScript --dataset Movie --model lightgcn-rns --variant both --device cuda:0 --context-hops 2 --lr 0.001 --l2 0.0001 --validation-only
python $cfScript --dataset Movie --model stabcf --variant both --device cuda:0 --context-hops 2 --lr 0.001 --l2 0.0001 --n-negs 64 --window-length 5 --alpha 1 --validation-only
python $cfScript --dataset Movie --model simgcl --variant both --device cuda:0 --context-hops 2 --lr 0.001 --l2 0.0001 --eps 0.1 --cl-rate 0.1 --ssl-temp 0.2 --validation-only
python $cfScript --dataset Movie --model directau --variant both --device cuda:0 --context-hops 2 --lr 0.001 --l2 0 --gamma 1 --validation-only
```

这些是每类模型的一组示例，不是已验证的最佳参数。
同一个 CLI 的 `--variant both` 对两个版本使用相同超参数；
最终若它们的验证最优参数不同，应分别使用 `--variant original` 和 `--variant item-only`。

最终锁定超参数后，使用 2025、2026、2027、2028、2029 五个 seed 重复，去掉 validation-only，
每次仍由验证集选择最佳轮次，最后只报告对应测试结果的均值和标准差。
不得根据测试结果更改超参数、挑选最佳 seed 或挑选汇报数据集。

注意：validation-only 表示不计算测试指标，不表示完全不读取 test.txt。
AlphaFree 原协议在验证时屏蔽 train+test，测试时屏蔽 train+valid；该规则保持不变。
