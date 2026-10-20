# 在 AlphaFree 数据集上运行 LightGCN-RNS / StabCF / SimGCL / DirectAU / FISM

新增入口 `train_cf.py` 支持五个模型各自的 original / item-only 版本，共十个版本。
它复用相邻 `ItemGCN` 仓库中的模型代码，读取 AlphaFree 原始划分，并实现与
AlphaFree 相同的 Recall、NDCG、HR 定义及候选屏蔽规则。
原始 `main.py`、AlphaFree 模型、下载脚本和已有 ItemGCN 模型均未被此适配改写。

这是独立的纯 CF 入口，不继承强耦合文本表示的 `AlphaFreeRS`。
不需要 LLM、文本 embedding、语义增强或 Linux 编译的 Cython 评估扩展。
它不是 AlphaFree 架构的消融实验，也不等同于完整复现论文的基线超参数。

## 模型对应关系

| 命令 | original | item-only | 保留的训练逻辑 |
|---|---|---|---|
| `--model lightgcn-rns`（别名 `rns`） | `LightGCN(ns='rns')` | `XLightGCN(ns='rns')` | 单随机负例、BPR、正则，无历史混合 |
| `--model stabcf` | `LightGCN(ns='stabcf')` | `XLightGCN(ns='stabcf')` | 历史正例混合、用户感知负例混合、BPR、正则 |
| `--model simgcl` | `SimGCL` | 本地类 `XSimGCL` | 单负例 BPR、扰动视图、对比损失、正则 |
| `--model directau` | `DirectAU` | `XDirectAU` | alignment + gamma × uniformity，不采负例 |

`--model fism --variant native` 单独运行 FISM-BCE：两个 item 表、训练历史聚合、排除当前目标、
可选用户标量 bias；没有 GNN。native 是 original 的兼容别名，item-only 使用可调的 B 聚合。
original 实现与本地 MixGCF FISM 的目标一致，
不需要复制 MixGCF 仓库。配置、公式及搜索命令见 [FISM_CF.md](FISM_CF.md)。

注意：本地类名 `XSimGCL` 是你的 item-only SimGCL，不是论文中的 XSimGCL 算法。
输出标签始终使用 `simgcl_item-only`，避免混淆。
StabCF 不使用 `LightGCN_StabCF.py` 中另一条 `StabCF2 / igcn` 分支。

item-only 删除可学习用户表，以 `U0 = B @ Q` 构造初始用户表示。
默认 `B = D_u^(-1/2) R D_i^(-1/2)`；支持 `--b-mode mean/sum/sqrt/sym/degree/weighted_mean`。
`degree` 使用 `--b-a`、`--b-b`；`weighted_mean` 使用 `--b-b`。
R 和所有度数只来自 train.txt。
四类模型仍保留各自原来的传播方式；本次没有额外引入 item-item 图。
DirectAU 默认 LightGCN 编码器，也支持 `--encoder mf`。

## 依赖与数据

需要 Python、PyTorch、NumPy、SciPy；前四类模型另需本地 ItemGCN 源码，FISM 不需要。
默认 ItemGCN 路径是 AlphaFree 的相邻目录；可用 `--itemgcn-root` 指定。
本机已验证环境：`D:\Anaconda3\envs\aminer39\python.exe`，Python 3.9.12，
PyTorch 1.11.0+cpu，NumPy 1.20.3，SciPy 1.8.1。
不建议使用之前下载时的 Python 3.6 环境或旧的 LightGCN / PyTorch 1.4 环境运行本入口。

| `--dataset` | 目录 |
|---|---|
| Movie | `data/amazon_movie/cf_data` |
| Book | `data/amazon_book_2014/cf_data` |
| Video | `data/amazon_video/cf_data` |
| Baby | `data/amazon_baby/cf_data` |
| Steam | `data/steam/cf_data` |
| Beauty | `data/amazon_beauty_personal/cf_data` |
| Health | `data/amazon_health/cf_data` |

每个目录需要 train.txt、valid.txt、test.txt，每行是 `user item1 item2 ...`。
不重新划分、不重编号；三份文件合起来的用户和物品 ID 应分别从 0 连续编号。
也可用 `--data-root` 指向另一个包含这些数据集目录的位置。
目录必须直接包含 cf_data，避免多一层解压目录。

## 先做短跑

PowerShell，从任意目录运行：

```powershell
& D:\Anaconda3\envs\aminer39\python.exe "D:\吴熙\实验结果\WWW2027\AlphaFree\train_cf.py" --dataset Movie --model all --variant both --device cpu --epochs 1 --batch-size 256 --max-steps 2 --eval-users 128 --eval-batch-size 64
```

训练保留完整 Movie 训练图，只限制优化步数和评估用户数；排序候选仍是全部物品。
`--max-steps` 或 `--eval-users` 非零时，输出会标记 `smoke_test: true`。
短跑指标不得用作论文性能结果。

## 正式训练

在配置好 GPU 版 PyTorch 的环境中，示例命令如下：

```bash
python train_cf.py --dataset Movie --model lightgcn-rns --variant both --device cuda:0 --epochs 1000 --eval-every 5 --patience 20
nohup python train_cf.py --dataset Steam --model stabcf --variant both --device cuda:0 --epochs 1000 --eval-every 5 --patience 20 &
nohup python train_cf.py --dataset Steam --model simgcl --variant both --device cuda:1 --epochs 1000 --eval-every 5 --patience 20 &
nohup python train_cf.py --dataset Steam --model directau --variant both --device cuda:3 --epochs 1000 --eval-every 5 --patience 20 &
```

这些 bash 示例在 AlphaFree 目录运行。没有 GPU 时可改为 `--device cpu`，但完整训练可能较慢。
`--model all --variant both` 会依次训练十个版本，不会并发占满显存。
省略设备参数时自动选择可用 GPU，否则 CPU。
正式训练必须移除短跑参数，默认值 0 表示不限制。

公共默认值：dim=64、context-hops=3、batch-size=2048、lr=0.001、l2=0.001、seed=2025。
StabCF 默认 n-negs=64、window-length=5、alpha=1。
LightGCN-RNS 始终只抽一个随机负例，n-negs、window-length、alpha 不参与它的训练逻辑。
SimGCL 默认 eps=0.1、cl-rate=0.5、ssl-temp=0.2，单个随机负例。
DirectAU 默认 gamma=1，Adam weight-decay=0；l2 不会额外加入 DirectAU 损失。
保留 SimGCL 原实现的求和式损失，不能拿损失绝对值与其他模型比较。
这些是起始配置，不是 AlphaFree 数据集上的已调优参数。
当前联合搜索空间与服务器命令见 [CF_JOINT_SEARCH.md](CF_JOINT_SEARCH.md)。单次调参加 `--validation-only`，
此时 result.json 的 test 为 null，不计算测试指标；AlphaFree 验证候选屏蔽规则不变。

## 评估协议和实验边界

- 每个 epoch 打乱训练交互对，逐对训练，保留 ItemGCN 的交互均匀采样语义。
  不采用 AlphaFree 的用户均匀抽正例策略，也不强行把各模型变成 InfoNCE 训练。
- 负例仅排除该用户的训练正例，不查询验证/测试标签进行训练采样。
- StabCF 历史只从训练集采样；历史长度大于窗口时无放回，否则有放回，与原实现一致。
- 验证候选屏蔽 train + test；测试候选屏蔽 train + valid；当前目标正例始终保留。
  这是 AlphaFree 原代码的规则，不是“验证只屏蔽 train”的常见另一种协议。
  测试标签只用于规定的候选屏蔽，不参与构图、梯度、负采样或最佳轮次选择。
- 所有待评估用户逐用户计算 Recall@20、NDCG@20、HR@20，再做宏平均。
  不采样评估负例。仅空目标集用户跳过；若目标 split 用户没有训练历史，仍评估。
- 仅验证集 Recall 选择最佳 checkpoint 和早停；最终测试在恢复最佳权重后运行。
- PyTorch topk 和原 C++ topk 对完全同分物品的排序可能不同。
- 需要用验证集调参、多随机种子重复；不能把单次短跑或默认配置当作论文结论。
- user-table-free 不等于零用户相关内存：训练图、历史索引和传播中的用户表示仍存在。

## 输出与复现

输出在 `weights/cf/<dataset>/<model>_<variant>/<时间戳>_seed<seed>/`：

- config.json：参数、数据文件 SHA256、模型源码/适配代码 SHA256、参数量、评估协议。
- history.jsonl：每轮训练损失和验证结果。
- best.pt：验证最优模型的 state_dict、轮次、配置和验证结果；不包含稀疏图，也不保存优化器状态。
- result.json：最佳轮次对应的最终验证/测试结果、耗时和短跑标记。
- 数据集输出目录还会生成 summary_*.json，汇总本次所有完成的模型。

每次新建独立运行目录，不覆盖旧结果。暂不支持训练断点续跑。
加载 best.pt 前需用相同配置和相同训练数据重建模型/图；图不在 checkpoint 中。
把工程移到服务器时，应同时复制 AlphaFree 和 ItemGCN，保持源码一致。

## 已验证范围（2026-10-19）

Movie 的三个模型 × 两种版本均已在 CPU 完成 1 轮、每轮 2 个 batch 的短跑，
batch-size=256，dim=64，context-hops=3。先检查验证/测试各前 128 位用户，
随后六个版本又完成全量评估：验证 25,947 位非空目标集用户，测试 26,073 位用户。
六个版本均完成反向传播、保存最佳权重、重载和评估，未出现非有限损失。
每个原版参数量 2,466,368；每个 item-only 版本 797,696。
尚未进行收敛训练、超参数搜索、多种子比较或 GPU 实测。
全量评估短跑汇总：`weights/cf/amazon_movie/summary_20261019_220441_991933.json`。
自动化测试共 10 项，其中 9 项通过，1 项 CUDA 测试因无 CUDA 而跳过。

自动化测试（在 AlphaFree 目录运行）：

```powershell
& D:\Anaconda3\envs\aminer39\python.exe -m unittest discover -s tests -p test_cf.py -v
```

覆盖六个版本的参数结构、反向传播、负例/历史采样、训练图与 B 的原实现一致性、
B@Q 梯度、评估屏蔽、指标与原评估公式一致性、checkpoint 重载、DirectAU 单样本尾批。
CUDA 测试在没有 CUDA 时显式跳过。

2026-10-20 更新：加入 LightGCN-RNS 两种版本，回归测试扩展为覆盖八个版本；
共 13 项测试，12 项通过，1 项 CUDA 测试跳过。
LightGCN-RNS 两种版本也已在 Movie 上完成各 2 个 batch 的 CPU 短跑、
全量验证/测试和权重重载；这仍不是收敛训练结果。
汇总：`weights/cf/amazon_movie/summary_20261020_013316_618134.json`。

## 数据集文件速查

| 文件/目录 | 用途 | 这八个纯 CF 版本是否需要 |
|---|---|---|
| cf_data/train.txt | 训练交互；每行 user 后跟多个 item，用于训练、构图和用户历史 | 必需 |
| cf_data/valid.txt | 验证正例，用于调参、早停；测试时也用于候选屏蔽 | 必需 |
| cf_data/test.txt | 测试正例；AlphaFree 协议下验证时也用于候选屏蔽 | 必需，但不参与训练图/采样 |
| item_info/item_title.json | 物品 ID 和标题，用于生成语言表示；Movie 包中没有此文件 | 不需要 |
| item_info/item_cf_embeds_large3_array.npy | Movie 包的预计算语言表示，每行对应一个物品 | 不需要 |
| item_info/item_cf_embeds_LLAMA_array.npy | Steam/Video 包的 Llama 语言表示 | 不需要 |
| preprocessed/user_interact_matrix.npz | AlphaFree 的训练用户-物品稀疏矩阵，代码按行 L1 归一化 | 不需要，纯 CF 入口自行从 train 构图 |
| preprocessed/aug_user_interact_matrix.npz | AlphaFree 邻居扩展后的用户交互矩阵，按行归一化 | 不需要 |
| preprocessed/item_cf_embeds_augmented.npy | AlphaFree 邻居聚合得到的增强物品语言表示 | 不需要 |
| data/pretrained/*.pth.tar | 作者的 AlphaFree 模型权重，不是原始交互，也不是你的模型权重 | 不需要 |

例如 train.txt 的 `0 6204 6728` 表示用户 0 与物品 6204、6728 有训练正反馈，
不是三列评分格式；文件中未出现的 user-item 对不自动等于真实负反馈。
各 split 的物品 ID 必须与语言矩阵的行号保持一致（运行 AlphaFree 时尤其重要）。
预处理缓存依赖训练划分、语言表示和增强参数；修改这些条件后不能盲目复用旧缓存。
若以后还要比较 AlphaFree，建议保留这些附加文件；本次未删除任何数据文件。
