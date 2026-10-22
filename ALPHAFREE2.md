# AlphaFree2：原始用户分支的 29 种 B 聚合

这是本地实验变体，不是作者官方模型。原版仍使用 `--model_name AlphaFree`（默认）。
新模型使用 `--model_name AlphaFree2`，代码在 `models/AlphaFree2.py`。
当前正式方案与四卡命令见 [ALPHA_EXPERIMENTS.md](ALPHA_EXPERIMENTS.md)：
1 个原版 mean + 28 个变体，Baby/Steam 共 58 次，不再重复训练 AlphaFree2 mean。

## 修改范围

原版：`U = mlp_origin(D_u^-1 R X)`。

AlphaFree2：`U = mlp_origin(B X)`。

- R 是从 `train_user_list` 重建的交互计数矩阵，保留重复次数，不从已归一化的预处理缓存反推。
- X 仍是固定的原始语义向量；没有添加可学习 item ID 表、用户向量表或 GNN。
- 保留先聚合语义、再经过 MLP 的顺序，不改成聚合最终 item embedding。
- 原始物品分支 `mlp_origin(X)` 保持不变。
- **增强用户分支仍使用作者的增强历史均值** `mlp_aug(B_aug X)`。
- 增强物品分支、InfoNCE、对齐损失、采样方式、余弦评分及候选屏蔽规则均保留。

这是“仅改变原始用户聚合”的实验，不能描述成两条用户分支都改了。
增强分支也改 B 属于另一项实验，当前没有启用。
默认 `--b-mode mean` 是原版对照；当原版缓存来自相同训练划分时，两者数学一致。
AlphaFree2 仍然使用语义信息，不能称为纯 CF；它检验的是我们的 B 构造能否迁移到语义模型。

## 29 种配置

1. `B = D_u^(-a) R D_i^(-b)`：a、b 各取 `{0, 0.25, 0.5, 0.75, 1}`，共 25 种。
   命名点 sum=(0,0)、mean=(1,0)、sqrt=(0.5,0)、sym=(0.5,0.5) 已包含其中，不重复计算。
2. `weighted_mean`：先计算 `W = R D_i^(-b)`，再对 W 做行和归一化；
   b 取 `{0.25,0.5,0.75,1}`，共 4 种。b=0 与 mean 重复，不加入。

所有度数只用训练集。零度数/空历史安全处理为零行。
命名模式使用固定指数，忽略传入的 a、b；degree 使用两个指数，weighted_mean 只使用 b。
CLI 同时接受 `--b-mode/--b-a/--b-b` 和下划线别名 `--b_mode/--b_a/--b_b`。

## GPU 服务器：单次训练

先准备原版 AlphaFree 的 Python 依赖、CUDA 和编译后的评估器。
与原版一样，在 AlphaFree 项目目录运行；不需要 ItemGCN 或 MixGCF。

```bash
cd models/base
python setup.py build_ext --inplace
cd ../..

# 原版平均聚合作为对照；只评估验证集
python main.py --phase train --model_name AlphaFree2 --dataset amazon_movie --cuda 0 --seed 2025 --b-mode mean --validation-only --saveID mean_control

# 对称归一化
python main.py --phase train --model_name AlphaFree2 --dataset amazon_movie --cuda 0 --seed 2025 --b-mode sym --validation-only --saveID sym_trial

# 自定义双指数
python main.py --phase train --model_name AlphaFree2 --dataset amazon_movie --cuda 0 --seed 2025 --b-mode degree --b-a 0.25 --b-b 0.75 --validation-only --saveID degree_trial

# 物品度加权均值
python main.py --phase train --model_name AlphaFree2 --dataset amazon_movie --cuda 0 --seed 2025 --b-mode weighted_mean --b-b 0.5 --validation-only --saveID weighted_trial
```

与原版相同，`--dataset` 使用 amazon_movie 等目录名，不接受 Movie 简写。
单次运行可在命令行覆盖 `--lr`、`--tau_a`、`--tau_r`、`--lambda_align` 等参数。
改变 B 不需要重做增强缓存；改变 K_c、语义特征或训练划分则需要对应的独立缓存。
当前数据加载器不会自动验证 K_c 与缓存的一致性，不能只改 K_c 后复用旧缓存。
AlphaFree2 当前入口仅支持从头训练，不支持加载作者权重或从指定 checkpoint 单独推理。
不要使用 `--phase inference` 或 `--test_only`，也不支持 `--neg_sample -1` 的未实现批内模型类。

## GPU 服务器：29 组搜索

```bash
# 预览 29 条命令；不读大型数据、不写文件、不启动训练
python search_alphafree2.py --dataset amazon_movie

# 先执行一个完整训练任务（不是短跑），检查资源与依赖
python -u search_alphafree2.py --dataset amazon_movie --cuda 0 --max-runs 1 --output weights/alphafree2_movie_v1 --execute

# 同一配置继续跑，已完成任务会跳过
python -u search_alphafree2.py --dataset amazon_movie --cuda 0 --output weights/alphafree2_movie_v1 --execute

# 或在后台运行整套搜索
nohup python -u search_alphafree2.py --dataset amazon_movie --cuda 0 --output weights/alphafree2_movie_v1 --execute > alphafree2_movie_search.log 2>&1 &
```

每个数据集、每个 seed 29 次独立初始化的训练：mean 对照使用原版 AlphaFree，其余 28 次使用 AlphaFree2。
默认固定各数据集 YAML 的 K_c/lambda_align/tau_a/tau_r 和原版其他默认值；
**这次只是 29-B 搜索，不是 AlphaFree 模型超参数 × 29-B 的完整联合网格。**
默认 seed=2025；支持覆盖 `--seed`、`--max-epoch`、`--batch-size`、`--lr`、`--num-workers`。
默认每个任务最多 500 轮，每 5 轮验证，连续 20 次验证无提升早停。
修改这些训练参数应使用新输出目录。显存不足时可降低 batch-size，但须对全部 B 及 mean 对照保持一致。
batch-size 也改变对齐损失的批内对比条件，不仅影响显存。

搜索强制训练期间 validation-only；每组训练结束后加载验证集最优权重测试一次，不导出测试推荐列表。
训练和最终测试输出统一保存在 `.attemptN.txt`；确认结果保存成功后默认删除最佳 checkpoint。
单独使用上方 main.py 诊断命令时，可加 `--final-test` 启用最终测试，但自动删除由搜索调度器负责。
AlphaFree2 保留原优化器/训练循环和文本日志，但不调用原版要求 valid/test 双曲线的训练结束绘图函数。
test.txt 仍因原版候选屏蔽协议在训练初始化时被读取，训练中不计算测试指标。
搜索不自动下载、生成语义特征或构造增强缓存；执行前检查所需缓存完整性。

## 输出、续跑和选参

搜索目录保存 manifest.json、每个 B 的状态 JSON、每次尝试的独立日志和 summary.json。
summary.json 按最佳验证 Recall@20 排名；未完成全部 29 项时 best 只是暂定。
args.txt、包含最终测试的 stats.txt、result.json 保存在模型目录；checkpoint 仅在训练/测试期间保留：

```text
weights/<dataset>/<AlphaFree或AlphaFree2>/<时间戳、唯一运行标识及参数>/
```

搜索目录的状态和 summary 会记录 checkpoint 的绝对路径。
训练和最终测试正常结束、结果元信息与 TXT 哈希匹配时跳过该任务，不要求已删除的 checkpoint 存在。
默认删除权重后不可再次推理；如需保留，搜索入口加 `--keep-checkpoints` 并使用独立输出目录。
这不是优化器/epoch 级续训，失败重试会从头训练并保留旧日志。
失败后先检查日志，确认问题解决，再在原命令加入 `--retry-failed`。

代码、数据/缓存、配置或 Python/PyTorch 版本变化时，manifest 会拒绝混用，需换新目录。
数据指纹只保证当前实验中数据不变，并不能证明下载缓存使用的 K_c；需使用匹配的官方配置和缓存。
同一输出目录支持 `--num-shards` / `--shard-index` 多 worker，先用 `--prepare` 冻结同一清单。
异常断电可能留下分片锁或任务锁；先核实没有对应活跃进程，再手工处理具体锁文件。
不要同时重复启动上述前台和后台命令。

与原始 AlphaFree 比较时，需对齐 seed、训练/评估协议和其他超参数；
29 组挑最佳也增加了搜索预算，应披露。多 seed 复验使用预先锁定的配置，不挑最佳 seed。

## 本机验证与边界

```bash
python -m unittest discover -s tests -p 'test_alphafree2.py' -v
```

- 合成数据：29 个 B 与现有 CF 构造一致，训练/验证无交互泄漏。
- 小型 CPU 数值测试：mean 的输出、损失、梯度、评分与原版一致；29 种设置均可反传。
  测试临时重定向原版 `.cuda()` 调用，**不意味着正式训练入口支持 CPU**。
- 测试确认增强分支、固定语义和参数集合未改变。
- 搜索的状态/跳过/清单保护使用模拟子进程测试；验证集筛选使用隔离的控制流测试。
- 真实 Movie 数据：重建 mean 与下载缓存最大差值 0；29 个 B 与现有 CF 矩阵最大差值 0。
- 本机没有可用 CUDA，未验证完整 GPU 训练、编译评估器或真实 29 次训练调度。
  CUDA 模型测试在有 GPU 的环境中才运行。

更新：本地 Baby 和 Steam 的计数 mean 与原版缓存均通过一致性检查；
新增四 worker 并发调度模拟、每组最终测试/权重清理，以及按验证集选参后导出已记录测试指标的入口。
旧搜索输出不兼容当前清单版本，必须使用新目录。
