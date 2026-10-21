# AlphaFree original + 28 聚合：Baby / Steam 四 GPU 实验

当前方案：每数据集 1 个 `AlphaFree`（original/mean）+ 28 个 `AlphaFree2`
（排除 mean），seed=2025，共 58 次独立初始化训练。不继承 original 权重。
四张 GPU 同时执行不同任务，分别分配 15、15、14、14 个；不是单模型 DDP，
不保证各 GPU 同时结束。

## 实验口径

- 原版模型、MLP、负采样、Adam、InfoNCE、对齐损失、语言特征和增强分支不变。
- 仅改变原始用户分支：`U = mlp_origin(B @ X)`，不是 `B @ mlp_origin(X)`。
- AlphaFree2 保留重复交互，`R_ui` 为次数；用户/物品度按次数计算，不排除目标。
- original 使用作者的原始历史缓存；准备阶段检查重建的 count-mean 与缓存最大差不超过 1e-6。
  不匹配就停止，不修改原始数据或缓存。也检查增强矩阵和语言特征形状。
- 两模型通过相同训练管理类运行，只评估验证集，不输出训练期间测试指标和测试推荐列表。
- 按验证 Recall@20 早停和选择；28 个变体内部选最佳，再与 original 各自最佳 epoch 比较。
- 各数据集使用原 YAML：Baby K_c=10、Steam K_c=3；两者 tau_r=tau_a=.2、
  lambda_align=.01、hidden_size=64、lr=.0005、batch_size=4096、neg_sample=256。
  最多 500 轮，每 5 轮验证，连续 20 次验证无提升早停。启动器 num_workers=2/进程。
- `weight_decay` / `regs` 仍未接入原版损失/优化器，本实验不新增正则。
- 独立的 AlphaFree3 实验仍使用旧聚合默认行为；本次没有改变其入口和测试集搜索行为。

## 服务器准备

同步本次全部修改；在项目根目录激活已配置的 AlphaFree CUDA 环境。
需要原版 Python 依赖和编译评估器，所需数据/语言特征/增强缓存必须完整。
准备阶段不会自动下载、运行 LLaMA 或重建增强缓存。若更改 K_c，须自行提供匹配缓存；
缓存一致性检查只验证历史 mean 和形状，不证明增强缓存与 K_c 相符。

先预览（不训练，不写文件）：

```bash
python search_alphafree2.py --datasets Baby Steam --num-workers 2
```

只验证输入、冻结清单（不需要 CUDA）：

```bash
python search_alphafree2.py --datasets Baby Steam --seed 2025 --num-workers 2 \
  --output weights/alpha_baby_steam_v1 --prepare
```

可先在空闲卡运行小型数值/CUDA 测试，不是正式结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests -p test_alphafree2.py -v
```

该测试验证小模型反传/打分，不能代替真实全量训练的显存检查。
如需先完成一个真实训练任务，可运行以下命令，再启动四卡继续剩余任务：

```bash
CUDA_VISIBLE_DEVICES=0 python -u search_alphafree2.py --datasets Baby Steam \
  --seed 2025 --num-workers 2 --cuda 0 --max-runs 1 \
  --output weights/alpha_baby_steam_v1 --execute
```

## 四 GPU 后台启动

```bash
bash start_alpha_baby_steam.sh 0,1,2,3 weights/alpha_baby_steam_v1
```

可以换成 `0,2,4,5` 等四个互不重复的物理 GPU ID。必须恰好指定四张卡。
每个 worker 通过 CUDA_VISIBLE_DEVICES 只看到一张卡，内部使用 `--cuda 0`。
`ALPHA_PYTHON=/path/to/python bash start_alpha_baby_steam.sh ...` 可指定环境。
先检查四张 GPU 并准备清单，再启动四个后台进程；终端输出 PID 和日志路径。
后台启动成功不表示训练已完成，日志在输出目录的 `scheduler_logs/`。

```bash
python search_alphafree2.py --output weights/alpha_baby_steam_v1 --summarize
```

`summary.json` 按数据集分别显示 original、best_variant、完成数量。
未全部完成的排名仅为暂定，不允许最终测试。训练期间每个 worker 只写自己的
`summary_shard_<i>_of_4.json`，不会相互覆盖汇总文件。

## 续跑和失败重试

同样的启动命令会跳过完成任务。不要在旧进程仍活跃时重复启动或更换分片数量。
任务锁和分片锁会阻止并发重复训练。正常异常会释放锁；断电/kill -9 后可能留下锁，
先确认 PID 已停止，再手动处理具体锁文件，不能自动清掉未知活跃进程的锁。
失败任务不会自动重试。修复原因后可以单独重试相应分片，例如物理 GPU 2 跑第 1 片：

```bash
CUDA_VISIBLE_DEVICES=2 python -u search_alphafree2.py --datasets Baby Steam \
  --seed 2025 --num-workers 2 --cuda 0 --num-shards 4 --shard-index 1 \
  --output weights/alpha_baby_steam_v1 --execute --retry-failed
```

重试从头训练，不是 epoch 级续训；保留旧日志和 checkpoint。
改变代码、数据、配置、seed、训练覆盖参数或 Python/PyTorch 版本必须使用新输出目录。
旧的 29 个 AlphaFree2 搜索目录不能直接接着写入。原始训练输出仍保存在
`weights/<dataset>/<AlphaFree或AlphaFree2>/...`；搜索状态记录绝对 checkpoint 路径及哈希。

## 锁定配置后最终测试

全部 58 个任务完成后，先预览验证集选择：

```bash
python eval_alpha_experiment.py --search-root weights/alpha_baby_steam_v1
```

正式测试（一个 GPU 顺序评估 4 个 checkpoint，不重新训练）：

```bash
CUDA_VISIBLE_DEVICES=0 python eval_alpha_experiment.py \
  --search-root weights/alpha_baby_steam_v1 --device cuda:0 --execute
```

每个数据集评估 original 和验证集选中的一个变体，共 4 个结果。
入口先校验代码/输入/checkpoint 指纹，先写锁定选择，再读取测试指标。
original 用原始缓存，变体重建其训练时 count-B；不会采用 AlphaFree3 旧的二值聚合。
只复用冻结 MLP 的推理实现，不运行 AlphaFree3 的 29-B 搜索。
真值使用唯一物品集合，候选屏蔽与原 AlphaFree C++ 后端一致。
输出在 `weights/alpha_baby_steam_v1/final_test/`，包括 selection.json、每个结果及 summary.json。
再次执行会跳过已完成评估；不允许改选择后覆盖原测试目录。

## 本地验证边界

覆盖含重复记录的 29-B 构造、mean 与原版输出/损失/梯度一致、增强分支不变、
58 任务分成四片、四 worker 同时调度模拟、任务锁/失败重试/恢复、完整搜索后验证选择、
最终 checkpoint 重载打分一致性和测试结果恢复。CPU 调度模拟不等于四 GPU 实机验证。
Baby、Steam 的真实 mean 缓存一致性检查通过。本机没有 CUDA，未启动正式训练。

本次验证：Alpha 专项 14 项测试中 13 项通过、1 项因无 CUDA 跳过；
Baby/Steam 的完整准备流程（58 个任务、14 份输入/缓存指纹）通过。
全项目回归仍有之前已确认的 3 个失败（CF 读取校验 2 个、AlphaFree3 测试集搜索 1 个），
属于本次范围外的既有改动，未修改它们；当前 Alpha 准备入口独立检查重复用户行和 ID 范围。
