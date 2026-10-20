# 模型参数 × 29 种用户表示的联合搜索

这是当前采用的搜索方案，替代先前 CF_TUNING.md 的分阶段草案。
入口为 `search_cf.py`，实际配置为 `configs/cf_joint_search.json`。
前四类模型的参数网格对应旧 ItemGCN/experiments.json；FISM 是新增的独立基线网格。
评估/早停遵循新的 AlphaFree 适配框架。
不把旧实验的 test Recall 选配置方式迁移过来：本次只按 validation Recall@20 选配置。

## 搜索空间

| 模型 | 模型参数网格 | Original | Item-only（×29 B） |
|---|---|---:|---:|
| LightGCN-RNS | 单随机负例，其余固定 | 1 | 29 |
| StabCF | n-negs={64,32,16} × window-length={5,10} × alpha={20,21} | 12 | 348 |
| SimGCL | cl-rate={0.5,0.2,2} × eps={0.1,0.5}，ssl-temp=0.2 | 6 | 174 |
| DirectAU | gamma={0.2,0.5,1,2,5,10} | 6 | 174 |
| 每数据集、每 seed 合计 | | 25 | 725 |

此外加入 FISM-BCE `native` 基线：`fism-alpha={0,0.5,1}` ×
`fism-reg={0.0001,0.001,0.01}`，共 9 组，不与 29 种 B 做笛卡尔积。
固定 4 个负例/正例、scalar user bias 开启、weight-decay=0；细节见 [FISM_CF.md](FISM_CF.md)。
该网格是新增起始搜索空间，不是从旧 ItemGCN 网格或论文调优结果中获得。

公共参数：dim=64、context-hops=3、batch-size=2048、lr=0.001、pool=mean、保留初始层
（SimGCL 自身按原实现排除初始层）。DirectAU encoder=lightgcn。
前三类 l2=0.001、weight-decay=0；DirectAU l2=0、weight-decay=0.001。
最多 1000 轮，每 5 轮验证一次，连续 20 次验证无提升早停，默认 seed=2025。
这些参数来自旧网格，不代表在新数据集上已最优。

每个模型配方、每种 B 都独立初始化、从头训练，不继承其他任务的权重。
完整 B 空间：

1. `B = D_u^(-a) R D_i^(-b)`，a、b 均取 `{0,0.25,0.5,0.75,1}`，共 25 种。
   sum、mean、sqrt、sym 是其中四个命名点，不另算。
2. weighted_mean 的 b 取 `{0.25,0.5,0.75,1}`，共 4 种。b=0 与 mean 重复，去掉。

新数据集默认完整包含 sym，不采用旧补跑脚本默认排除 sym 的 28-B 空间。
原四类模型仍为 750 次/数据集/seed；加上 FISM 后默认总量为 759 次，7 个数据集为 5,313 次/seed。
item-only 的搜索预算大于 Original，应在论文中披露；不要声称是相等搜索预算实验。

## 服务器准备

复制同一父目录下的 AlphaFree 和 ItemGCN 两个目录，或通过 `--itemgcn-root` 指定源模型目录。
至少准备各数据集 `cf_data/train.txt`、`valid.txt`、`test.txt`，无需语言表示或预处理缓存。
激活安装了 GPU 版 PyTorch、NumPy、SciPy 的 Python 环境；不需要编译 AlphaFree 的 Cython 评估器。
默认路径基于脚本位置，而不是当前工作目录。

以下命令在服务器的 AlphaFree 目录执行。输出路径可以改成容量充足的磁盘。
先检查环境和数据，不会启动训练：

```bash
python -c 'import torch, numpy, scipy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python search_cf.py --datasets Movie
python search_cf.py --datasets all --num-shards 4 --shard-index 0
```

不加 `--execute` 或 `--prepare` 时只是 dry-run：校验组合、显示数量，不读大数据文件、不写输出、不训练。
`--datasets all` 包含 Movie、Book、Video、Baby、Steam、Beauty、Health。

## 单 GPU：先跑 Movie

```bash
python -u search_cf.py --datasets Movie --device cuda:0 --output weights/joint_movie --execute
```

这是正式完整搜索，默认 759 个任务，不是短跑。
可以先用 `--max-jobs 1` 只执行一个完整训练任务，确认资源占用后，再去掉该限制用相同命令续跑。
同一输出目录的完成任务会跳过；中断任务会重新从头训练，不是恢复优化器的 epoch 级续训。

只跑部分模型/版本的示例（必须使用独立输出目录）：

```bash
python search_cf.py --datasets Movie --models stabcf simgcl --phase item-only --device cuda:0 --output weights/joint_movie_stabcf_simgcl --execute
```

更改模型或数据集筛选会改变清单，不能复用原完整搜索目录。
`--phase original` / `item-only` 只筛选前四类模型，不含 FISM；`--phase native` 只筛选 FISM。
默认 `--phase both` 包含前四类的两个版本及 FISM 的唯一 native 版本。
此次代码和默认网格已变更，不要续写旧 750 任务的输出目录或修改旧 manifest。
若只需要补跑 FISM，使用新的独立目录：

```bash
python search_cf.py --datasets Movie --models fism --device cuda:0 --output weights/fism_movie_v1 --execute
```

## 多 GPU：后台启动

四张卡分摊 Movie 的 759 个任务（使用新的输出目录）：

```bash
bash start_cf_joint.sh Movie 0,1,2,3 weights/joint_movie_with_fism
```

四张卡分摊全部 7 个数据集的 5,313 个任务：

```bash
bash start_cf_joint.sh all 0,1,2,3 weights/joint_all7_with_fism
```

该脚本先统一冻结清单，然后每张 GPU 启动一个后台 Python 进程，每个进程内部串行训练。
每个进程通过 CUDA_VISIBLE_DEVICES 看到一张卡，所以内部始终使用 `--device cuda:0`。
终端打印 PID 和独立日志路径。后台启动成功不等于训练成功，请查看日志和状态。
可用 `CF_PYTHON=/absolute/path/to/python bash start_cf_joint.sh ...` 指定环境。

手动启动分片（例如 GPU 2 执行第 2 片，共 4 片）：

```bash
python search_cf.py --datasets Movie --output weights/joint_movie --prepare
CUDA_VISIBLE_DEVICES=2 python -u search_cf.py --datasets Movie --device cuda:0 --num-shards 4 --shard-index 2 --output weights/joint_movie --execute
```

各 worker 的数据集、模型、phase、seeds 和输出目录必须相同，仅设备和 shard-index 不同。
全局序号按 `index % num_shards` 分片，无遗漏/重复；不保证各片耗时相等。
共享文件系统上也可多节点手动分片。不得在不同节点各用独立输出目录后误认为已经全局续跑。

## 汇总、续跑和失败处理

```bash
python search_cf.py --output weights/joint_movie --summarize
```

生成 summary.json，按数据集、模型、版本、seed 分组，记录计划数、完成数、是否完整、
当前验证最优任务及其参数和结果。只要组内尚有缺失任务，complete 就是 false，最优结果只是暂定。
短跑结果永远不参与最优选择。汇总不计算测试指标。

执行时输出结构：

```text
output/
  manifest.json                       # 冻结任务清单、数据/代码哈希、运行条件
  scheduler_logs/                     # 后台 worker 日志（启动脚本生成）
  summary_shard_0_of_4.json            # worker 完成/退出前的全局快照
  summary.json                        # 显式 summarize 生成
  amazon_movie/<job-id>/
    status.json                       # running / completed / failed
    running.lock                      # 活跃 worker 锁，正常结束自动移除
    attempt_<timestamp>/
      job.json
      train.log
      run/config.json
      run/history.jsonl
      run/best.pt
      run/result.json
```

- 完成任务只有在结果和 best.pt 都存在时才跳过。
- 失败任务保留错误堆栈和原 attempt，默认不重复尝试。修复后在原命令添加 `--retry-failed`。
- 训练中断但正常退出会释放锁；机器掉电或 kill -9 可能遗留 running.lock。
  应先根据锁中的主机名、PID 确认没有活跃训练，再手工移除那一个锁。脚本不会擅自抢占。
- 并行初始化清单发生竞争会提示稍后重试；启动脚本已用 prepare 避免此问题。
- 数据、模型源码、适配代码、网格、评估 batch-size、线程数或 Python/PyTorch 版本变化时，
  续跑会拒绝混用旧清单。换新输出目录，不要直接修改旧 manifest。
- num-shards、shard-index、device 和 max-jobs 不属于模型搜索空间，可更换后跳过已完成任务；
  重分片前先停止旧 worker，避免不必要的竞争。
- 同一目录不应重复启动同一套后台 worker；任务锁会保护活跃任务，但不能替代资源管理。

每个任务均保留最佳 checkpoint，失败重试也保留旧文件，不会自动清理。
大数据集可能需要数十 GB 甚至更多磁盘空间，请预留空间；降低训练 batch-size 则属于实验配置变更。

## 测试集边界与最终报告

搜索强制 validation-only，所有 result.json 的 test 为 null。
AlphaFree 的候选屏蔽协议仍保持：验证屏蔽 train+test，测试屏蔽 train+valid。
因此“仅验证选参”不是完全不读取 test.txt；测试交互只参与规定的屏蔽，不参与训练图/负采样/梯度。
选择最佳模型参数及 B 时只使用验证 Recall@20。
锁定配置后再单独做测试评估与多 seed 复验；本调度器不自动执行最终测试，也不挑选最佳 seed。

## 本机验证

```bash
python -m unittest discover -s tests -p 'test*cf.py' -v
python search_cf.py --datasets Movie --models lightgcn-rns --device cpu --smoke --max-jobs 2 --output weights/joint_smoke --execute
```

smoke 会强制 1 轮、1 个 batch、batch-size=32、验证前 32 位用户；保留完整训练图。
不显式指定 max-jobs 时，smoke 最多尝试 1 个任务。
smoke 使用独立默认输出目录，并冻结短跑标记，不可能误续跑成正式结果。
本机 CPU 可验证采样、训练、保存、选参汇总逻辑；不代表 GPU 性能或模型已经收敛。

2026-10-20 验证记录：共 23 项自动化测试，22 项通过、1 项 CUDA 测试跳过；
Movie 上四类模型的 original/item-only 共 8 个调度任务均通过上述短跑，
并实测完成任务续跑时正确跳过。记录保存在 `weights/cf_joint_verify_20261020/`。
启动脚本已通过 Bash 语法检查；尚未在 GPU 服务器启动正式搜索。

加入 FISM 后：30 项自动化测试中 28 项通过、2 项 CUDA 测试跳过；
Movie 上 FISM 的 9 组调度短跑完成，续跑全部正确跳过，记录在
`weights/fism_verify_20261020/`。旧四类模型的配方和 29-B 搜索空间保持不变。
