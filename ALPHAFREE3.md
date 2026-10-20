# AlphaFree3：冻结已有权重，只替换推理时的用户聚合

AlphaFree3 是本地实验变体，不是作者官方模型。与 AlphaFree2 的区别：

| 模式 | 权重 | 用户表示 | 是否训练 |
|---|---|---|---|
| AlphaFree2 | 每个 B 独立从头训练 | `f_theta_B(B X)` | 是 |
| AlphaFree3 | 所有 B 使用同一 checkpoint | `f_theta_star(B X)` | 否 |

AlphaFree3 保留原始语义特征 X、MLP 权重、物品表示 `f_theta_star(X)` 和余弦评分。
只改变用户历史的聚合矩阵 B，随后仍经过同一个已训练的 MLP；不是聚合最终 item embedding。
所有参数 `requires_grad=False`，评分禁用梯度，没有优化器，训练调用会明确报错。
保留并严格加载两个 MLP 的权重，但仅使用原始分支推理，不读取增强数据。
归一化物品表示分批编码一次并缓存，29 个 B 共用同一份物品表示。
每次实际评估会检查全部模型参数的哈希在执行前后不变。

## 依赖与数据

入口为 **`eval_alphafree3.py`**，不是 `main.py`、`train_cf.py` 或 `search_alphafree2.py`。
CPU / CUDA 都支持；仅需 Python、PyTorch、NumPy、SciPy，不需要 Cython 评估扩展、
transformers、大语言模型、ItemGCN 或 MixGCF。

需要：

```text
data/<dataset>/cf_data/train.txt
data/<dataset>/cf_data/valid.txt
data/<dataset>/cf_data/test.txt
data/<dataset>/item_info/<原始语言向量>.npy
data/pretrained/<dataset>.pth.tar
```

Movie / Book 默认读取 `item_cf_embeds_large3_array.npy`，其他数据集读取
`item_cf_embeds_LLAMA_array.npy`。可用 `--features` 显式指定原始语义矩阵。
不需要 `preprocessed/`、增强语义文件或 item_title.json，也不会下载或修改 checkpoint。

`--checkpoint` 可指定作者权重或自己训练的完整 AlphaFree checkpoint。
支持 `{'state_dict': ...}` 格式和直接保存的 state_dict；严格检查参数名称及维度。
只加载可信 checkpoint。较新 PyTorch 使用 weights_only；旧版 torch.load 仍使用其原生反序列化方式。
加载时自动从权重推断 MLP 维度，不需要手工设置 hidden_size。
文件形状匹配并不能证明 item ID 顺序或语义来源匹配；自定义权重必须配套原始数据与特征。

## 29 个 B

与 AlphaFree2 和现有 CF 实验使用同一组：

- `B = D_u^(-a) R D_i^(-b)`：a、b 各为 `{0,0.25,0.5,0.75,1}`，共 25 个点。
- sum、mean、sqrt、sym 为上述点的别名，不重复计算。
- `weighted_mean`：对 `R D_i^(-b)` 再做行归一化，b 为 `{0.25,0.5,0.75,1}`，共 4 个点。
- 所有度数和历史仅来自 train.txt。测试时也不自动把验证交互并入历史。

默认 mean 是原版对照；29 组搜索优先评估 mean。
训练时使用平均聚合的 MLP 未必适应其他 B 的输入尺度，因此负结果不能直接否定重训版的有效性。

## 使用命令

以下命令在 AlphaFree 目录执行。它们**会执行评估，不是 dry-run**，但均不训练。

```bash
# 1. mean 对照，只评估验证集
python eval_alphafree3.py --dataset Movie --device cuda:0 --b-mode mean --output weights/af3_movie_mean_valid

# 可选：先用 32 位验证用户做流程短跑，不能据此选参
python eval_alphafree3.py --dataset Movie --device cuda:0 --search --eval-users 32 --output weights/af3_movie_smoke

# 2. 固定一个 checkpoint，正式评估 29 个 B；仅在验证集选参
python -u eval_alphafree3.py --dataset Movie --device cuda:0 --search --output weights/af3_movie_search_v1

# 3. 全部 29 组完成后，读取锁定的配置，只对该 B 评估测试集
python eval_alphafree3.py --dataset Movie --device cuda:0 --split test --selection weights/af3_movie_search_v1/selection.json --output weights/af3_movie_selected_test

# 锁定配置后，可单独报告同一 checkpoint 的 mean 测试对照
python eval_alphafree3.py --dataset Movie --device cuda:0 --split test --b-mode mean --output weights/af3_movie_mean_test

# 使用自己的 checkpoint；正式测试时也必须传入同一个 checkpoint
python eval_alphafree3.py --dataset Movie --device cuda:0 --checkpoint /path/to/epoch=99.checkpoint.pth.tar --search --output weights/af3_custom_search

# 单个自定义 B 的验证评估
python eval_alphafree3.py --dataset Movie --device cuda:0 --b-mode degree --b-a 0.25 --b-b 0.75 --output weights/af3_custom_B
```

数据集接受 Movie 等别名或 amazon_movie 等目录名。CPU 环境将 device 改为 cpu。
默认 device=auto，优先 CUDA，否则 CPU。默认 top-k=20，评估 batch=128，物品编码 batch=1024，CPU threads=4。
GPU 上原始语义矩阵常驻显存；大数据集需预留显存，降低评估 batch 不会消除这部分固定占用。
改变评估 batch 主要影响评分矩阵开销，改变 encode batch 影响物品 MLP 编码时的中间张量。

## 选参、输出与续跑

- `--search` 仅允许 split=valid；不允许在测试集上搜索 29 组。
- 每个 B 输出 `<B-id>.json`：指标、权重/特征/数据哈希、参数哈希及短跑标记。
- manifest.json 固定本次数据、checkpoint、代码和运行条件；summary.json 汇总已完成项。
- 全部 29 组、全量验证用户评估完成后才生成 selection.json，按验证 Recall@K 选 B。
- 相同 Recall 时优先按执行顺序取结果，mean 排在第一位。不会用测试指标打破平局。
- `--eval-users` 非零标记 smoke_test，不生成最优选择文件。
- `--max-settings 1` 可先只完成 mean 的全量评估；相同命令去掉限制后会跳过已完成项并继续。
- 不完整搜索的 best_validation 只是暂定，不生成 selection.json。
- 选定配置的测试入口核对 checkpoint、语义特征、三份划分及代码指纹，拒绝短跑或不完整选择文件。
- 自己指定 `--split test --b-mode ...` 属于手动评估；不要手动扫描测试集并据此挑选 B。

同一输出目录只允许一个进程。代码、checkpoint、数据或运行条件变化时需换新目录，
不能修改旧 manifest 强行续跑。原权重与数据均只读，不会覆盖训练结果。
中断时已完成 B 的 JSON 可复用；异常退出可能遗留 running.lock，应核实进程结束后再处理具体锁文件。

## 评估协议与实验边界

复用本项目纯 PyTorch 全目录评估器：验证屏蔽 train+test，测试屏蔽 train+valid，当前目标保留。
指标为 Recall、NDCG、HR。测试标签仍用于原版规定的候选屏蔽，不用于构造 B 或选择参数。
与原版 C++ 后端采用相同指标定义/屏蔽协议，但不同设备、批大小和排序后端可能产生浮点/并列排名差异。
先核对 mean 对照，再解释其他 B 的改动效果；不保证逐位重现论文数值。

冻结评估无训练随机种子；重复评估同一 checkpoint 不是多 seed 训练复验。
如果需要多 seed 证据，应分别使用不同训练 seed 得到的 checkpoint。
AlphaFree3 仍使用语义信息，不能标为纯 CF；本实验检验的是无需重新训练的聚合替换效果。

## 验证

```bash
python -m unittest discover -s tests -p 'test_alphafree3.py' -v
```

覆盖 mean 与原版评分一致、29 个 B 的公式与冻结参数/物品缓存不变、严格权重加载、
验证搜索与最终测试隔离、完成任务跳过、短跑/不完整选择拒绝及运行锁保护。
CUDA 对照测试仅在有 GPU 时运行。本机 CPU 可直接运行本入口，不需要模拟 CUDA。

2026-10-20 验证记录：全套 45 项测试中 41 项通过、4 项因无 CUDA 跳过。
Movie 官方 checkpoint 在 CPU 上完成全部 29 个 B 的短跑（前 32 位验证用户），
参数哈希不变，重复执行全部跳过；结果位于 `weights/alphafree3_movie_smoke_20261020/`。
该短跑不生成 selection.json，也不用于性能结论。
同一 checkpoint、原始 mean 缓存和 32 位用户下，AlphaFree3 与原版评分最大绝对差为
`2.086162567138672e-07`（浮点数值误差范围）。未执行全量正式搜索或 GPU 验证。
