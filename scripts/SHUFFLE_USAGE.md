# 离线 Shuffle 使用指南

## 方案概述

本方案采用**全局索引置换**实现离线物理 shuffle，确保：
1. **真正的全局随机**：不同域的数据完美混合
2. **语义完整性**：每个序列（4096 tokens）内部保持完整
3. **训练速度无损**：shuffle 后的数据顺序读取，无随机 I/O

## 核心原理

```
❌ Buffer Shuffle（有问题）:
   - 顺序读取数据到 buffer，buffer 满后 shuffle
   - 如果 dclm 占 50%，buffer 内可能全是 dclm
   - 结果：域聚集问题未解决

✅ 全局索引置换（本方案）:
   1. 扫描所有文件，构建全局索引 [(file_idx, local_idx), ...]
   2. 对索引进行全局随机置换（不是 buffer shuffle！）
   3. 按置换顺序读取序列，顺序写入新文件

   结果：任意位置都可能来自任意域，完美随机
```

## 快速开始

### 1. 执行离线 Shuffle

```bash
python scripts/offline_shuffle.py \
    --input_dir /path/to/domino-50B \
    --output_dir /path/to/domino-50B-shuffled \
    --seq_len 4096 \
    --num_shards 64 \
    --seed 42
```

**参数说明：**
- `--input_dir`: 原始 token 数据目录（支持嵌套目录结构）
- `--output_dir`: 输出目录
- `--seq_len`: 序列长度（必须与训练时一致）
- `--num_shards`: 输出 shard 数量（建议 >= GPU 数量）
- `--seed`: 随机种子（可复现）

**时间估算：**
- 50B tokens 约 30-60 分钟（取决于存储速度）
- 内存占用约 100-200MB（只存索引）

### 2. 验证 Shuffle 质量

```bash
python scripts/verify_offline_shuffle.py \
    --original_dir /path/to/domino-50B \
    --shuffled_dir /path/to/domino-50B-shuffled \
    --tokenizer_path /path/to/tokenizer \
    --seq_len 4096
```

验证项：
1. Token 总数一致性
2. Token 分布一致性
3. Shuffle 质量（相邻序列无关联）
4. 语义完整性（解码样本检查）

### 3. 训练时使用

```bash
python train.py \
    --training.dataset /path/to/domino-50B-shuffled \
    --training.seq_len 4096 \
    ...
```

## 输出格式

```
domino-50B-shuffled/
├── shard_0000.bin          # 二进制 token 序列
├── shard_0001.bin
├── ...
├── shard_0063.bin
└── shuffle_metadata.json   # 元信息
```

每个 shard 包含：
- 连续的 token 序列（每个 seq_len=4096 tokens）
- 已经充分打乱，训练时顺序读取即可

## 技术细节

### Sequence-level vs Token-level Shuffle

```
❌ Token-level Shuffle（破坏语义）:
   原始: [今, 天, 天, 气, 很, 好, ...]
   打乱: [好, 气, 今, 很, 天, 天, ...]  → 乱码！

✅ Sequence-level Shuffle（本方案）:
   原始序列排列:
     seq_0: [今天天气很好... (4096 tokens)]  ← math 域
     seq_1: [The quick brown... (4096 tokens)] ← web 域

   打乱后:
     seq_1: [The quick brown... (4096 tokens)] ← web 域
     seq_0: [今天天气很好... (4096 tokens)]  ← math 域

   每个序列内部保持完整，只是序列顺序打乱
```

### 全局索引置换流程

```
Phase 1: 构建索引（不加载数据，只记录位置）
┌────────────────────────────────────────────────────────┐
│ index = [                                              │
│   (dclm/0.npy, 0),   # global_idx = 0                 │
│   (dclm/0.npy, 1),   # global_idx = 1                 │
│   ...                                                  │
│   (math/0.npy, 0),   # global_idx = 5,000,000         │
│   ...                                                  │
│ ]                                                      │
│ 内存占用: ~100MB (10M indices × 8 bytes)               │
└────────────────────────────────────────────────────────┘
                            ↓
Phase 2: 全局置换（关键！）
┌────────────────────────────────────────────────────────┐
│ rng = np.random.default_rng(seed=42)                  │
│ shuffled_order = rng.permutation(10_000_000)          │
│                                                        │
│ 原始: [0, 1, 2, 3, ..., 9999999]                      │
│ 置换: [7823456, 234, 9012345, 5000123, ...]           │
│        ↑code   ↑dclm  ↑code    ↑math                  │
│                                                        │
│ 每个位置都可能来自任意域！                               │
└────────────────────────────────────────────────────────┘
                            ↓
Phase 3: 按置换顺序读取并写入
┌────────────────────────────────────────────────────────┐
│ for global_idx in shuffled_order:                     │
│     file, local = index[global_idx]                   │
│     seq = read_sequence(file, local)                  │
│     write_to_shard(seq)                               │
│                                                        │
│ 输出: shard_0000.bin, shard_0001.bin, ...             │
│       每个 shard 内部已是跨域混合                       │
└────────────────────────────────────────────────────────┘
```

### 为什么能解决域聚集问题？

```
假设数据分布:
  dclm: 50% (5M 序列)
  math: 25% (2.5M 序列)
  code: 25% (2.5M 序列)

全局置换后:
  - 任意位置是 dclm 的概率: 50%
  - 任意位置是 math 的概率: 25%
  - 任意位置是 code 的概率: 25%

写出的 shard 文件中:
  每 1000 个连续序列，期望包含:
  - ~500 个 dclm 序列
  - ~250 个 math 序列
  - ~250 个 code 序列

  且它们的顺序是随机的，不会出现 "连续 N 个都是 dclm"
```

## 常见问题

### Q1: Shuffle 需要多少磁盘空间？

需要 **2x** 原始数据空间（原始 + shuffled）。

如果空间紧张，可以：
1. Shuffle 完成后删除原始数据
2. 或使用增量删除（边 shuffle 边删除已处理的原文件）

### Q2: 可以只 shuffle 部分数据吗？

不建议。应该一次性 shuffle 全部数据，确保所有域充分混合。

### Q3: 不同 epoch 可以用不同的 shuffle 吗？

本方案是离线物理 shuffle，训练时顺序读取。如需每个 epoch 不同顺序，可以：
1. 生成多份 shuffled 数据（用不同 seed）
2. 或在 dataloader 中添加轻量级的 shard 级 shuffle

### Q4: 如何验证 shuffle 是否正确？

```bash
# 完整验证
python scripts/verify_offline_shuffle.py \
    --original_dir /path/to/original \
    --shuffled_dir /path/to/shuffled \
    --tokenizer_path /path/to/tokenizer

# 快速验证（跳过分布检查）
python scripts/verify_offline_shuffle.py \
    --shuffled_dir /path/to/shuffled \
    --tokenizer_path /path/to/tokenizer \
    --skip-distribution
```

### Q5: 训练时 loss 曲线应该是什么样？

正确 shuffle 后：
- Loss 应该**平滑下降**
- **不应该**有明显的阶梯状波动
- 如果仍有波动，可能是其他问题（学习率、batch size 等）

## 文件说明

```
scripts/
├── offline_shuffle.py          # 离线 shuffle 主脚本
├── verify_offline_shuffle.py   # 验证脚本
└── SHUFFLE_USAGE.md            # 本文档
```

## 示例输出

```
$ python scripts/offline_shuffle.py --input_dir /data/domino-50B --output_dir /data/domino-50B-shuffled

============================================================
离线全局 Shuffle
============================================================
输入目录: /data/domino-50B
输出目录: /data/domino-50B-shuffled
序列长度: 4096
Shard 数量: 64
随机种子: 42
============================================================

============================================================
Phase 1: 收集文件信息
============================================================
扫描目录: /data/domino-50B
找到 1234 个文件，正在分析...
有效文件: 1200
总 tokens: 50,000,000,000 (50.00B)
总序列数: 12,207,031
序列长度: 4096
各域数据分布:
  dclm: 6,103,515 序列 (50.0%)
  basic_math: 3,051,757 序列 (25.0%)
  code: 3,051,759 序列 (25.0%)

============================================================
Phase 2: 构建并置换全局索引
============================================================
构建全局索引: 12,207,031 个序列
全局置换 12,207,031 个索引 (seed=42)
置换后前 10000 个序列的域分布:
  dclm: 49.8% (期望 50.0%)
  basic_math: 25.3% (期望 25.0%)
  code: 24.9% (期望 25.0%)

============================================================
Phase 3: 按置换顺序写入 shards
============================================================
写入数据: 100%|████████████████████| 12207031/12207031 [35:42<00:00]

============================================================
✓ 离线 Shuffle 完成！
============================================================
输出目录: /data/domino-50B-shuffled
Shard 数量: 64
总序列: 12,207,031

训练时使用:
  --training.dataset /data/domino-50B-shuffled
```
