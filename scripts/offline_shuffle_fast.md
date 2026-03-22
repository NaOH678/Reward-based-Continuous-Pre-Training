# offline_shuffle_fast.py 逻辑讲解

这个脚本实现了大规模预训练数据的**全局随机打乱（shuffle）**。

## 整体目标

将多个域（domain）的预训练数据文件打乱混合，确保训练时每个 batch 包含来自不同域的样本，避免模型连续看到同一域的数据导致灾难性遗忘。

## 核心流程

```
输入：多个域的 .npy/.bin 文件（按域分目录存储）
      domain_A/file1.bin, domain_A/file2.bin, ...
      domain_B/file1.bin, ...

输出：全局打乱后的 shard 文件
      shard_0000.bin, shard_0001.bin, ... shard_0063.bin
```

---

## Phase 1: 收集文件信息

```python
file_seq_info = []  # [(path, num_seqs, global_offset), ...]
```

扫描输入目录，计算每个文件包含多少个序列：

```
文件 A/file1.bin: 10000 tokens → 10000 / 4096 = 2 个序列
文件 A/file2.bin: 50000 tokens → 12 个序列
文件 B/file1.bin: 80000 tokens → 19 个序列
...
```

同时记录每个文件的**全局偏移**（global_offset），用于后续定位：

```
A/file1: offset=0,  num_seqs=2   → 全局序列 [0, 1]
A/file2: offset=2,  num_seqs=12  → 全局序列 [2, 13]
B/file1: offset=14, num_seqs=19  → 全局序列 [14, 32]
```

---

## Phase 2: 生成全局置换

核心思想：**先决定每个序列的目标位置，再按顺序读取写入**

```python
shuffled_positions = rng.permutation(total_sequences)
# 例如 total_sequences = 1000
# shuffled_positions = [537, 12, 891, 3, ...]
# 含义：原始序列 0 写入位置 537，原始序列 1 写入位置 12，...
```

**为什么这样设计？**

- 朴素方法：随机读取 → 大量随机 I/O，极慢
- 本方法：顺序读取每个文件，根据预计算的位置写入 → 读取是顺序的，写入是 mmap 随机写（性能可接受）

**分布验证**：检查打乱后各域在输出中的分布是否均匀

```
检查前 50000 个输出位置:
  domain_A: 实际=25123, 期望=25000, 偏差=0.5%
  domain_B: 实际=24877, 期望=25000, 偏差=0.5%
✓ 分布均匀
```

---

## Phase 3: 预分配输出文件

创建固定大小的 memory-mapped 文件：

```python
# 64 个 shard，每个存储 total_sequences / 64 个序列
shard_0000.bin: shape=(15625, 4096), dtype=uint32
shard_0001.bin: shape=(15625, 4096), dtype=uint32
...
```

使用 `np.memmap` 直接映射到磁盘，避免内存爆炸。

---

## Phase 4: 顺序读取 → 并行随机写入

这是核心处理逻辑：

```python
def process_file(file_task):
    """处理单个文件"""
    file_idx, path, num_seqs, global_offset = file_task

    # 1. 顺序读取整个文件（高效）
    tokens = np.fromfile(path, dtype=dtype)

    # 2. 遍历每个序列
    for local_idx in range(num_seqs):
        global_seq_idx = global_offset + local_idx

        # 3. 查找目标位置
        target_pos = shuffled_positions[global_seq_idx]  # 例如 537
        shard_idx, local_pos = find_shard_and_local_pos(target_pos)
        # shard_idx=8, local_pos=412 表示写入 shard_0008.bin 的第 412 行

        # 4. 读取序列并写入
        seq = tokens[local_idx * seq_len : (local_idx + 1) * seq_len]
        with shard_locks[shard_idx]:  # 加锁防止并发写入冲突
            shard_mmaps[shard_idx][local_pos] = seq
```

**并行优化**：
- 多个文件用 `ThreadPoolExecutor` 并行处理
- 每个 shard 有独立的锁，不同 shard 的写入不会互相阻塞

**shard 定位优化**：
```python
shard_cumsum = [0, 15625, 31250, 46875, ...]  # 前缀和

def find_shard_and_local_pos(target_pos):
    shard_idx = np.searchsorted(shard_cumsum[1:], target_pos, side='right')
    local_pos = target_pos - shard_cumsum[shard_idx]
    return shard_idx, local_pos
```
使用二分查找 O(log n)，而非线性遍历 O(n)。

---

## Phase 5: 保存元信息

```json
{
  "version": 2,
  "shuffle_type": "global_permutation_fast",
  "seq_len": 4096,
  "total_sequences": 1000000,
  "num_shards": 64,
  "domain_stats": {"domain_A": 500000, "domain_B": 500000},
  "shard_sizes": [15625, 15625, ...]
}
```

---

## 可视化示例

```
原始数据（按域连续存储）:
┌─────────────────────────────────────────────────────┐
│ A A A A A A A A │ B B B B B B │ C C C C C C C C C C │
└─────────────────────────────────────────────────────┘

全局置换后:
shuffled_positions = [17, 3, 25, 8, 1, 22, ...]
                      ↓  ↓   ↓  ↓  ↓   ↓
原始序列 0 → 位置 17
原始序列 1 → 位置 3
...

输出（全局打乱）:
┌─────────────────────────────────────────────────────┐
│ B A C A B C A C │ A B A C B A │ C A B C A B C A B A │
└─────────────────────────────────────────────────────┘
```

---

## 性能特点

| 操作 | I/O 模式 | 性能 |
|------|----------|------|
| 读取文件 | 顺序读 | 很快 |
| 写入 mmap | 随机写 | 较快（OS 会批量刷盘）|
| 查找 shard | O(log n) | 很快 |
| 多线程 | 文件级并行 | 可扩展 |

50B tokens 约 10-20 分钟完成。

---

## 使用方法

```bash
python scripts/offline_shuffle_fast.py \
    --input_dir /path/to/domino-50B \
    --output_dir /path/to/domino-50B-shuffled \
    --seq_len 4096 \
    --num_shards 64 \
    --num_workers 8 \
    --seed 42
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | 必填 | 输入数据目录 |
| `--output_dir` | 必填 | 输出目录 |
| `--seq_len` | 4096 | 序列长度 |
| `--num_shards` | 64 | 输出分片数 |
| `--num_workers` | 4 | 并行线程数 |
| `--seed` | 42 | 随机种子 |
| `--dtype` | uint32 | 数据类型 |
