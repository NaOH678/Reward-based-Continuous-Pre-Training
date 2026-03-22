#!/usr/bin/env python3
"""
测试 shuffle 是否破坏语义
"""

import numpy as np
from transformers import AutoTokenizer

# 加载一个样本文件
file_path = "../../formalverification-shared/shichaojian/domino-50B-shuffled/shard_0000.bin"

print("=" * 80)
print("测试: Shuffle 是否破坏语义?")
print("=" * 80)

# 加载 tokens
tokens = np.fromfile(file_path, dtype = np.uint32)
print(f"\n文件: {file_path}")
print(f"总 tokens: {len(tokens):,}")

# 取前 200 个 tokens 作为样本
sample_tokens = tokens[:500]

# 加载 tokenizer 来解码
print("\n加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/mnt/shared-storage-user/shichaojian/OLMo-1B/dolma-tokenizer")

# 显示原始文本
print("\n" + "=" * 80)
print("shuffl 后的文本 (前 200 tokens):")
print("=" * 80)
shuffled_text = tokenizer.decode(sample_tokens)
print(shuffled_text)

# # 模拟 token-level shuffle
# print("\n" + "=" * 80)
# print("如果进行 token-level shuffle (我之前的错误代码):")
# print("=" * 80)
# shuffled_tokens = sample_tokens.copy()
# np.random.seed(42)
# np.random.shuffle(shuffled_tokens)
# shuffled_text = tokenizer.decode(shuffled_tokens)
# print(shuffled_text)

# # 显示差异
# print("\n" + "=" * 80)
# print("结论:")
# print("=" * 80)
# print("原始文本有意义:", "✓" if len(original_text.split()) > 10 else "✗")
# print("Shuffled 文本是乱码:", "✓" if shuffled_text != original_text else "✗")

# # 分析我之前的代码做了什么
# print("\n" + "=" * 80)
# print("分析 shuffle_pretokenized_data.py 中的代码:")
# print("=" * 80)

# print("""
# two_level_shuffle() 函数中:
#     chunk_tokens = concatenated[:chunk_size]  # 取 10M tokens
#     rng.shuffle(chunk_tokens)                 # ← 这行会 shuffle tokens!

# 这确实会打乱 token 顺序，破坏语义！
# """)

# print("\n⚠️  警告: 原始代码会破坏文本语义！")
# print("✓ 需要改成 document-level shuffle")
