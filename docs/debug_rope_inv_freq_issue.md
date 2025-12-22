# OLMo Mid-Training Loss 异常问题排查与修复

## 问题背景

使用 OLMo 官方的 Dolmino-50B 数据集在 OLMo-1B 模型上进行 mid-training（继续预训练），但效果远不如官方技术报告：

| 指标 | 官方结果 | 实际结果 |
|------|----------|----------|
| GSM8K | 43% | 4% |
| 初始 Loss | ~2.2 | ~6 |
| 最终 Loss | ~1.83 | ~1.85 |

## 问题分析过程

### 1. 初步排查：框架差异

首先对比了官方 OLMo-core 框架和我们的 flame 框架（基于 TorchTitan）：

| 配置项 | OLMo-core | flame |
|--------|-----------|-------|
| LR Scheduler | LinearWithWarmup | Cosine |
| Weight Decay | 0.033 | 0.1 |
| Z-loss | 有 | 无 |
| Optimizer | SkipStepAdamW | 标准 AdamW |

这些差异可能影响最终效果，但无法解释初始 loss 的巨大差异。

### 2. 关键发现：初始 Loss 异常

观察 loss 曲线发现关键线索：

```
随机初始化 loss ≈ ln(vocab_size) ≈ ln(100000) ≈ 11.5
官方初始 loss ≈ 2.2
我们初始 loss ≈ 6
```

**初始 loss 在随机初始化(~12)和预训练模型(~2.2)之间，说明权重只部分加载成功。**

### 3. Checkpoint 验证

编写诊断脚本验证 DCP checkpoint 加载：

```python
# 对比 HuggingFace 直接加载 vs DCP 加载
hf_model = AutoModelForCausalLM.from_pretrained(...)  # loss ≈ 3.12
dcp_model = load_from_dcp(...)                         # loss ≈ 4.16 或 NaN
```

第一版诊断结果：
- ✅ 所有 179 个 state_dict keys 匹配
- ❌ Loss 是 NaN

### 4. NaN 问题定位

添加 `post_init()` 调用后 NaN 消失，但 loss 仍不匹配：

```
post_init 后 inv_freq[:5]: [0.0, 0.0, 0.0, 0.0, 0.0]  # 全是 0！
inv_freq 差异: 1.00e+00
DCP 模型 loss: 4.1648
HF 模型 loss:  3.1216
```

---

## 发现 inv_freq 问题的思考过程

### 思考 1：Loss 的数学含义

**关键洞察：Loss 值本身包含重要信息**

```
随机初始化的 LM 模型，其 loss 约等于 ln(vocab_size)
因为随机模型对每个 token 的预测是均匀分布的
P(token) ≈ 1/vocab_size
loss = -ln(P) = ln(vocab_size) ≈ ln(100000) ≈ 11.5
```

当用户指出初始 loss 是 6（而不是 2.2）时，这个数值给了关键线索：
- 如果 loss ≈ 11.5：完全随机初始化
- 如果 loss ≈ 2.2：权重完全加载
- **如果 loss ≈ 6：权重部分加载，或某些关键组件失效**

### 思考 2：为什么是"部分"失效？

既然所有 179 个 state_dict keys 都匹配了，为什么还会有问题？

可能的原因：
1. ~~某些权重虽然 key 匹配，但值不对~~ （已验证权重值正确）
2. ~~数据类型不匹配~~ （已验证都是 float32）
3. **某些组件不在 state_dict 中，但对模型至关重要**

### 思考 3：什么组件会不在 state_dict 中？

PyTorch 模型有两类状态：
1. **Parameters**: 可学习参数，一定在 state_dict 中
2. **Buffers**: 非学习参数，可能是 persistent 或 non-persistent

```python
# Persistent buffer - 会保存到 state_dict
self.register_buffer("buf", tensor, persistent=True)

# Non-persistent buffer - 不会保存到 state_dict！
self.register_buffer("buf", tensor, persistent=False)
```

### 思考 4：Transformer 中有什么重要的 Buffer？

回顾 Transformer 架构，有哪些关键的非学习参数：
1. **Attention mask** - 通常动态生成
2. **Position encodings** - 取决于实现方式
   - 绝对位置编码：通常是 embedding，在 state_dict 中
   - **RoPE 位置编码：使用 inv_freq buffer 计算**

### 思考 5：验证 inv_freq 假设

在诊断脚本中添加 inv_freq 检查：

```python
# 打印 HF 模型的 inv_freq
print(f"HF inv_freq[:5]: {hf_model.model.rotary_emb.inv_freq[:5]}")
# 输出: [1.0, 0.866, 0.749, 0.649, 0.562]

# 打印 DCP 模型的 inv_freq（post_init 后）
print(f"DCP inv_freq[:5]: {dcp_model.model.rotary_emb.inv_freq[:5]}")
# 输出: [0.0, 0.0, 0.0, 0.0, 0.0]  <-- 全是 0！
```

**确认：inv_freq 没有正确初始化！**

### 思考 6：为什么 post_init() 没有初始化 inv_freq？

查看 HuggingFace 源码：

```python
# transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, ...):
        inv_freq = 1.0 / (self.base ** (...))
        # 关键：persistent=False！
        self.register_buffer("inv_freq", inv_freq, persistent=False)
```

`post_init()` 的实现：
- 它主要处理权重初始化和 tie_weights
- 不会重新计算 non-persistent buffer
- 假设这些 buffer 在 `__init__` 时已正确创建

但当使用 `meta device → to_empty()` 流程时：
- `__init__` 在 meta device 上执行，inv_freq 是虚拟的
- `to_empty()` 将模型移到实际设备，但 inv_freq 变成空张量（全 0）
- `post_init()` 不知道需要重新计算 inv_freq

### 思考 7：验证 DCP Checkpoint 内容

最后验证 inv_freq 确实不在 checkpoint 中：

```python
# 检查 DCP checkpoint metadata
with open('.metadata', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')

if 'inv_freq' in content:
    print("Found!")
else:
    print("NOT found!")  # <-- 实际输出
```

**完整确认：inv_freq 是 non-persistent buffer，不在 checkpoint 中，且 post_init() 无法正确初始化它。**

---

## 技术细节

### RoPE (Rotary Position Embedding) 原理

RoPE 使用 `inv_freq` buffer 计算位置编码：

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
# theta 默认为 10000.0
# dim 为 head_dim = hidden_size / num_attention_heads
```

RoPE 通过旋转 query 和 key 向量来注入位置信息。如果 inv_freq 全是 0：
- 所有位置的旋转角度都是 0
- 等价于没有位置编码
- 模型退化为"词袋模型"，无法理解词序

### 问题发生的流程

```python
# 1. 在 meta device 创建模型
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
    # inv_freq 在 meta device 上，未初始化

# 2. 移到实际设备
model.to_empty(device=device)
# inv_freq 现在是空张量（全 0）

# 3. 调用 post_init()
model.post_init()
# post_init() 无法正确初始化 inv_freq

# 4. 加载 DCP checkpoint
DCP.load(state_dict, checkpoint_id=ckpt_path)
model.load_state_dict(state_dict, strict=False)
# inv_freq 不在 checkpoint 中，保持为 0！

# 结果：RoPE 位置编码完全失效
```

## 修复方案

### 添加手动初始化函数

```python
def init_rope_inv_freq(model, device):
    """
    手动初始化 RoPE 的 inv_freq buffer。
    """
    config = model.config
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    head_dim = config.hidden_size // config.num_attention_heads

    # 计算 inv_freq
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    ))
    inv_freq = inv_freq.to(device)

    # 设置到所有 rotary_emb 模块
    for name, module in model.named_modules():
        if hasattr(module, 'inv_freq') and 'rotary' in name.lower():
            if module.inv_freq.shape == inv_freq.shape:
                module.inv_freq.copy_(inv_freq)
```

### 修改训练代码

```python
# flame/train.py
model.to_empty(device=init_device)
with torch.no_grad():
    model.post_init()
    # 关键修复：手动初始化 RoPE inv_freq buffer
    init_rope_inv_freq(model, init_device)
model.train()
```

## 修复后验证

```
============================================================
Checkpoint 加载诊断（修复版 v3 - 手动初始化 inv_freq）
============================================================

1. 加载 HuggingFace 原始模型...
   HuggingFace 模型 loss: 3.1216
   HF inv_freq[:5]: [1.0, 0.8660..., 0.7498..., 0.6493..., 0.5623...]

2. 从 DCP checkpoint 加载模型...
   计算 inv_freq: rope_theta=10000.0, head_dim=64
   初始化 model.rotary_emb.inv_freq ✓
   inv_freq 与 HF 匹配: ✅

5. 测试 DCP 模型的 loss...
   DCP 模型 loss: 3.1216
   HF 模型 loss:  3.1216
   差异: 0.0000
   ✅ Loss 完全匹配！Checkpoint 加载正确！
```

## 经验总结

### 1. Loss 值是重要的诊断信号

- 知道理论值（如 ln(vocab_size)）有助于判断问题严重程度
- Loss 介于两个极端之间往往意味着"部分失效"

### 2. State Dict 匹配 ≠ 模型正确

- Non-persistent buffers 不在 state_dict 中
- 需要额外检查关键 buffer 的值

### 3. Meta Device 初始化的陷阱

使用 `meta device → to_empty() → post_init()` 流程时：
- `post_init()` 可能无法正确初始化所有 buffer
- 需要手动处理 non-persistent buffer

### 4. 追踪问题的方法论

1. **观察异常现象**：Loss 6 而不是 2.2
2. **量化分析**：6 在 2.2 和 11.5 之间意味着什么
3. **排除法**：权重匹配但效果不对 → 看其他组件
4. **知识回顾**：Transformer 中有哪些非学习参数
5. **假设验证**：打印 inv_freq 值确认假设
6. **根因分析**：追溯到 persistent=False 的注册方式

### 5. 相关文件

- 诊断脚本：`diagnose_checkpoint_loading.py`
- 训练代码修复：`flame/train.py` (添加 `init_rope_inv_freq` 函数)
- DCP 转换脚本：`convert_hf_to_dcp_fixed.py`

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [HuggingFace Transformers - LlamaRotaryEmbedding](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html)
