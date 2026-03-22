```bash
[rank0]:RuntimeError: aten._foreach_norm.Scalar: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!
```



DTensor is a torch.Tensor subclass. This means once a DTensor is created, it could be used in very similar way to torch.Tensor, including running different types of PyTorch operators as if running them in a single device, allowing proper distributed computation for PyTorch operators.





把你这两组 debug 放在一起，完整逻辑其实很清楚：

```python
OLMo:
{('DTensor', torch.float32, 'cuda:0'): 179,
 ('Tensor',  torch.float32, 'cuda:0'):   6}

Llama:
{('DTensor', torch.bfloat16, 'cuda:0'): 146,
 ('Tensor',  torch.float32,  'cuda:0'):   6}
```

核心差别只有一个：

**OLMo 里，主干 DTensor 梯度和 `future_predictor` 那 6 个普通 Tensor 梯度，dtype 同为 `float32`；Llama 里，两者 dtype 不同，前者是 `bfloat16`，后者还是 `float32`。**

而你已经确认报错点就是 `_foreach_norm` 那条路径，PyTorch 这边对应的问题现象也是：当同一次 foreach norm 调用里混入 `DTensor` 和普通 `torch.Tensor` 时，会报 `got mixed torch.Tensor and DTensor`。([GitHub][1])

---

## 先把前提链路理顺

### 1）为什么会同时出现 DTensor 和普通 Tensor

主干模型走了 TP/FSDP2 一类分布式并行路径时，参数会进入 DTensor 体系。PyTorch 文档明确说明，DTensor 是分布式张量抽象；而 FSDP2/`fully_shard` 会把 `model.parameters()` 从普通 `torch.Tensor` 转成 DTensor。([PyTorch 文档][2])

但 `future_predictor` 如果没有被同样 parallelize，它就还是一个普通 `nn.Module`，其参数/梯度仍然是普通 Tensor。于是就形成了两类梯度并存：

* 主干：DTensor grads
* aux (`future_predictor`)：普通 Tensor grads

这就是你 debug 里两类 key 同时存在的来源。([PyTorch 文档][2])

---

### 2）为什么不是一开始就必炸

因为“同时存在 DTensor 和 Tensor”还不够，**还得它们被送进同一个 foreach 分组**。
你已经观察到 `_foreach_norm` 会按 dtype 分组；虽然你没贴源码，但从你的 debug 现象可以直接反推出：dtype 是它的重要分流条件。于是：

* 同 dtype → 更可能进同一个 foreach 调用
* 不同 dtype → 先被拆成不同组，各自单独算

所以“混用是否真正发生”，不只是看类型，还看 dtype。

---

## 现在分别看 OLMo 和 Llama

---

## OLMo 为什么炸

你的 OLMo 统计是：

```python
('DTensor', torch.float32): 179
('Tensor',  torch.float32):   6
```

这意味着：

1. 主干的大部分梯度是 **DTensor + float32**
2. `future_predictor` 的 6 个梯度是 **普通 Tensor + float32**
3. 二者 **dtype 完全一样**

于是到了 `_foreach_norm`/grad norm 那一步，按 dtype 分组时，这 185 个梯度很可能被归到同一个 `float32` 组里。
这时候同组里既有 DTensor，又有普通 Tensor，于是触发 PyTorch 已知的混用错误：`got mixed torch.Tensor and DTensor`。([GitHub][1])

所以 OLMo 的因果链是：

```python
主干是 DTensor fp32
+ future_predictor 是普通 Tensor fp32
→ foreach 按 dtype 分组时进入同一组
→ 同一组里混 DTensor 和 Tensor
→ 报错
```

这就是它“必炸”的完整逻辑。

---

## Llama 为什么没炸

你的 Llama 统计是：

```python
('DTensor', torch.bfloat16): 146
('Tensor',  torch.float32):    6
```

这里差别在于：

1. 主干梯度是 **DTensor + bf16**
2. `future_predictor` 仍然是 **普通 Tensor + fp32**
3. 它们 **dtype 不同**

于是到了同样的 `_foreach_norm`/grad norm 阶段，按 dtype 分组时会先分裂成至少两组：

* 一个 `bfloat16` 组：只包含 DTensor
* 一个 `float32` 组：只包含普通 Tensor

这样每一次 foreach norm 调用内部都只看到了“同一种 dtype 的一类张量”，不会出现“DTensor 和普通 Tensor 同时进同一个调用”的情况，所以没触发那个 mixed error。([GitHub][1])

所以 Llama 的链路是：

```python
主干是 DTensor bf16
+ future_predictor 是普通 Tensor fp32
→ foreach 按 dtype 分组时被拆成不同组
→ 没有单次调用内的 DTensor/Tensor 混用
→ 不报错
```

注意，这不是“Llama 真没问题”，而是**它刚好因为 dtype 分流，躲开了这个 bug**。

---

## 为什么 OLMo 和 Llama 会出现这种 dtype 差异

这通常和并行/混合精度策略有关。FSDP/FSDP2 的 mixed precision policy 可以分别控制参数计算和 reduction 的 dtype；PyTorch 文档明确提到 mixed precision policy 会控制 parameter/reduction precision。([PyTorch 文档][3])

所以很可能是：

* OLMo 那套配置/实现路径下，主干最终用于 grad norm 的梯度表现成 `float32`
* Llama 那套配置/实现路径下，主干最终表现成 `bfloat16`

而 `future_predictor` 这 6 个参数因为没被纳入同一并行/精度路径，仍然保持普通 `float32 Tensor`

于是就形成了你现在看到的两个 debug 分布。

也就是说，**真正决定“炸不炸”的不是模型名字叫 OLMo 还是 Llama，而是：主干 DTensor grads 的 dtype，是否与 aux 普通 Tensor grads 的 dtype 撞到了一起。** ([PyTorch 文档][3])

---

## 那修复代码为什么有效

你已经确认后来加了：

```python
if len(list(future_predictor.parameters())) > 0:
    _parallelize_aux_module(future_predictor, world_mesh, parallel_dims, job_config)
    model_parts.append(future_predictor)
```

而 `_parallelize_aux_module(...)` 在 shard/cp 场景下会对它执行 `apply_fsdp(...)`。
FSDP2/`fully_shard` 的一个关键效果就是把参数从普通 Tensor 路径转换进 DTensor/FSDP 管理路径。([PyTorch 文档][4])

于是修复前后差别是：

### 修复前

* 主干：DTensor
* `future_predictor`：普通 Tensor

### 修复后

* 主干：DTensor
* `future_predictor`：也进入 FSDP/DTensor 路径

这样一来，即使 OLMo 那边主干还是 `float32`，也不再是“DTensor + 普通 Tensor”混用，而更像是“都在统一的分布式张量体系里”。
所以真正被消除的是：

> **张量类型路径不一致**

而不是单纯改了数值精度。

---

## 你这个复盘可以总结成一句非常准确的话

**OLMo 之所以报错，是因为主干梯度和 `future_predictor` 梯度同为 `float32`，导致 `_foreach_norm` 按 dtype 分组后，把 DTensor 和普通 Tensor 放进了同一个 foreach 调用；Llama 之所以没报错，只是因为主干是 `bfloat16 DTensor`，而 `future_predictor` 仍是 `float32 Tensor`，两者在 dtype 分组阶段被拆开，从而偶然避开了混用错误。后续对 `future_predictor` 进行 `_parallelize_aux_module(...)/FSDP` 包装，则从根源上统一了其张量表示路径，消除了该类 bug 的触发前提。** ([GitHub][1])

如果你要，我可以把这段直接整理成一版 PR 里的“Root cause / Why only OLMo failed / Fix”英文说明。

[1]: https://github.com/pytorch/pytorch/issues/121020?utm_source=chatgpt.com "[DTensor] `clip_grad_norm_` follow-ups · Issue #121020"
[2]: https://docs.pytorch.org/docs/stable/distributed.tensor.html?utm_source=chatgpt.com "torch.distributed.tensor"
[3]: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html?utm_source=chatgpt.com "torch.distributed.fsdp.fully_shard"
[4]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=chatgpt.com "Getting Started with Fully Sharded Data Parallel (FSDP2)"
