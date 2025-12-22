#!/usr/bin/env python3
"""
诊断 checkpoint 加载问题的脚本（修复版 v3）。
在训练环境中运行以检查权重是否正确加载。

修复：手动初始化 inv_freq buffer，因为 post_init() 在 to_empty() 后不能正确工作。

用法：
torchrun --nproc_per_node=8 diagnose_checkpoint_loading.py
"""

import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def init_rope_inv_freq(model, device, rank=0):
    """
    手动初始化 RoPE 的 inv_freq buffer。
    这是因为 post_init() 在 to_empty() 后不能正确初始化 buffers。
    """
    config = model.config

    # 获取 RoPE 参数
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    head_dim = config.hidden_size // config.num_attention_heads

    # 计算 inv_freq: 1 / (theta ^ (2i / dim))
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq = inv_freq.to(device)

    if rank == 0:
        print(f"   计算 inv_freq: rope_theta={rope_theta}, head_dim={head_dim}")
        print(f"   inv_freq shape: {inv_freq.shape}, values[:5]: {inv_freq[:5].tolist()}")

    # 找到并设置所有 rotary_emb.inv_freq buffers
    initialized = False
    for name, module in model.named_modules():
        if hasattr(module, 'inv_freq') and 'rotary' in name.lower():
            if module.inv_freq.shape == inv_freq.shape:
                module.inv_freq.copy_(inv_freq)
                if rank == 0:
                    print(f"   初始化 {name}.inv_freq ✓")
                initialized = True
            else:
                if rank == 0:
                    print(f"   ⚠️ {name}.inv_freq shape 不匹配: {module.inv_freq.shape} vs {inv_freq.shape}")

    # 也处理顶层 rotary_emb
    if hasattr(model.model, 'rotary_emb') and hasattr(model.model.rotary_emb, 'inv_freq'):
        if model.model.rotary_emb.inv_freq.shape == inv_freq.shape:
            model.model.rotary_emb.inv_freq.copy_(inv_freq)
            if rank == 0:
                print(f"   初始化 model.rotary_emb.inv_freq ✓")
            initialized = True

    if not initialized and rank == 0:
        print("   ⚠️ 未找到需要初始化的 inv_freq buffer！")


def main():
    # 初始化分布式
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("Checkpoint 加载诊断（修复版 v3 - 手动初始化 inv_freq）")
        print("=" * 60)

    # 1. 直接从 HuggingFace 加载模型（作为参考）
    if rank == 0:
        print("\n1. 加载 HuggingFace 原始模型...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        '/mnt/shared-storage-user/shichaojian/OLMo-1B',
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/shared-storage-user/shichaojian/OLMo-1B/dolma-tokenizer'
    )

    # 测试 HF 模型的 loss
    hf_model.eval()
    test_text = "The capital of France is Paris. Mathematics is the study of"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs, labels=inputs['input_ids'], output_hidden_states=True)
        hf_loss = hf_outputs.loss.item()

    if rank == 0:
        print(f"   HuggingFace 模型 loss: {hf_loss:.4f}")
        if hf_outputs.hidden_states:
            print(f"   hidden_states[-1] shape: {hf_outputs.hidden_states[-1].shape}")

    # 记录关键参数
    hf_embed_mean = hf_model.model.embed_tokens.weight.mean().item()
    hf_embed_std = hf_model.model.embed_tokens.weight.std().item()
    hf_inv_freq = hf_model.model.rotary_emb.inv_freq.clone()

    if rank == 0:
        print(f"   HF inv_freq[:5]: {hf_inv_freq[:5].tolist()}")
        print(f"   HF inv_freq shape: {hf_inv_freq.shape}")

    del hf_model
    torch.cuda.empty_cache()

    # 2. 从 DCP 加载模型
    if rank == 0:
        print("\n2. 从 DCP checkpoint 加载模型...")

    config = AutoConfig.from_pretrained('/mnt/shared-storage-user/shichaojian/OLMo-1B/config.json')

    if rank == 0:
        print(f"   config.rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
        print(f"   config.hidden_size: {config.hidden_size}")
        print(f"   config.num_attention_heads: {config.num_attention_heads}")

    # 在 meta device 创建模型
    with torch.device("meta"):
        dcp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # 移到实际设备
    dcp_model.to_empty(device=device)

    # 关键修复：手动初始化 inv_freq buffer
    if rank == 0:
        print("\n   手动初始化 inv_freq buffer...")

    with torch.no_grad():
        init_rope_inv_freq(dcp_model, device, rank)

    # 检查初始化后的 inv_freq
    if rank == 0:
        dcp_inv_freq = dcp_model.model.rotary_emb.inv_freq
        print(f"\n   初始化后 inv_freq[:5]: {dcp_inv_freq[:5].tolist()}")
        inv_freq_match = torch.allclose(dcp_inv_freq, hf_inv_freq.to(device), atol=1e-6)
        print(f"   inv_freq 与 HF 匹配: {'✅' if inv_freq_match else '❌'}")

    # 记录初始化后的统计（权重仍然是未初始化的）
    post_init_embed_mean = dcp_model.model.embed_tokens.weight.mean().item()
    post_init_embed_std = dcp_model.model.embed_tokens.weight.std().item()

    if rank == 0:
        print(f"\n   初始化后 embed_tokens: mean={post_init_embed_mean:.6f}, std={post_init_embed_std:.6f}")
        print(f"   （注：权重仍未加载，这些值是 to_empty 后的随机值）")

    # 获取 state_dict 用于加载
    state_dict = dcp_model.state_dict()

    # 加载 DCP checkpoint
    ckpt_path = '/mnt/shared-storage-user/shichaojian/OLMo-1B/checkpoint_final/checkpoint/step-0'

    if rank == 0:
        print(f"\n   从 {ckpt_path} 加载...")

    try:
        DCP.load(state_dict, checkpoint_id=ckpt_path)
        dcp_model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            print("   ✅ DCP.load 成功")
    except Exception as e:
        if rank == 0:
            print(f"   ❌ DCP.load 失败: {e}")
        dist.destroy_process_group()
        return

    # 3. 验证加载后的权重
    if rank == 0:
        print("\n3. 验证加载后的权重...")

    dcp_embed_mean = dcp_model.model.embed_tokens.weight.mean().item()
    dcp_embed_std = dcp_model.model.embed_tokens.weight.std().item()

    if rank == 0:
        print(f"   加载后 embed_tokens: mean={dcp_embed_mean:.6f}, std={dcp_embed_std:.6f}")
        print(f"   HF 原始 embed_tokens: mean={hf_embed_mean:.6f}, std={hf_embed_std:.6f}")

        # 检查是否接近 HF 值
        if abs(dcp_embed_mean - hf_embed_mean) < 0.01 and abs(dcp_embed_std - hf_embed_std) < 0.01:
            print("   ✅ embed_tokens 权重匹配！")
        else:
            print("   ❌ embed_tokens 权重不匹配！")
            print(f"      差异: mean={abs(dcp_embed_mean - hf_embed_mean):.6f}, std={abs(dcp_embed_std - hf_embed_std):.6f}")

        # 检查 inv_freq (buffer)
        inv_freq_diff = (dcp_model.model.rotary_emb.inv_freq - hf_inv_freq.to(device)).abs().max().item()
        print(f"   inv_freq 差异: {inv_freq_diff:.2e}")
        if inv_freq_diff < 1e-5:
            print("   ✅ inv_freq 匹配！")
        else:
            print("   ❌ inv_freq 不匹配！")

    # 4. 检查 NaN
    if rank == 0:
        print("\n4. 检查 NaN...")

        has_nan = False
        nan_params = []
        nan_buffers = []

        for name, param in dcp_model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
                has_nan = True

        for name, buf in dcp_model.named_buffers():
            if torch.isnan(buf).any():
                nan_buffers.append(name)
                has_nan = True

        if nan_params:
            print(f"   ❌ NaN in parameters: {nan_params}")
        if nan_buffers:
            print(f"   ❌ NaN in buffers: {nan_buffers}")
        if not has_nan:
            print("   ✅ 没有 NaN")

    # 5. 测试 DCP 模型的 loss
    if rank == 0:
        print("\n5. 测试 DCP 模型的 loss...")

    dcp_model.eval()
    with torch.no_grad():
        dcp_outputs = dcp_model(**inputs, labels=inputs['input_ids'])
        dcp_loss = dcp_outputs.loss.item()

    if rank == 0:
        print(f"   DCP 模型 loss: {dcp_loss:.4f}")
        print(f"   HF 模型 loss:  {hf_loss:.4f}")
        print(f"   差异: {abs(dcp_loss - hf_loss):.4f}")

        if torch.isnan(torch.tensor(dcp_loss)):
            print("   ❌ Loss 是 NaN！模型有问题")
        elif abs(dcp_loss - hf_loss) < 0.01:
            print("   ✅ Loss 完全匹配！Checkpoint 加载正确！")
        elif abs(dcp_loss - hf_loss) < 0.1:
            print("   ✅ Loss 基本匹配（差异 < 0.1）")
        else:
            print(f"   ❌ Loss 不匹配！")
            if dcp_loss > 10:
                print("   ⚠️  DCP loss 非常高，接近随机初始化 (ln(vocab_size)≈11.5)")
            elif dcp_loss > 5:
                print("   ⚠️  DCP loss 很高，可能大部分权重没有正确加载")

    # 6. 详细检查各层权重
    if rank == 0:
        print("\n6. 详细检查各层权重...")

        # 重新加载 HF 模型进行详细对比
        hf_model2 = AutoModelForCausalLM.from_pretrained(
            '/mnt/shared-storage-user/shichaojian/OLMo-1B',
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)

        hf_state = hf_model2.state_dict()
        dcp_state = dcp_model.state_dict()

        mismatch_count = 0
        for key in sorted(hf_state.keys()):
            if key in dcp_state:
                diff = (hf_state[key].to(device) - dcp_state[key].to(device)).abs().max().item()
                if diff > 1e-5:
                    print(f"   ❌ {key}: max_diff={diff:.6e}")
                    mismatch_count += 1
            else:
                print(f"   ❌ {key}: 缺失！")
                mismatch_count += 1

        if mismatch_count == 0:
            print("   ✅ 所有权重都匹配！")
        else:
            print(f"   ❌ 有 {mismatch_count} 个权重不匹配或缺失")

        del hf_model2

    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60)


if __name__ == "__main__":
    main()
