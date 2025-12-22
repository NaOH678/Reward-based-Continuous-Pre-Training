"""
对比 HuggingFace 原始模型 vs DCP checkpoint 转换后的模型在相同数据上的 loss
"""

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import sys


def test_hf_model(model_path, test_texts, tokenizer):
    """测试 HuggingFace 原始模型"""
    print("=" * 80)
    print("测试 1: HuggingFace 原始模型")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    print(f"✅ 模型加载完成")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    for i, text in enumerate(test_texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            losses.append(loss)
            print(f"   文本 {i+1} loss: {loss:.4f}")

    avg_loss = sum(losses) / len(losses)
    print(f"\n✅ HuggingFace 模型平均 loss: {avg_loss:.4f}")

    # 清理内存
    del model
    torch.cuda.empty_cache()

    return avg_loss, losses


def test_dcp_model(checkpoint_path, model_path, test_texts, tokenizer):
    """测试从 DCP checkpoint 加载的模型"""
    print("\n" + "=" * 80)
    print("测试 2: 从 DCP Checkpoint 加载的模型")
    print("=" * 80)

    # 检查是否在分布式环境下运行
    if "RANK" not in os.environ:
        print("❌ 警告: 不在 torchrun 环境下运行")
        print("   DCP 需要分布式环境才能正确加载")
        print("   请使用: torchrun --nproc_per_node=1 compare_hf_vs_dcp_loss.py")
        return None, []

    # 初始化分布式环境
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # 方法 1: 加载到空的 state_dict
    state_dict = {}
    try:
        DCP.load(state_dict, checkpoint_id=str(checkpoint_path))
        print(f"✅ DCP checkpoint 加载完成")
        print(f"   Keys: {len(state_dict)}")

        if len(state_dict) == 0:
            print("❌ 警告: checkpoint 为空！")
            return None, []

        # 检查是否有 lm_head.weight
        has_lm_head = 'lm_head.weight' in state_dict
        print(f"   包含 lm_head.weight: {has_lm_head}")

        # 创建模型
        print("\n创建模型并加载参数...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()

        # 加载参数
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"✅ 参数加载完成")
        print(f"   缺失的 keys: {len(missing_keys)}")
        print(f"   多余的 keys: {len(unexpected_keys)}")

        if missing_keys:
            print(f"   缺失的 keys 示例: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"   多余的 keys 示例: {unexpected_keys[:5]}")

        # 测试 loss
        losses = []
        for i, text in enumerate(test_texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                losses.append(loss)
                print(f"   文本 {i+1} loss: {loss:.4f}")

        avg_loss = sum(losses) / len(losses)
        print(f"\n✅ DCP 模型平均 loss: {avg_loss:.4f}")

        # 清理
        del model
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.destroy_process_group()

        return avg_loss, losses

    except Exception as e:
        print(f"❌ 加载 DCP checkpoint 失败: {e}")
        import traceback
        traceback.print_exc()

        if dist.is_initialized():
            dist.destroy_process_group()

        return None, []


def main():
    # 配置
    model_path = "../OLMo-1B"
    checkpoint_path = "../OLMo-1B/checkpoint_test/step-0"  # 修改为你的 checkpoint 路径

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "HF vs DCP Checkpoint Loss 对比" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n模型路径: {model_path}")
    print(f"Checkpoint 路径: {checkpoint_path}")

    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Tokenizer 加载完成")

    # 准备测试数据（使用多样化的文本）
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "In the beginning, there was nothing but darkness. " * 50,
        "Machine learning is a subset of artificial intelligence. " * 50,
        "The capital of France is Paris, which is known for the Eiffel Tower. " * 50,
        "Python is a high-level programming language widely used in data science. " * 50,
    ]

    print(f"\n测试数据: {len(test_texts)} 个文本")
    print(f"每个文本长度: ~{len(tokenizer(test_texts[0])['input_ids'])} tokens\n")

    # 测试 HF 模型
    hf_avg_loss, hf_losses = test_hf_model(model_path, test_texts, tokenizer)

    # 测试 DCP 模型
    dcp_avg_loss, dcp_losses = test_dcp_model(checkpoint_path, model_path, test_texts, tokenizer)

    # 对比结果
    print("\n" + "=" * 80)
    print("最终对比")
    print("=" * 80)

    print(f"\nHuggingFace 原始模型:")
    print(f"  平均 loss: {hf_avg_loss:.4f}")
    for i, loss in enumerate(hf_losses):
        print(f"    文本 {i+1}: {loss:.4f}")

    if dcp_avg_loss is not None:
        print(f"\nDCP Checkpoint 模型:")
        print(f"  平均 loss: {dcp_avg_loss:.4f}")
        for i, loss in enumerate(dcp_losses):
            print(f"    文本 {i+1}: {loss:.4f}")

        # 计算差异
        diff = abs(hf_avg_loss - dcp_avg_loss)
        print(f"\n差异:")
        print(f"  平均 loss 差异: {diff:.4f}")
        print(f"  相对差异: {diff / hf_avg_loss * 100:.2f}%")

        # 逐个文本对比
        print(f"\n逐个文本差异:")
        for i, (hf_loss, dcp_loss) in enumerate(zip(hf_losses, dcp_losses)):
            diff_i = abs(hf_loss - dcp_loss)
            print(f"    文本 {i+1}: HF={hf_loss:.4f}, DCP={dcp_loss:.4f}, 差异={diff_i:.4f}")

        # 判断
        print("\n" + "=" * 80)
        print("诊断结果")
        print("=" * 80)

        if diff < 0.01:
            print("\n✅ 完美！checkpoint 转换正确，loss 几乎一致")
            print("   checkpoint 包含了完整的预训练权重")
        elif diff < 0.1:
            print("\n⚠️  checkpoint 基本正确，但有小的差异")
            print("   可能原因：精度转换、部分参数缺失等")
        else:
            print("\n❌ checkpoint 转换有问题，loss 差异较大")
            print("   可能原因：")
            print("   1. 参数加载不完整")
            print("   2. 参数值不正确")
            print("   3. 模型配置不匹配")

        print(f"\n训练起始 loss: 4.2891")
        print(f"HF 原始模型 loss: {hf_avg_loss:.4f}")
        if abs(hf_avg_loss - 4.2891) < 0.1:
            print("   → 训练起始 loss 与 HF 模型一致，说明 checkpoint 正确加载了")
        else:
            print(f"   → 差异: {abs(hf_avg_loss - 4.2891):.4f}")
            print("   → 如果 HF loss 更低，说明训练时参数没有正确加载")
            print("   → 如果 HF loss 也是 4.x，可能是数据/tokenizer 问题")
    else:
        print("\n❌ DCP checkpoint 加载失败，无法对比")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
