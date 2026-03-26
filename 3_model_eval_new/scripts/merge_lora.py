#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Adapter 合并脚本

将 LoRA adapter 合并到基础模型中，生成完整的合并模型。

用法:
    python scripts/merge_lora.py \
        --base_model /path/to/base/model \
        --lora_adapter /path/to/lora/adapter \
        --output_dir /path/to/output
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_base(base_model_path: str, lora_adapter_path: str, output_dir: str):
    """
    将 LoRA adapter 合并到基础模型中。

    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA adapter 路径
        output_dir: 输出目录
    """
    print(f"[1/4] 加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/4] 加载 LoRA adapter: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("[3/4] 合并 LoRA 权重到基础模型...")
    model = model.merge_and_unload()

    print(f"[4/4] 保存合并后的模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    # 复制 tokenizer
    print("[+] 复制 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("[完成] 模型合并成功！")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 到基础模型")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基础模型路径（LoRA 训练时的基础模型）"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA adapter 目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="合并后模型的输出目录"
    )
    args = parser.parse_args()

    merge_lora_to_base(args.base_model, args.lora_adapter, args.output_dir)


if __name__ == "__main__":
    main()
