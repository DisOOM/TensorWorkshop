# TensorInformation.py
import os
import yaml
import json
import argparse
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM
from safetensors import safe_open

def get_tensor_info(model_path, output_path):
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 加载模型的safetensors索引文件
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

    # 获取模型的权重映射
    weight_map = index["weight_map"]

    tensor_info = defaultdict(list)

    # 遍历模型的每个权重
    for weight_name in weight_map:
        # 打开模型的safetensors文件
        with safe_open(os.path.join(model_path, weight_map[weight_name]), framework="pt") as f:
            # 加载权重张量
            tensor = f.get_tensor(weight_name)

        # 将张量形状转换为字符串格式
        shape_str = "x".join(str(dim) for dim in tensor.shape)

        # 将张量名称和形状添加到tensor_info字典中
        tensor_info[weight_name].append(f"[{shape_str}]")

    # 使用正则表达式合并具有相似模式的张量名称
    merged_tensor_info = {}
    for tensor_name, shapes in tensor_info.items():
        # 如果张量名称包含数字,则将数字替换为{id}占位符
        merged_name = re.sub(r"\.\d+\.", ".{id}.", tensor_name)
        
        # 如果合并后的名称已经存在,则检查形状是否一致
        if merged_name in merged_tensor_info:
            assert merged_tensor_info[merged_name] == shapes[0], f"Inconsistent shapes for {merged_name}: {merged_tensor_info[merged_name]} vs {shapes[0]}"
        else:
            merged_tensor_info[merged_name] = shapes[0]

    # 将merged_tensor_info字典写入yml文件
    with open(output_path, "w") as f:
        yaml.dump(merged_tensor_info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get tensor information from a model and save it to a yml file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the yml file.")

    args = parser.parse_args()
    get_tensor_info(args.model_path, args.output_path)