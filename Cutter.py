import os
import yaml
import json
import argparse
import re
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import torch

def get_nested_attr(obj, attr_path):
    attr_names = attr_path.split('.')
    for attr_name in attr_names:
        if attr_name.isdigit():
            obj = obj[int(attr_name)]
        else:
            obj = getattr(obj, attr_name)
    return obj

def set_nested_attr(obj, attr_path, value):
    attr_names = attr_path.split('.')
    for attr_name in attr_names[:-1]:
        if attr_name.isdigit():
            obj = obj[int(attr_name)]
        else:
            obj = getattr(obj, attr_name)
    setattr(obj, attr_names[-1], value)

def adjust_model_width(model_path, output_path, width_config_path, max_file_size=2**33):
    config = AutoConfig.from_pretrained(model_path)
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)
    model = AutoModelForCausalLM.from_config(config)
    weight_map = index["weight_map"]

    with open(width_config_path, "r") as f:
        width_config = yaml.safe_load(f)

    progress_bar = tqdm(total=len(weight_map), desc="Adjusting weights", unit="weight")

    for weight_name in weight_map:
        with safe_open(os.path.join(model_path, weight_map[weight_name]), framework="pt") as f:
            tensor = f.get_tensor(weight_name)

        merged_name = None
        for config_name in width_config:
            if "{id}" in config_name:
                merged_name = re.sub(r"\.(\d+)\.", ".{id}.", weight_name)
                if merged_name == config_name:
                    break
                
        if merged_name:
            new_width = int(width_config[merged_name][1:-1].split("x")[-1])
            if tensor.shape[-1] > new_width:
                new_tensor = tensor[..., :new_width].clone()
            else:
                new_tensor = tensor.clone()
        else:
            new_tensor = tensor.clone()

        new_param = torch.nn.Parameter(new_tensor)

        # 根据模型的实际属性路径,转换 weight_name
        if "model.layers" in weight_name:
            weight_name = weight_name.replace("model.layers", "model.layers")
        elif "model.embed_tokens" in weight_name:
            weight_name = weight_name.replace("model.embed_tokens", "model.embed_tokens")
        elif "model.norm" in weight_name:
            weight_name = weight_name.replace("model.norm", "model.norm")
        elif "lm_head" in weight_name:
            weight_name = weight_name

        try:
            set_nested_attr(model, weight_name, new_param)
        except AttributeError as e:
            print(f"Warning: {e}. Skipping the modification of '{weight_name}'.")

        progress_bar.update(1)

    progress_bar.close()

    for tensor_name, shape_str in width_config.items():
        if tensor_name.endswith(".attention.query.weight"):
            config.hidden_size = int(shape_str[1:-1].split("x")[-1])
        elif tensor_name.endswith(".attention.key.weight"):
            config.hidden_size = int(shape_str[1:-1].split("x")[-1])
        elif tensor_name.endswith(".attention.value.weight"):
            config.hidden_size = int(shape_str[1:-1].split("x")[-1])
        elif tensor_name.endswith(".attention.query_key_value.weight"):
            config.hidden_size = int(shape_str[1:-1].split("x")[-1]) // 3
        elif tensor_name.endswith(".intermediate.dense.weight"):
            config.intermediate_size = int(shape_str[1:-1].split("x")[-1])
        elif tensor_name.endswith(".output.dense.weight"):
            config.hidden_size = int(shape_str[1:-1].split("x")[-1])

    config.save_pretrained(output_path)

    adjusted_state_dict = {}
    for name in weight_map:
        try:
            attr = get_nested_attr(model, name)
            adjusted_state_dict[name] = attr.cpu()
        except (AttributeError, IndexError, TypeError):
            print(f"Warning: Failed to get attribute '{name}' from the model. Skipping.")
        
    adjusted_index = {}

    print(f"Max file size: {max_file_size}")
    print(f"Adjusted state dict length: {len(adjusted_state_dict)}")

    for key in adjusted_state_dict:
        adjusted_state_dict[key] = adjusted_state_dict[key].half()

    total_size = sum(tensor.numel() * tensor.element_size() for tensor in adjusted_state_dict.values())
    total_files = (total_size - 1) // max_file_size + 1
    print(f"Total files: {total_files}")

    current_file_index = 0
    current_file_size = 0
    current_file_tensors = {}
    safetensors_file = f"model-{current_file_index + 1:05d}-of-{total_files:05d}.safetensors"

    for layer_name, tensor in adjusted_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_file_size + tensor_size > max_file_size:
            safetensors_file = f"model-{current_file_index + 1:05d}-of-{total_files:05d}.safetensors"
            safetensors_path = os.path.join(output_path, safetensors_file)
            save_file(current_file_tensors, safetensors_path)
            current_file_index += 1
            current_file_tensors = {}
            current_file_size = 0

        current_file_tensors[layer_name] = tensor
        current_file_size += tensor_size
        adjusted_index[layer_name] = safetensors_file

    if current_file_size > 0:
        safetensors_file = f"model-{current_file_index + 1:05d}-of-{total_files:05d}.safetensors"
        safetensors_path = os.path.join(output_path, safetensors_file)
        save_file(current_file_tensors, safetensors_path)

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index["metadata"], "weight_map": adjusted_index}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust the width of model A based on specified configurations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model A.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the adjusted model.")
    parser.add_argument("--width_config_path", type=str, required=True, help="Path to the yml file specifying the desired shape for each tensor type.")
    parser.add_argument("--max_file_size", type=int, default=2**33, help="Maximum size (in bytes) for each safetensors file.")

    args = parser.parse_args()
    adjust_model_width(args.model_path, args.output_path, args.width_config_path, args.max_file_size)