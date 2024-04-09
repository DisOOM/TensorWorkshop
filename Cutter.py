import os
import yaml
import json
import argparse
import re
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
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
    try:
        for attr_name in attr_names[:-1]:
            if attr_name.isdigit():
                obj = obj[int(attr_name)]
            else:
                obj = getattr(obj, attr_name)
        setattr(obj, attr_names[-1], value)
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Warning: Failed to set attribute '{attr_path}': {str(e)}")

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
            shape_str = width_config[merged_name]
            shape_parts = shape_str.strip("[]").split("x")
            new_width = int(shape_parts[-1])

            if "self_attn.q_proj" in weight_name or "self_attn.k_proj" in weight_name or "self_attn.v_proj" in weight_name or "self_attn.o_proj.weight" in weight_name:
                if len(tensor.shape) >= 2:
                    # 对于 self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj.weight, 同时处理最后两个维度
                    if tensor.shape[-1] > new_width and tensor.shape[-2] > new_width:
                        new_tensor = tensor[..., :new_width, :new_width].clone()
                    elif tensor.shape[-1] > new_width:
                        new_tensor = tensor[..., :new_width].clone()
                    elif tensor.shape[-2] > new_width:
                        new_tensor = tensor[..., :new_width, :].clone()
                    else:
                        new_tensor = tensor.clone()
                else:
                    new_tensor = tensor.clone()

            elif "mlp.down_proj.weight" in weight_name or "mlp.up_proj.weight" in weight_name:
                if len(shape_parts) == 2:
                    new_shape = [int(shape_parts[0]), int(shape_parts[1])]
                    if tensor.shape != new_shape:
                        new_tensor = tensor[:new_shape[0], :new_shape[1]].clone()
                    else:
                        new_tensor = tensor.clone()
                else:
                    new_tensor = tensor.clone()

            elif "mlp.gate_proj.weight" in weight_name:
                if len(shape_parts) == 2:
                    new_shape = [int(shape_parts[0]), int(shape_parts[1])]
                    if tensor.shape != new_shape:
                        new_tensor = tensor[:new_shape[0], :new_shape[1]].clone()
                    else:
                        new_tensor = tensor.clone()
                else:
                    new_tensor = tensor.clone()

            else:
                if len(shape_parts) == 2:
                    # 处理形如 [151936x4096] 的张量
                    new_shape = [int(shape_parts[0]), new_width]
                else:
                    # 处理形如 [4096] 的张量
                    new_shape = [new_width]

                if tensor.shape != new_shape:
                    new_tensor = tensor[..., :new_width].clone()
                else:
                    new_tensor = tensor.clone()

            # 处理权重张量
            try:
                new_param = torch.nn.Parameter(new_tensor)
                set_nested_attr(model, weight_name, new_param)
            except (AttributeError, IndexError, TypeError):
                try:
                    setattr(model, weight_name, new_param)
                except (AttributeError, IndexError, TypeError):
                    print(f"Warning: Failed to process the weight tensor for {weight_name}.")

            # 为 self_attn.q_proj, self_attn.k_proj, self_attn.v_proj 处理偏置张量
            if weight_name.endswith(".weight") and any(
                x in weight_name for x in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
            ):
                bias_name = weight_name[:-6] + "bias"  # 将 "weight" 替换为 "bias"
                try:
                    bias_tensor = get_nested_attr(model, bias_name)
                    new_bias_tensor = bias_tensor[:new_width].clone()
                    new_bias_param = torch.nn.Parameter(new_bias_tensor)
                    try:
                        set_nested_attr(model, bias_name, new_bias_param)
                    except (AttributeError, IndexError, TypeError):
                        try:
                            setattr(model, bias_name, new_bias_param)
                        except (AttributeError, IndexError, TypeError):
                            print(f"Warning: Failed to process the corresponding bias tensor for {weight_name}.")
                except (AttributeError, IndexError, TypeError):
                    print(f"Warning: Failed to get the corresponding bias tensor for {weight_name}.")
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
        shape_str = shape_str.strip("[]")  # 去除形状字符串两端的方括号
        if tensor_name.endswith(".self_attn.q_proj.weight") or tensor_name.endswith(".self_attn.k_proj.weight") or tensor_name.endswith(".self_attn.v_proj.weight"):
            config.hidden_size = int(shape_str.split("x")[-1])
        elif tensor_name.endswith(".mlp.down_proj.weight"):
            config.intermediate_size = int(shape_str.split("x")[-1])
        elif tensor_name.endswith(".mlp.up_proj.weight"):
            config.hidden_size = int(shape_str.split("x")[0])
            
    # 调整注意力头的数量
    config.num_attention_heads = config.hidden_size // 128  # 假设每个注意力头的维度为128,这个和具体架构有关，到时候得依据架构来读
    config.num_key_value_heads = config.num_attention_heads

    # 更新配置文件
    config_dict = config.to_dict()
    config_dict["hidden_size"] = config.hidden_size
    config_dict["intermediate_size"] = config.intermediate_size

    # 保存更新后的配置文件
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

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

        # 将最后一个文件中的张量添加到索引文件中
        for tensor_name in current_file_tensors:
            adjusted_index[tensor_name] = safetensors_file

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