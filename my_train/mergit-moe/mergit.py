

import sys
import os
import re
import shutil

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    print("debugger 失败")

# 获取当前脚本所在的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将父级目录添加到搜索路径
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from finetune.arguments import ModelArguments, DataTrainingArguments, \
    RetrieverTrainingArguments as TrainingArguments
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# from MoE.qwen2moe.models.modeling_qwen2moe import Qwen2ForCausalLM
# from MoE.qwen2moe.models.configuration_qwen2moe import Qwen2Config
from models.configuration_qwen2moe import Qwen2MoEConfig
from models.modeling_qwen2moe import Qwen2MoEForCausalLM
from transformers import (AutoConfig, HfArgumentParser,AutoModelForCausalLM,AutoTokenizer)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

def get_parametered_layers_list(model, num_hidden_layers):
    """
    Given a model and the number of hidden layers, this function returns two lists:
    one containing the parameterized layers of the model, grouped by hidden layer,
    and the other containing the remaining parameterized layers of the model.

    :param model: A PyTorch model object.
    :param num_hidden_layers: An integer representing the number of hidden layers in the model.
    :return: A tuple containing two lists - one for the parameterized layers of the model, grouped by hidden layer,
             and the other for the remaining parameterized layers of the model.
    """
    layers = []
    for name, module in model.named_parameters():
        layers.append((name, module))
    hidden_layers = [[] for _ in range(num_hidden_layers)]

    other_layers = []

    for layer in layers:
        if re.search("layers\.\d+", layer[0]):
            hidden_layers[int(layer[0].split("layers.")[1].split(".")[0])].append(layer)
        else:
            other_layers.append(layer)

    return hidden_layers, other_layers

def count_parameters(module):
    """
    统计给定nn.Module的参数量。

    参数:
    module (nn.Module): 要统计参数的模型或层。

    返回:
    int: 参数的总数。
    """
    total_params = 0
    for param in module.parameters():
        if param.requires_grad:
            total_params += param.numel()  # numel() 返回参数元素的总数
    return total_params

#可自定义从hf中读取config，但这里会强制转换为qwen2_moe_config
def convert_qwen2config_to_qwen2moeconfig(
    qwen2config:Qwen2Config,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
    output_router_logits: bool = False,
    router_aux_loss_coef: float = 0.001
    )->Qwen2MoEConfig:
    """convert qwen2config to qwen2moeconfig
    Args:
        qwen2config (Qwen2Config): 原始的qwen_config
        num_local_experts (int, optional): 专家数量. Defaults to 8.
        num_experts_per_tok (int, optional): 每个token过几个专家. Defaults to 2.
        output_router_logits (bool, optional): _description_. Defaults to False.
        router_aux_loss_coef (float, optional): _description_. Defaults to 0.001.

    Returns:
        Qwen2MoEConfig: _description_
    """
    qwen2_moe_config = Qwen2MoEConfig(**qwen2config.to_dict())
    qwen2_moe_config.architectures = ["Qwen2MoEForCausalLM"]
    del qwen2_moe_config._name_or_path
    
    qwen2_moe_config.num_experts_per_tok = num_experts_per_tok
    qwen2_moe_config.num_local_experts = num_local_experts
    qwen2_moe_config.output_router_logits = output_router_logits
    qwen2_moe_config.router_aux_loss_coef = router_aux_loss_coef
    
    return qwen2_moe_config

def get_parameters_layers_list(model, num_hidden_layers):
    layers = []
    for name, module in model.named_parameters():
        layers.append((name, module))
        
    hidden_layers = [[] for _ in range(num_hidden_layers)]
    
    other_layers = []
    
    for layer in layers:
        if re.search("layers\.\d+", layer[0]):
            hidden_layers[int(layer[0].split("layers.")[1].split(".")[0])].append(layer)
        else:
            other_layers.append(layer)

    return hidden_layers, other_layers    
    

def convert_model(model_raw, moe_model):
    """
    copy weights from model_raw to moe_model

    Args:
        model_raw (_type_): 原始的模型
        moe_model (_type_): moe模型
    """
    assert model.config.num_hidden_layers == moe_model.config.num_hidden_layers, "num_hidden_layers is not matched."
    num_hidden_layers = model.config.num_hidden_layers
    
    #hidden_layers是多少层decoder，other_layers是像embedding、norm层的非decoder层
    model_hidden_layers, model_other_layers = get_parametered_layers_list(model_raw, model_raw.config.num_hidden_layers)
    moe_hidden_layers, moe_other_layers = get_parametered_layers_list(moe_model, moe_model.config.num_hidden_layers)

    assert set([other_layer[0] for other_layer in model_other_layers]) == set([other_layer[0] for other_layer in moe_other_layers]), "Other layer is not matched."

    model_other_layers = {layer[0]: layer[1] for layer in model_other_layers}
    moe_other_layers = {layer[0]: layer[1] for layer in moe_other_layers}
    
    # copy non-decoder weights 
    for other_layer in model_other_layers:
        moe_other_layers[other_layer].data = model_other_layers[other_layer].data
        
    for i in range(num_hidden_layers):
        model_layer = model_hidden_layers[i]
        moe_layer = moe_hidden_layers[i]

        attn_weights = {x[0]: x[1] for x in model_layer if "mlp" not in x[0]}
        mlp_weights = {x[0].split("mlp.")[1]: x[1] for x in model_layer if "mlp" in x[0]}

        for weight_name, weight in moe_layer:
            #copy attn weights
            if weight_name in attn_weights:
                weight.data.copy_(attn_weights[weight_name].data)

            else:
                # skip gate in moe
                if "block_sparse_moe.gate" in weight_name:
                    continue
                # copy mlp weights
                weight.data.copy_(mlp_weights[re.split("experts\.\d+\.", weight_name)[1]].data)
     

def save_model(training_args, moe_model):
    """
    Save the Qwen2MoE model and related configurations and files.

    Args:
        training_args (TrainingArguments): The training arguments.
        moe_model (Qwen2MoEModel): The model to be saved.

    """
    
    moe_model.config.auto_map = {
        "AutoConfig": "configuration_qwen2moe.Qwen2MoEConfig",
        "AutoModelForCausalLM": "modeling_qwen2moe.Qwen2MoEForCausalLM"
    }

    moe_model.save_pretrained(training_args.output_dir)
    # qwen2_moe_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # copy src.modeling_qwen2_moe.py / src.configuration_qwen2_moe.py to output_dir
    shutil.copy("/Users/a58/Downloads/my_test/my_train/models/configuration_qwen2moe.py", training_args.output_dir)
    shutil.copy("/Users/a58/Downloads/my_test/my_train/models/modeling_qwen2moe.py", training_args.output_dir)

    print("Done! Saved in ", training_args.output_dir)

if __name__=="__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model_config = model.config
    
    # 1. 初始化自定义的模型,并把config进行修改
    # model_config = AutoConfig.from_pretrained(model_args.my_model_name_or_path)
    model_moe_config = convert_qwen2config_to_qwen2moeconfig(model_config)
    moe_model = Qwen2MoEForCausalLM(model_moe_config)
    
    print("MoE model config: ", model_moe_config.to_json_string())

    print("Origin model parameters: ", count_parameters(model))
    print("Moe model parameters: ", count_parameters(moe_model))
    print("Moe model Parameters ratio: ", count_parameters(moe_model) / count_parameters(model))
    
    print("copying weights...")
        
    convert_model(model, moe_model)
    del model
    print("copying has done!")
    save_model(training_args, moe_model)