import torch
from enum import Enum, unique

logger = get_logger(__name__)

@unique
class QuantizationMethod(str, Enum):
    r"""
    Borrowed from `transformers.utils.quantization_config.QuantizationMethod`.
    """

    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"

def find_all_linear_modules(model: "PreTrainedModel") -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    quantization_method = getattr(model, "quantization_method", None)
    if quantization_method is None:
        linear_cls = torch.nn.Linear
    elif quantization_method == QuantizationMethod.BITS_AND_BYTES:
        import bitsandbytes as bnb

        linear_cls = bnb.nn.Linear4bit if getattr(model, "is_loaded_in_4bit", False) else bnb.nn.Linear8bitLt
    else:
        raise ValueError("Finding linear modules for {} models is not supported.".format(quantization_method))

    output_layer_names = ["lm_head"]
    if model.config.model_type == "chatglm":
        output_layer_names.append("output_layer")
    elif model.config.model_type == "internlm2":
        output_layer_names.append("output")

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(output_layer in name for output_layer in output_layer_names):
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)