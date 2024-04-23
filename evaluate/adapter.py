from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled
from models.utils import find_all_linear_modules

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)

def init_adapter(
    model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments", is_trainable: bool
) -> "PreTrainedModel":

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))
        adapter_to_resume = None

        if model_args.adapter_name_or_path is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
                assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                is_mergeable = False

            if is_deepspeed_zero3_enabled():
                assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
                is_mergeable = False

            if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = model_args.adapter_name_or_path[:-1]
                adapter_to_resume = model_args.adapter_name_or_path[-1]
            else:
                adapter_to_merge = model_args.adapter_name_or_path

            for adapter in adapter_to_merge:
                model: "LoraModel" = PeftModel.from_pretrained(
                    model, adapter, offload_folder=model_args.offload_folder
                )
                model = model.merge_and_unload()

            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

            if adapter_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(
                    model, adapter_to_resume, is_trainable=is_trainable, offload_folder=model_args.offload_folder
                )

        if is_trainable and adapter_to_resume is None:  # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model)
            else:
                target_modules = finetuning_args.lora_target

            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
            }

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                modules_to_save=finetuning_args.additional_target,
                use_dora=finetuning_args.use_dora,
                **peft_kwargs,
            )
            model = get_peft_model(model, lora_config)

        if model_args.adapter_name_or_path is not None:
            logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model
