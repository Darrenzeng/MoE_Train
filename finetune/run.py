import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9503))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    print("启动debug失败")
import logging
import os
os.environ["WANDB_PROJECT"]="Moe_finetuning"
# os.environ["WANDB_DISABLED"] = "true"
import torch
import wandb
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    HfArgumentParser,
    set_seed,
)

import sys
sys.path.append("/Users/a58/Downloads/MoE_Train")
from hparams.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from hparams.data import TrainDatasetForSft, CustomCollator
from trainer import CustomTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config = config, trust_remote_code=True)#, torch_dtype=torch.bfloat16

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Moe_finetuning",
    #     entity="darren_zeng",
    #     config={
    #     "learning_rate": config.learning_rate,
    #     "architecture": model_args.model_name_or_path,
    #     "batch_size": training_args.per_device_train_batch_size,
    #     "test_batch_size": training_args.per_device_eval_batch_size,
    #     "epochs": training_args.num_train_epochs
    #     },
    #     name="test1"
    # )

    if training_args.use_lora:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if training_args.q_lora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            if not ddp and torch.cuda.device_count() > 1:
                # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
                model.is_parallelizable = True
                model.model_parallel = True

        model = get_peft_model(model, lora_config)
        if training_args.deepspeed is not None and training_args.local_rank == 0:
            model.print_trainable_parameters()

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForSft(args=data_args, tokenizer=tokenizer)

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = CustomCollator(tokenizer, args=data_args, pad_to_multiple_of=8)
    else:
        data_collator = CustomCollator(
            tokenizer,
            args=data_args
        )
        
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    wandb.finish()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
