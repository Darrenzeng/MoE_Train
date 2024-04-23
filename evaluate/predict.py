import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
import logging
import pandas as pd
from tqdm import tqdm
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
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
# from .modeling import BiEncoderModel
from finetune.trainer import CustomTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

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
    
    train_dataset = TrainDatasetForSft(args=data_args, tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=CustomCollator(
            tokenizer
        ),
        tokenizer=tokenizer
    )
    trainer.evaluate()

def my_predict():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config = config, trust_remote_code=True)
    
    eval_dataset = TrainDatasetForSft(args=data_args, tokenizer=tokenizer)
    collator = CustomCollator(tokenizer, args=data_args)
    dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=collator)
    df_data = pd.DataFrame(columns=["text", "label", "pred"])
    flag = True
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]#.to('cuda')
        label = batch["labels"]
        input_length = len(input_ids[0])
        if flag:
            print("原始的输入为:{}".format(tokenizer.decode(input_ids[0], skip_special_tokens=True)))
            print("!"*30)
            pred = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            pred_zh = tokenizer.decode(pred.cpu()[0][input_length:], skip_special_tokens=True)
            label_zh = tokenizer.decode(label[0], skip_special_tokens=True)
            print("预测结果为:{}\n标签结果为:{}".format(pred_zh, label_zh))
            flag = False
        else:
            pred = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            pred_zh = tokenizer.decode(pred.cpu()[0][input_length:], skip_special_tokens=True)
            label_zh = tokenizer.decode(label[0], skip_special_tokens=True)
        
        df_data = df_data.append({"text": tokenizer.decode(input_ids[0], skip_special_tokens=True), "label": label_zh, "pred": pred_zh}, ignore_index=True)
        
    df_data.to_csv(training_args.output_dir+"/result.csv", index=False)
    
if __name__=="__main__":
    my_predict()