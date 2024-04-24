import torch
import os.path
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from template import get_template_and_fix_tokenizer, add_prompt_form_template

from arguments import DataArguments

IGNORE_INDEX = -100

###用于制作属于自己的数据处理格式
class TrainDatasetForSft(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        #有多个文件的时候，对每个文件进行处理
        if args.train_file:
            if os.path.isdir(args.train_file):
                train_filesets = []
                for file in os.listdir(args.train_file):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_file, file),
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_filesets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_filesets)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.train_file, split='train')
        elif args.validation_file:
            self.dataset = datasets.load_dataset('json', data_files=args.validation_file, split='train')
        
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        
        #提前处理好数据
        self.template = get_template_and_fix_tokenizer(self.tokenizer, name=self.args.template)
        self.dataset = self.dataset.map(self.process_fn, batched=True, batch_size=128, num_proc=1, remove_columns=self.dataset.column_names)  
        self.dataset = self.dataset.sort("input_length")
        self.dataset = self.dataset.filter(lambda example: example["input_length"] < self.args.cutoff_len)

    def __len__(self):
        return self.total_len

    def process_fn(self, example):
        """思路：如果是llama格式，则直接拼接instruction和input（包括qwen）
                如果是gpt格式，以conversation对话形式存在
        """
        result = {"instruction":[], "input":[], "output":[], "input_length":[]}
        for idx in range(len(example['input'])):
            query = example['instruction'][idx] + example['input'][idx]
            label = example['output'][idx]
            #需要调用模板，对query进行处理
            query = add_prompt_form_template(template=self.template, query=query)
            result["instruction"].append(query)
            result["input_length"].append(len(query))
            result["input"].append("")
            result["output"].append(label)
            
        return result
        
    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['instruction']
        label = self.dataset[item]['output']
        
        return query, label


@dataclass
class CustomCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    label_max_len: int = 128
    args: DataArguments = None

    # def padding_score(self, teacher_score):
    #     group_size = None
    #     for scores in teacher_score:
    #         if scores is not None:
    #             group_size = len(scores)
    #             break
    #     if group_size is None:
    #         return None

    #     padding_scores = [100.0] + [0.0] * (group_size - 1)
    #     new_teacher_score = []
    #     for scores in teacher_score:
    #         if scores is None:
    #             new_teacher_score.append(padding_scores)
    #         else:
    #             new_teacher_score.append(scores)
    #     return new_teacher_score

    def __call__(self, features):
        # 需要对features进行拆分
        query = [f[0] for f in features]
        label = [f[1] for f in features]

        self.query_max_len = max(self.query_max_len, max([len(q) for q in query]))
        self.label_max_len = max(self.label_max_len, max([len(l) for l in label]))
        
        q_collator = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        
        l_collator = self.tokenizer(
            label,
            padding=True,
            truncation=True,
            max_length=self.label_max_len,
            return_tensors="pt",
        )
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for input_ids, label_ids in zip(q_collator["input_ids"], l_collator["input_ids"]):
            source_mask = [IGNORE_INDEX] * len(input_ids)
            input_ids = input_ids.tolist()
            label_ids = label_ids.tolist()
            if self.args.train_file:
                model_inputs["input_ids"].append(torch.tensor(input_ids + label_ids + [self.tokenizer.eos_token_id]))
                model_inputs["labels"].append(torch.tensor(source_mask + label_ids + [self.tokenizer.eos_token_id]))
            elif self.args.validation_file:
                model_inputs["input_ids"].append(torch.tensor(input_ids))
                model_inputs["labels"].append(torch.tensor(label_ids))
            model_inputs["attention_mask"].append([1]*len(input_ids))
            
            assert len(torch.tensor(input_ids + label_ids + [self.tokenizer.eos_token_id])) == len(source_mask + label_ids + [self.tokenizer.eos_token_id]), "长度不一致"
        
        model_inputs["input_ids"] = torch.stack(model_inputs["input_ids"])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
        model_inputs["labels"] = torch.stack(model_inputs["labels"])
        
        return model_inputs
