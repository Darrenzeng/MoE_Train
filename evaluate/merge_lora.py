import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
#        low_cpu_mem_usage=True,
       trust_remote_code=True
    )
    for name, param in base.named_parameters():
        print(name, param.dtype)
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    for name, param in base.named_parameters():
        print(name, param.dtype)
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)



if __name__ == "__main__":
    model_name_or_path = "/workspace/Yi-34B-Chat"
    lora_path = "/workspace/checkpoints_yi"
    output_path = "/workspace/checkpoints_merge_lora_Yi-34B-Chat"
    apply_lora(model_name_or_path, output_path, lora_path)
