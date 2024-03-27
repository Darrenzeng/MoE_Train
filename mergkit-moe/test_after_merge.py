from transformers import (AutoConfig, HfArgumentParser,AutoModelForCausalLM,AutoTokenizer)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

def conversation_test(tokenizer, moe_model):
    
    def get_prompt_qwen(sentence):
        ENDOFTEXT = "<|endoftext|>"
        IMSTART = "<|im_start|>"
        IMEND = "<|im_end|>"
        system = "system"
        user = "user"
        instruction = "You are a helpful assistant."
        prompt = IMSTART + system + "\n" + \
            instruction + IMEND + "\n" + \
            IMSTART + user + "\n" +\
            sentence + IMEND + "\n" + \
            IMSTART + "assistant" + "\n"
        return prompt
    text  = "我今天上课迟到了"
    text = get_prompt_qwen(text)
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to("cpu")
    
    pred = moe_model.generate(**inputs,max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
    
    # inputs = tokenizer("你好", return_tensors="pt")

    # outputs = model.generate(
    #     **inputs, max_length=32, do_sample=True, top_p=0.9, top_k=50
    # )

    # print(tokenizer.decode(outputs[0]))

if __name__=="__main__":
    # model_name_or_path = "/Users/a58/Downloads/pretrain_model/Qwen/Qwen1.5-0.5B-Chat"
    
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # conversation_test(tokenizer, moe_model=model)
    
    # del model
    
    print("测试合并后的模型")
    model_name_or_path = "/Users/a58/Downloads/my_test/my_train/outputs"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    print(model)
    conversation_test(tokenizer, moe_model=model)