# MoE_Train
定制化构建qwen_moe架构，并实现训练和微调

# Step1 从原始模型copy权重给moe架构的模型
--mergeit-moe
    python mergit.py (注意，需要将run.sh中的参数添加进去)
    或者 sh run.sh
