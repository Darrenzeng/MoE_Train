# MoE_Train
定制化构建qwen_moe架构，并实现训练和微调

## 普通模型转换为MoE架构
从原始模型copy权重给moe架构的模型
```bash
python mergekit-moe/mergit.py (需要将run.sh中的参数添加进去)
```
或者
```bash
sh run.sh
```
运行完成后，可以对moe模型进行测试
```bash
python mergekit-moe/test_after_merge.py
```

## 预训练
```bash
$cd pretrain
sh run.sh
```

## 微调
```bash
$cd finetune
sh run.sh
```
注意：如果需要关闭wandb，请在终端运行命令(开启也是一样的命令)
```bash 
wandb offline
```
