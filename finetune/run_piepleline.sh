echo "========= start install requirements ========"
pip install -U transformers
pip install -U accelerate
pip install evaluate


echo "========= 准备数据 ========"
#python /code/zengyufei/datasets/use_for_pretrain/data_process.py

echo "========= 合成模型 ========"
mkdir /workspace/Qwen1.5-7B
cp -r /workspace/data/model/qwen/Qwen1.5-7B/* /workspace/Qwen1.5-7B/

mkdir /workspace/merge
cd /code/zengyufei/qiming_moe
pip install -e .

python my_moe_hidden.py --gate_mode hidden
mergekit-moe-qwen2 config.yaml /workspace/merge --i-understand-this-is-not-useful-without-training

echo "========= 计算合并后的模型权重大小 ========"
cd /workspace/merge
du -sh
cd /code/zengyufei/MoE_Train/finetune
echo "========= 开始训练 ========"

deepspeed --include localhost:0,1,2,3 --master_port=9902 /code/zengyufei/MoE_Train/finetune/run_moe.py \
    --deepspeed ds_zero2.config \
    --model_name_or_path /workspace/merge \
    --do_train True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1.0 \
    --block_size 1024 \
    --train_file /code/zengyufei/datasets/use_for_pretrain/all_results.csv \
    --output_dir /workspace/data/outputs \
    --overwrite_output_dir \
    --logging_dir /workspace/data/outputs \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 1 \
    --logging_strategy   steps  \
    --logging_steps   10  \
    --trust_remote_code True \
    --bf16