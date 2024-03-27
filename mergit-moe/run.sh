python /Users/a58/Downloads/my_test/my_train/mergit-moe/mergit.py \
    --model_name_or_path /Users/a58/Downloads/pretrain_model/Qwen/Qwen1.5-0.5B-Chat \
    --my_model_name_or_path /Users/a58/Downloads/my_test/MoE/mymoe_qwen/docs\
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1.0 \
    --block_size 32 \
    --train_file /Users/a58/Downloads/my_test/data/pre_train/train.csv \
    --output_dir /Users/a58/Downloads/my_test/my_train/outputs \
    --overwrite_output_dir \
    --logging_dir /Users/a58/Downloads/my_test/my_train/outputs \
    --logging_strategy steps \
    --logging_steps 10