python /Users/a58/Downloads/MoE_Train/finetune/run_clm.py \
    --model_name_or_path /Users/a58/Downloads/pretrain_model/Qwen/Qwen1.5-0.5B-Chat \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_local_experts 4 \
    --num_experts_per_tok 2 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1.0 \
    --block_size 32 \
    --train_file /Users/a58/Downloads/my_test/data/pre_train/all_results.csv \
    --output_dir /Users/a58/Downloads/my_test/my_train/outputs \
    --overwrite_output_dir \
    --logging_dir /Users/a58/Downloads/my_test/data/pre_train  \
    --logging_strategy   steps  \
    --logging_steps   10  \
    --trust_remote_code True \
    --template qwen


    --model_name_or_path /workspace/Qwen1.5-0.5B-Chat \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1.0 \
    --block_size 32 \
    --train_file /workspace/all_results.csv \
    --output_dir /Users/a58/Downloads/my_test/my_train/outputs \
    --overwrite_output_dir \
    --logging_dir /Users/a58/Downloads/my_test/my_train/outputs \
    --logging_strategy steps \
    --logging_steps 10