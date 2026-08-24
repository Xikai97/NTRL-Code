python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir '/path/to/checkpoints/global_step_xxx/actor' \
    --target_dir '/path/to/checkpoints/global_step_xxx/hf_model' 