set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain ./ckpt/7b_llama_ppo_1_epoch/ \
    --reward_pretrain ./ckpt/7b_llama_rm_deepseek_100_new_2/ \
    --save_path ./ckpt/7b_llama_ppo_1_epoch_normal/ \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 512 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 2048 \
    --num_episodes 3 \
    --max_epochs 3 \
    --prompt_max_len 1024 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data HumanEval-instruction-llama-10.jsonl \
    --input_key instruction \
    --prompt_data_probs 1.0 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --pretrain /root/paddlejob/workspace/code_llama_pack/experiments/alpaca-magic-130k/ \

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
