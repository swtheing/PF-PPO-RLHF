set -x

read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path ./ckpt/7b_llama_rm_deepseek_100_AC_CL \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 2 \
     --pretrain /root/paddlejob/workspace/code_llama_pack/pretrained_weights/deepseek-coder-6.7b-base/ \
     --bf16 \
     --max_epochs 1 \
     --prompt_key instruction \
     --chosen_key pos \
     --rejected_key neg \
     --max_len 4096 \
     --zero_stage 3 \
     --learning_rate 1e-5 \
     --dataset /root/paddlejob/workspace/code_llama_pack/rm_train_pairs.jsonl \
     --dataset_probs 1.0 \
     --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi

