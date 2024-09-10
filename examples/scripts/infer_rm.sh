set -x 

read -r -d '' training_commands <<EOF
../batch_inference.py \
     --pretrain ./ckpt/7b_llama_rm_deepseek_100_new_2 \
     --output_path ./output/$1 \
     --eval_task rm \
     --bf16 \
     --input_key instruction \
     --output_key output \
     --max_len 4096 \
     --zero_stage 3 \
     --dataset $2 \
     --dataset_probs 1.0
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi


