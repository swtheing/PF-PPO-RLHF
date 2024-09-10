export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x 
read -r -d '' training_commands <<EOF
../batch_inference.py \
     --pretrain $1 \
     --output_path ./output/$2 \
     --eval_task generate \
     --bf16 \
     --input_key instruction \
     --max_len 2048 \
     --zero_stage 2 \
     --dataset $3 \
     --dataset_probs 1.0
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
