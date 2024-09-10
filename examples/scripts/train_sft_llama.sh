set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 2048 \
    --dataset /root/paddlejob/workspace/code_llama_pack/data/1k_data.jsonl \
    --dataset_probs 1.0 \
    --input_key instruction \
    --output_key output \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --max_samples 500000 \
    --pretrain /root/paddlejob/workspace/code_llama_pack/pretrained_weights/CodeLlama-7b-hf \
    --save_path ./ckpt/7b_llama \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
