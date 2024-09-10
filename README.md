# Policy Filtration in RLHF to Fine-Tune LLM for Code Generation

Paper link:

Author's code: [PF-PPO](https://github.com/swtheing/OpenRLHF_Tool)

The implementation is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main)

Credit: Wei Shen ([@swtheing](https://github.com/swtheing)), Chuheng Zhang ([zhangchuheng123](https://github.com/zhangchuheng123))


## Quick Start

### PPO without Ray
deepspeed --module openrlhf.cli.train_ppo \
  --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
  --save_path ./checkpoint/llama-3-8b-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}

## Performance

| Family                      | Method                              | HumanEval | MBPP  | LeetCode |
|-----------------------------|-------------------------------------|-----------|-------|----------|
| **Supervised Fine-Tuning**   | SFT                                 | 74.2      | 70.8  | 15.2     |
|                             | RAFT (Dong et al., 2023)            | 76.9      | 71.3  | 17.8     |
|                             | BOND (Sessa et al., 2024)           | 80.8      | 75.2  | 30.0     |
| **Direct Policy Optimization** | DPO (Rafailov et al., 2024)       | 78.4      | 73.7  | 23.0     |
|                             | IPO (Azar et al., 2024)             | 78.2      | 72.9  | 23.2     |
|                             | KTO (Ezhayarajh et al., 2024)       | 77.9      | 72.5  | 22.4     |
|                             | Iterative-DPO (Pang et al., 2024)   | 78.1      | 74.8  | 23.8     |
| **Reinforcement Learning**   | PPO-S (Hu et al., 2024)             | 78.1      | 73.8  | 25.2     |
|                             | PPO-M (cf. Shao et al., 2024)       | 80.2      | 75.0  | 29.8     |
|                             | PF-PPO (BoN)                        | 75.8      | 71.7  | 16.8     |
|                             | PF-PPO (BR)                         | **82.9**      | 75.9  | **33.0**     |
|                             | PF-PPO (BW)                         | 82.4      | **76.2**  | 30.4     |
| **SOTA (7B models)**         | Magicoder (Wei et al., 2023)        | 76.8      | 75.7  |          |

