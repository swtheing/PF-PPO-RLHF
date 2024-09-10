# Policy Filtration in RLHF to Fine-Tune LLM for Code Generation

Paper link:

Author's code: [PF-PPO](https://github.com/swtheing/OpenRLHF_Tool)

The implementation is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main)

Credit: Wei Shen ([@swtheing](https://github.com/swtheing)), Chuheng Zhang ([zhangchuheng123](https://github.com/zhangchuheng123))


## Quick Start

### PF-PPO
You need to set the following parameters in the `combine_train_ana.sh` first, like:

```bash
save_path=./ckpt/7b_llama_ppo_eb4_multi/
rollout_batch_size=2048
output_file=test_he.jsonl
test_file=HumanEval-10-instruction-llama.jsonl
```
Then, run the script:
```bash
sh combine_train_ana.sh
```
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

