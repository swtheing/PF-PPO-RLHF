import argparse
from datetime import datetime
from typing import List

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)


def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"
    assert (
        critic_world_size & (critic_world_size - 1)
    ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
    assert (
        actor_world_size % critic_world_size == 0
    ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [
            {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
            for _ in range(args.actor_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
    # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
    # If we allocate 0.5 gpu for both actor and ref model, then gpu allocation is
    #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
    #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
    #
    # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
    #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
    #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.25 if pg else 1,
    )

    # if colocated, create placement group for critic and reward model explicitly.
    pg = None
    if args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [
            {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node}
            for _ in range(args.critic_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    critic_model = PPORayActorGroup(
        args.critic_num_nodes,
        args.critic_num_gpus_per_node,
        CriticModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    # multiple reward models
    reward_pretrains = args.reward_pretrain.split(",")
    reward_models = []
    for _ in reward_pretrains:
        reward_models.append(
            PPORayActorGroup(
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )
        )

    # init reference/reward/actor model
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None:
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines, args.vllm_tensor_parallel_size, args.pretrain, args.seed
        )

    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    refs.extend(critic_model.async_init_model_from_pretrained(strategy, reward_pretrains[0], max_steps))
    ray.get(refs)

    # train actor and critic mdoel
    refs = actor_model.async_fit_actor_model(
        critic_model, ref_model, reward_models, reward_fn=reward_fn, vllm_engines=vllm_engines
    )
    ray.get(refs)

    # save model
    ray.get(actor_model.async_save_actor_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument("--vllm_num_engines", type=int, default=None, help="number of vLLM Engines")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )

    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.02)
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)

    # evaluation
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()
    train(args)
