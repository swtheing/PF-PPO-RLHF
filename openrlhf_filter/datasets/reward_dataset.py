from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None) -> str:
    system_prompt = None

    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman:\n").replace("assistant:", "\nAssistant:\n")
                    + "\nAssistant:\n"
                )
            chosen = data["chosen"]
            reject = data["rejected"]
            input_template = None  # do not modified with input template again
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"] + "\n")
                return "".join(result)

            prompt = ""
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        else:
            raise ValueError("Unknown reward dataset")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)

    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    return prompt, chosen, reject, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human:\n{}\nAssistant:\n",
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo

        self.prompts = []
        self.chosens = []
        self.rejects = []
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_dpo = is_dpo

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key
            )

            # prompt_ids_len for prompt mask
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(margin)

            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]

        chosen = prompt + chosen + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
