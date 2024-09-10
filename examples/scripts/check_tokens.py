import torch
import sys
import jsonlines
from transformers import AutoTokenizer

# Load the tokenizer for Llama2
tokenizer = AutoTokenizer.from_pretrained("./ckpt/7b_llama_ppo_2_epoch_RS_new/")

fp = jsonlines.open("prompt-filter-5.jsonl", "w")
for obj in jsonlines.open(sys.argv[1]):
    test = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{}\n### Response:\n"
    input = test + obj["instruction"]
    tokens = tokenizer.encode(input)
    if len(tokens) < 1024:
        fp.write(obj)
