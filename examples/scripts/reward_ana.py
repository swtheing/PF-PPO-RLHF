import sys

train_reward = []
eval_reward = []
for line in open(sys.argv[1]):
    title, reward = line.strip().split(":")
    if title == "train_reward_sw":
        train_reward.append(float(reward))   
    elif title == "eval_reward_sw":    
        eval_reward.append(float(reward))

x = []

y = []
train_len = int(sys.argv[2])
count = 0
for i in range(0, len(train_reward), train_len):
    if len(train_reward[i:i+train_len]) == train_len:
        y.append(sum(train_reward[i:i+train_len]) / len(train_reward[i:i+train_len]))
        x.append(count)
        count += 1

z = []
test_len = int(len(eval_reward) / len(y))
print(test_len)
for i in range(0, len(eval_reward), test_len):
    z.append(sum(eval_reward[i:i+test_len]) / len(eval_reward[i:i+test_len]))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='train_reward', color='#6a4c93', linestyle='-', linewidth=2)  # Soft purple
plt.plot(x, z, label='eval_reward', color='#e76f51', linestyle='--', linewidth=2)  # Soft red-orange

# Enhancing the plot
plt.title('Reward Over Epochs', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(sys.argv[3])
