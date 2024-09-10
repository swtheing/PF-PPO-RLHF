import jsonlines
import sys
import re
import re
def remove_template(text):
    """
    Extract the instruction part from the given text.
    """
    # Define the pattern to match the content between "### Instruction:\n" and "### Response:\n"
    pattern = re.compile(r'### Instruction:\n(.*?)### Response:', re.DOTALL)
    # Find the instruction content
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return ''

fp = jsonlines.open("Active_Learning.jsonl", "w")
fp_2 = jsonlines.open("RM_Query.jsonl", "w")
from collections import defaultdict
pos = defaultdict(list)
neg = defaultdict(list)
pos_mean = 0
all_pos = 0
for obj in jsonlines.open(sys.argv[1]):
    pos[obj["input"]].append((obj["output"], obj["reward"]))
    pos_mean += obj["reward"]
    all_pos += 1
print(pos_mean / all_pos)
neg_mean = 0
all_neg = 0
for obj in jsonlines.open(sys.argv[2]):
    neg[obj["input"]].append((obj["output"], obj["reward"]))
    neg_mean += obj["reward"]
    all_neg += 1
print(neg_mean / all_neg)
right = 0
count = 0
a_r = 0
for key, pos_r in pos.items():
    all_right = False
    print("pos:")
    print([r for pos, r in pos_r])
    print("neg:")
    print([r for neg, r in neg[key]])
    for output, r_1 in pos_r:
        flag = True
        for output, r_2 in neg[key]:
            if r_1 > r_2:
                right += 1
            else:
                flag = False
            count += 1
        all_right = all_right or flag
    if all_right:
        a_r += 1
    else:
        pos_sort = sorted(pos_r, key=lambda x:x[1])
        neg_sort = sorted(neg[key], key=lambda x:x[1])
        print(pos_sort[-1][1])
        print()
        nobj = {}
        nobj["instruction"] = remove_template(key)
        nobj["pos"] = pos_sort[-1][0]
        nobj["neg"] = neg_sort[-1][0]
        fp.write(nobj)
        #nobj = {}
        #nobj["instruction"] = remove_template(key)
        #fp.write(nobj)
        '''
        for output, n in neg[key]: 
            nobj = {}
            nobj["instruction"] = remove_template(key)
            nobj["pos"] = pos_sort[-1][0]
            nobj["neg"] = output
            fp.write(nobj)
        '''
print(right / count)
print(a_r / len(pos))

