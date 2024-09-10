import re
import sys
import jsonlines
from collections import defaultdict

def remove_template(text):
    """ 
    Extract the instruction part from the given text.
    """ 
    # Define the pattern to match the content between "### Instruction:\n" and "### Resp
    pattern = re.compile(r'### Instruction:\n(.*?)### Response:', re.DOTALL)
    # Find the instruction content
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return ''

candidate = defaultdict(list)
for obj in jsonlines.open(sys.argv[1]):
    input = remove_template(obj["input"])
    reward = obj["reward"]
    candidate[input].append(reward)

fp_2 = jsonlines.open(sys.argv[2], "w")
for key, val in candidate.items():
    val = sorted(val)
    if val[-1] > 0.5:
        nobj = {}
        nobj["instruction"] = key
        fp_2.write(nobj)
