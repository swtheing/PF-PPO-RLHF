import jsonlines
import sys

for obj in jsonlines.open(sys.argv[1]):
    instruction = obj["instruction"]
    nobj = {}
    nobj["instruction"] = instruction
    nobj["output"]
