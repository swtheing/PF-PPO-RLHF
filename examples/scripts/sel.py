import jsonlines
import sys
last = None
fp = jsonlines.open(sys.argv[2], "w")
for obj in jsonlines.open(sys.argv[1]):
    instruction = obj["instruction"]
    if instruction == last:
        last = instruction
        continue
    last = instruction
    fp.write(obj)
    
