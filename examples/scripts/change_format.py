import jsonlines
import sys 
from collections import defaultdict
fp = jsonlines.open(sys.argv[3], "w")
dic = defaultdict(list)
for i, obj in enumerate(jsonlines.open(sys.argv[1])):
    dic[obj["input"]].append(obj)

last = None
for i, obj in enumerate(jsonlines.open(sys.argv[2])):
    for key, val in dic.items():
        if obj["instruction"] in key:
            fp.write(val[0])
            del val[0]
fp.close()
