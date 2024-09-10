import jsonlines
import sys 

fp = jsonlines.open("LF_rm.jsonl", "w")
for obj in jsonlines.open(sys.argv[1]):
    nobj = {}
    nobj["instruction"] = obj["instruction"]
    nobj["input"] = ""
    nobj["output"] = [obj["pos"], obj["neg"]]
    fp.write(nobj)
