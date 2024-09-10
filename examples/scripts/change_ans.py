import jsonlines
import sys

fp = jsonlines.open(sys.argv[2], "w")
for obj in jsonlines.open(sys.argv[1]):
    for item in obj["tgt"]:
        nobj = {}
        nobj["instruction"] = obj["src"][0]
        nobj["output"] = item
        fp.write(nobj)
