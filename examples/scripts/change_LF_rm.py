import jsonlines
import sys

fp = jsonlines.open(sys.argv[2], "w")
for obj in jsonlines.open(sys.argv[1]):
    obj["output"] = [obj["output"], ""]
    obj["input"] = ""
    fp.write(obj)
