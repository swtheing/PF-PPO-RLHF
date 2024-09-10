import jsonlines
import sys
import re

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

fp = jsonlines.open(sys.argv[2], "w")
for obj in jsonlines.open(sys.argv[1]):
    obj["input"] = remove_template(obj["input"])
    fp.write(obj)    
