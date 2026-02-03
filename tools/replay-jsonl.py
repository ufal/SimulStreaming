#!/usr/bin/env python3

# Replays jsonl inputs with original timing.

# Assume a jsonl file where each line is a json object with an "emission_time" field. 
# I.e. an output of simulstreaming_whisper.py that you want to simulate with simulstreaming_translate.py on an input, 
# to simulate "computational aware simulation."

# For computationally unaware simulation, just use "cat input.jsonl | python3 simulstreaming_translate.py"

import sys
import time
import json
offset = 0
if len(sys.argv) == 2:
    offset = float(sys.argv[1])
start_time = time.time() + offset
for line in sys.stdin:
    row = json.loads(line)
    et = row["emission_time"]
    # Simple active waiting. Non-active waiting could be inacurrate.
    while True:
        tc = time.time()-start_time
        if tc >= et:
            break
    print(line, end="")
    sys.stdout.flush()
