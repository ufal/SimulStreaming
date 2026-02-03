#!/usr/bin/env python3
import sys
import time
offset = 0
if len(sys.argv) == 2:
    offset = float(sys.argv[1])
start_time = time.time()
for line in sys.stdin:
    i = line.index(" ")
    lt = float(line[:i])
    line = line[i+1:]
    # Simple active waiting. Non-active waiting could be inacurrate.
    while True:
        t = time.time()
        tc = (-start_time + t)*1000 + offset
        if tc >= lt:
            break
    print(line, end="")
    sys.stdout.flush()
