#!/bin/python3

import sys
import os
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: %s trans.txt output_dir" % sys.argv[0])
    sys.exit(1)

with open(sys.argv[1]) as f:
    lines = list(map(lambda s: s.strip(), f.readlines()))

for line in lines:
    audio_name, transcription_text = line.split(' ', 1)
    transcription_path = Path(sys.argv[2]) / f"{audio_name}.txt"
    transcription_path.parent.mkdir(parents = True, exist_ok = True)
    with open(transcription_path, 'w') as f:
        print(transcription_text, file = f)
