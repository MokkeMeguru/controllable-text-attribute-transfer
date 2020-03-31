import sys
from pathlib import Path

file = Path("reference")
target = Path("re_reference")

result = []
with file.open("r", encoding='utf-8') as f:
    line = f.readline().rstrip()
    while line:
        result.append((line.split("\t")[0]))
        line = f.readline().rstrip()

with target.open("w", encoding='utf-8') as f:
    for line in result:
        f.write(line + "\n")
