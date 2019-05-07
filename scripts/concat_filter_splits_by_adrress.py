import argparse
import sys
import re
from collections import defaultdict

argparser = argparse.ArgumentParser(\
    description="The scripts reads a sentence file list in a standart input and outputs only those \
                sentences not present in a black list of sentence addresses passed as a command \
                line argument.")
argparser.add_argument("in_blacklist_file", type=str, help="A file containing addresses to sentences to be excluded.")
args = argparser.parse_args()

adrresses = defaultdict()
with open(args.in_blacklist_file, "r", encoding="utf-8") as bl_f:
    for line in bl_f:
        line = line.rstrip()
        m = re.match(r'^(.*)/(\d+)-(\d+)$', line)
        adrresses[m.group(1)][m.group(2)][m.group(3)] = 1
        print(m.group(1))
        print(m.group(2))
        print(m.group(3))

#for line in sys.stdin:
#    re.sub("")
