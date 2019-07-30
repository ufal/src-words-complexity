import argparse
import gzip
import re
import sys
from collections import defaultdict

argparser = argparse.ArgumentParser(\
    description="During extraction of lemmatized bitexts, some too long sentences were skipped. \
                 Different pairs of (src, mt, pe) thus contain different number of sentences. \
                 This script searches over lemmatized bitexts for GIZA with indexes to original documents \
                 and stores the sentence positions for which at least some of the align pairs are missing. \
                 It stores it both as indexes to the document splits as well as the positions in the original \
                 lemmatized bitexts.")
argparser.add_argument("out_position_file", type=str, help="A file to output positions to delete in the input files (tab-delimited).")
argparser.add_argument("out_address_file", type=str, help="A file to output addresses to splitted documents.")
argparser.add_argument("in_forgiza_file", nargs="+", type=str, help="Input gzipped bitexts.")
args = argparser.parse_args()

addresses_for_input = {}
addresses_count = defaultdict(int)

for file in args.in_forgiza_file:
    print("Processing " + file, file=sys.stderr)
    addresses = []
    with gzip.open(file, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("Lines finished " + str(i), file=sys.stderr)
            addr = re.sub(r'^(.*)/part\.[^.]*\.(\d+)\.txt-s(\d+)$', r'\1/\2-\3', line.split("\t")[0].split(",")[-1])
            addresses.append(addr)
            addresses_count[addr] += 1
    addresses_for_input[file] = addresses

with open(args.out_position_file, "w", encoding="utf-8") as pos_f:
    for file in args.in_forgiza_file:
        print("Printing positions of " + file + " to " + args.out_position_file, file=sys.stderr)
        pos_to_del = [str(i+1) for i, addr in enumerate(addresses_for_input[file]) if (addresses_count[addr] < len(args.in_forgiza_file))]
        print("\t".join([file, " ".join(pos_to_del)]), file=pos_f)

with open(args.out_address_file, "w", encoding="utf-8") as addr_f:
    print("Printing addresses to " + args.out_address_file, file=sys.stderr)
    del_addr_list = [k for k in addresses_count.keys() if addresses_count[k] < len(args.in_forgiza_file)]
    del_addr_list.sort()
    print("\n".join(del_addr_list), file=addr_f)



