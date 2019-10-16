import sys
import argparse
import termcolor


def colored(text, color=None, on_color=None, attrs=None):
    fg = lambda text, color: "\33[38;5;" + str(color) + "m" + text + "\33[0m"
    bg = lambda text, color: "\33[48;5;" + str(color) + "m" + text + "\33[0m"
    if isinstance(color, int):
        text = fg(text, color)
        color = None
    if isinstance(on_color, int):
        text = bg(text, on_color)
        on_color = None
    return termcolor.colored(text, color, on_color, attrs)


def parse_sentence_from_strlist(lines):
    trg_sent = lines.pop(0).rstrip()
    src_sent = lines.pop(0).rstrip()
    src_token_items = []
    for line in lines:
        entropy, *variants = line.rstrip().split("\t")
        tokens = variants[::2]
        probs = variants[1::2]
        variants_list = list(zip(tokens, [float(x) for x in probs]))
        src_token_items.append({"entropy": float(entropy), "variants": variants_list})
    return {"trg_sent": trg_sent, "src_sent": src_sent, "src_tokens": src_token_items}


DECORATION_STYLES = {
    "src_sent": {"color": "cyan"},
    "trg_sent": {"color": "yellow"},
    "top_src_token": {"color": "red"},
    "max_entropy": {"color": "yellow", "attrs": ["reverse"]},
}
COLOR_RANGE = list(range(21, 201, 36)) + list(range(201, 195, -1))
for i, code in enumerate(COLOR_RANGE):
    DECORATION_STYLES["entropy_" + str(i)] = {"on_color": code, "attrs": ["bold"]}



def decorate_lines(lines, annot):
    decor_lines = []
    for i in range(len(lines)):
        dline = ""
        s = 0
        for sa, ea, a in annot[i]:
            dline += lines[i][s:sa]
            dline += colored(lines[i][sa:ea], **DECORATION_STYLES[a])
            s = ea
        e = len(lines[i])
        dline += lines[i][s:e]
        decor_lines.append(dline)
    return decor_lines


def print_sentence_info(sent_info):
    trg_lines, trg_annot = sent_lines(sent_info, "trg_sent")
    src_lines, src_annot = sent_lines(sent_info, "src_sent")
    src_tokens_lines, src_tokens_annot = src_sent_lines(sent_info)
    if not args.no_color:
        decor_lines = decorate_lines(trg_lines+src_lines+src_tokens_lines, trg_annot+src_annot+src_tokens_annot)
    else:
        decor_lines = trg_lines+src_lines+src_tokens_lines
    for l in decor_lines:
        print(l)
    print()


def sent_lines(sent_info, label):
    sent = sent_info[label]
    annot = [(0, len(sent), label)]
    return [sent], [annot]


def src_sent_lines(sent_info, hbar_size=4, ):
    entr_levels = entropy_levels([si["entropy"] for si in sent_info["src_tokens"]], len(COLOR_RANGE))
    out_lines = []
    out_annot = []
    total_width = 0
    for item_idx, src_item in enumerate(sent_info["src_tokens"]):
        # add additional lines if the token has more variants than the tokens processed so far
        # +1: an additional line for entropy
        while len(out_lines) < len(src_item["variants"])+1:
            out_lines.append(" "*total_width)
            out_annot.append([])

        entropy_cell = " "*2
        entropy_cell += "{:.4f}".format(src_item["entropy"])
        variant_cells = []
        variant_annots = []
        for i, token_prob in enumerate(src_item["variants"]):
            token, prob = token_prob
            c = " "*2
            c += "{:>3.0f}% ".format(prob*100) + proportion_bar(prob, size=hbar_size, horizontal=True)
            # annotate top variants for each src position
            annot = []
            if not i:
                annot.append((len(c), len(c)+len(token), "top_src_token"))
            c += token

            variant_cells.append(c)
            variant_annots.append(annot)

        # what is the maximum width needed for the token column?
        col_width = max(len(c) for c in [entropy_cell] + variant_cells)

        # add padded entropy info to out_lines
        out_lines[0] += "{:{align}{width}}".format(entropy_cell, align=">", width=col_width)
        out_annot[0].append((total_width, total_width + col_width, "entropy_"+str(entr_levels[item_idx])))

        # add padded variants info to out_lines
        for i in range(len(out_lines)-1):
            c = ""
            if i < len(variant_cells):
                c = variant_cells[i]
                out_annot[i+1] += [
                    (total_width + s, total_width + e, a)
                    for s, e, a in variant_annots[i]
                ]
            out_lines[i+1] += "{:{align}{width}}".format(c, align="<", width=col_width)

        total_width += col_width
    return out_lines, out_annot


def entropy_levels(entropies, levels=5):
    # turning entropies to perplexities
    entropies = [2**x for x in entropies]
    # print(entropies, file=sys.stderr)
    # total = sum(entropies)
    mine = min(entropies)
    maxe = max(entropies)
    levels = [round((levels-1)*(x-mine)/(maxe-mine)) for x in entropies]
    # print(levels, file=sys.stderr)
    return levels


FULL_BLOCK = u"\u2588"
LEFT_ONE_EIGHT_BLOCK = u"\u258F"
LOWER_ONE_EIGHT_BLOCK = u"\u2581"


def proportion_bar(ratio, size=4, horizontal=True):
    full_block_num = int(size*ratio)
    last_block_prop = int((size*ratio - full_block_num) * 8)
    if horizontal:
        barstr = FULL_BLOCK*full_block_num
        if last_block_prop:
            barstr += chr(ord(LEFT_ONE_EIGHT_BLOCK)-last_block_prop+1)
        barstr += " "*(size-len(barstr))
        return barstr
    else:
        bararr = [FULL_BLOCK]*full_block_num
        if last_block_prop:
            bararr.insert(0, chr(ord(LOWER_ONE_EIGHT_BLOCK)+last_block_prop-1))
        for i in range(size-len(bararr)):
            bararr.insert(0, " ")
        return bararr


argparser = argparse.ArgumentParser()
argparser.add_argument("--no-color", action="store_true", default=False)
args = argparser.parse_args()

item_lines = []
for line in sys.stdin:
    if not line.strip():
        sent_info = parse_sentence_from_strlist(item_lines)
        print_sentence_info(sent_info)
        item_lines = []
    else:
        item_lines.append(line)

if len(item_lines) > 0:
    sent_info = parse_sentence_from_strlist(item_lines)
    print_sentence_info(sent_info)

