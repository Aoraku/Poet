import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

from label import rhyme_label

import config


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BOS = "<B>"

FORM_SPEC = {
    "五言绝句": [5, 5, 5, 5],
    "七言绝句": [7, 7, 7, 7],
    "五言律诗": [5, 5, 5, 5, 5, 5, 5, 5],
    "七言律诗": [7, 7, 7, 7, 7, 7, 7, 7],
}

BAD_END = set("的不了在是有和与以为之其而也")
BAD_WORD = ["不如何", "如何处", "何处闲", "不如何处"]


def split_words(text):
    return [x for x in re.split(r"[,，;；、\s]+", text or "") if x]


def get_need(args):
    words = []
    for key in ["theme", "emotion", "style", "ci_style"]:
        val = getattr(args, key)
        if val and key in config.CATEGORY:
            words += config.CATEGORY[key].get(val, [])
    words += split_words(args.word)
    chars = set()
    for w in words:
        for ch in config.only_ch(w):
            chars.add(ch)
    return chars


def load_data(args):
    poems, labels = config.load_split("train", limit=0)
    data = []
    for x, lab in zip(poems, labels):
        if args.kind == "poem" and x.get("kind") == "song_ci":
            continue
        if args.kind == "ci" and x.get("kind") != "song_ci":
            continue
        if args.cipai and x.get("cipai") != args.cipai:
            continue
        ok = True
        for key in ["theme", "emotion", "style", "ci_style"]:
            val = getattr(args, key)
            if val and lab.get(key) != val:
                ok = False
        if ok:
            data.append(x)
        if args.limit and len(data) >= args.limit:
            break

    if len(data) >= 50:
        return data

    data = []
    for x in poems:
        if args.kind == "poem" and x.get("kind") == "song_ci":
            continue
        if args.kind == "ci" and x.get("kind") != "song_ci":
            continue
        data.append(x)
        if args.limit and len(data) >= args.limit:
            break
    return data


def train_model(data):
    uni = Counter()
    start = Counter()
    end = Counter()
    bi = defaultdict(Counter)
    bank = defaultdict(list)

    for x in data:
        for line in config.split_line(x.get("content", "")):
            chars = list(config.only_ch(line))
            if len(chars) < 2:
                continue
            bank[len(chars)].append("".join(chars))
            start[chars[0]] += 1
            end[chars[-1]] += 1
            prev = BOS
            for ch in chars:
                uni[ch] += 1
                bi[prev][ch] += 1
                prev = ch
    return {"uni": uni, "start": start, "end": end, "bi": bi, "bank": bank}


def top_items(cnt, n):
    if not cnt:
        return []
    return cnt.most_common(n)


def rhyme_chars(model, rhyme):
    if not rhyme:
        return None
    rhyme = rhyme.strip()
    if len(config.only_ch(rhyme)) == 1:
        rhyme = rhyme_label(rhyme)
    result = set()
    for ch in model["uni"]:
        if rhyme_label(ch) == rhyme:
            result.add(ch)
    return result


def get_cands(model, prev, pos, size, rhyme_set, need, old_line):
    cnt = Counter()
    cnt.update(dict(top_items(model["bi"].get(prev, Counter()), 160)))
    if pos == 0:
        cnt.update(dict(top_items(model["start"], 120)))
    if pos == size - 1:
        cnt.update(dict(top_items(model["end"], 160)))
    if old_line:
        for ch in old_line[-2:]:
            cnt[ch] += 0.5
    if rhyme_set is not None and pos == size - 1:
        cnt = Counter({ch: c for ch, c in cnt.items() if ch in rhyme_set and ch not in BAD_END})
    if not cnt:
        cnt.update(dict(top_items(model["uni"], 200)))
    return cnt


def line_score(ch, cnt, need, used, old_line):
    s = math.log(cnt + 1)
    if ch in need:
        s += 0.2
    if ch in used:
        s -= 2.0
    if ch in old_line:
        s -= 0.4
    return s


def make_line(model, size, rhyme_set=None, need=None, old_line=""):
    need = need or set()
    beam = [(0.0, BOS, "")]

    # viterbi
    for pos in range(size):
        new = []
        for score, prev, text in beam:
            cnt = get_cands(model, prev, pos, size, rhyme_set, need, old_line)
            for ch, c in top_items(cnt, 80):
                if not config.only_ch(ch):
                    continue
                if pos == size - 1 and ch in BAD_END:
                    continue
                sc = score + line_score(ch, c, need, text, old_line)
                new.append((sc, ch, text + ch))
        new.sort(key=lambda x: x[0], reverse=True)
        beam = new[:40]
    if not beam:
        return pick_bank(model, size, rhyme_set, need, old_line)
    top = beam[:8]
    line = random.choice(top)[2]
    if bad_line(line):
        other = pick_bank(model, size, rhyme_set, need, old_line)
        if other:
            return other
    return line


def bad_line(line):
    if not line:
        return True
    for w in BAD_WORD:
        if w in line:
            return True
    if len(set(line)) / len(line) < 0.65:
        return True
    grams = Counter()
    for i in range(len(line) - 1):
        grams[line[i:i + 2]] += 1
    if grams and grams.most_common(1)[0][1] > 1:
        return True
    return False


def pick_bank(model, size, rhyme_set=None, need=None, old_line=""):
    need = need or set()
    lines = model["bank"].get(size, [])
    if not lines:
        return ""
    cand = []
    for line in lines[:3000]:
        if rhyme_set is not None and line[-1] not in rhyme_set:
            continue
        s = 0
        for ch in need:
            if ch in line:
                s += 2
        for ch in old_line:
            if ch in line:
                s -= 0.2
        cand.append((s, line))
    if not cand:
        return random.choice(lines)
    cand.sort(key=lambda x: x[0], reverse=True)
    top = cand[:20]
    return random.choice(top)[1]


def poem_sizes(args):
    return FORM_SPEC.get(args.form, FORM_SPEC["七言绝句"])


def ci_sizes(args):
    data = config.load_json(config.CIPAI_FILE)
    if args.cipai and args.cipai in data:
        return fix_sizes(data[args.cipai]["pattern"])
    if "浣溪沙" in data:
        return fix_sizes(data["浣溪沙"]["pattern"])
    return [7, 7, 7, 7, 7, 7]


def fix_sizes(sizes):
    result = []
    for n in sizes:
        if n <= 9:
            result.append(n)
        else:
            while n > 9:
                result.append(7)
                n -= 7
            if n > 0:
                result.append(n)
    return result


def add_punc(lines):
    out = []
    for i, line in enumerate(lines):
        p = "。" if i == len(lines) - 1 or i % 2 == 1 else "，"
        out.append(line + p)
    return "\n".join(out)


def gen_poem(model, args, need):
    sizes = poem_sizes(args)
    lines = []
    rhyme_set = rhyme_chars(model, args.rhyme)
    left = set(need)
    for i, size in enumerate(sizes):
        old = "".join(lines[-2:])
        now_rhyme = rhyme_set if i % 2 == 1 else None
        line = make_line(model, size, now_rhyme, left, old)
        lines.append(line)
        left = left - set(line)
        if args.rhyme == "" and i == 1 and line:
            rhyme_set = rhyme_chars(model, line[-1])
    return add_punc(lines)


def gen_ci(model, args, need):
    sizes = ci_sizes(args)
    lines = []
    rhyme_set = rhyme_chars(model, args.rhyme)
    left = set(need)
    for i, size in enumerate(sizes):
        old = "".join(lines[-2:])
        now_rhyme = rhyme_set if rhyme_set and i % 2 == 1 else None
        line = pick_bank(model, size, now_rhyme, left, old)
        if not line:
            line = make_line(model, size, now_rhyme, left, old)
        lines.append(line)
        left = left - set(line)
    title = args.cipai or "浣溪沙"
    return title + "\n" + add_punc(lines)


def review(text, args, need):
    ch = config.only_ch(text)
    lines = config.split_line(text)
    rep = 0
    for c in set(ch):
        if ch.count(c) > 3:
            rep += 1
    hit = [c for c in need if c in ch]
    print("review")
    print("字数", len(ch), "句数", len(lines), "重复字", rep)
    print("关键词", "".join(hit[:20]) if hit else "N/A")
    if args.rhyme:
        last = [x[-1] for x in lines if x]
        print("韵脚", "".join(last))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="poem", choices=["poem", "ci"])
    parser.add_argument("--form", default="七言绝句")
    parser.add_argument("--theme", default="")
    parser.add_argument("--emotion", default="")
    parser.add_argument("--style", default="")
    parser.add_argument("--ci_style", default="")
    parser.add_argument("--cipai", default="")
    parser.add_argument("--rhyme", default="")
    parser.add_argument("--word", default="")
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    data = load_data(args)
    model = train_model(data)
    need = get_need(args)
    if args.kind == "ci":
        text = gen_ci(model, args, need)
    else:
        text = gen_poem(model, args, need)
    print(text)
    print()
    review(text, args, need)

    path = os.path.join(config.RESULT_DIR, "generate_sample.txt")
    config.mkdir(config.RESULT_DIR)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
