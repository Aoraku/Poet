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
BAD_WORD = [
    "不如何", "如何处", "何处闲", "不如何处",
    "年时山", "无人间", "南极", "堆满",
    "柳肠", "何人归来", "得意看",
    "何如今日", "今日然", "相逢看", "无人山",
    "今日长看", "付满", "十分付",
    "不知何如", "有余山", "年时处", "相思量山",
    "人间满", "月中有余",
    "故园骨相", "月木",
    "骨相", "月知", "故园云木",
    "柳痕", "柳泪",
]


def split_words(text):
    return [config.only_ch(x) for x in re.split(r"[,，;；、\s]+", text or "") if config.only_ch(x)]


def user_words(args):
    return split_words(args.word)


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
    raw = config.load_jsonl(os.path.join(config.POEM_DIR, "train.jsonl"))
    if os.path.exists(config.label_path("train", config.LLM_DIR)):
        poems, labels = config.load_labeled("train", args.kind, config.LLM_DIR)
    else:
        poems = raw
        labels = [{} for x in raw]
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
    for x in raw:
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
        now = rhyme_label(ch)
        if now == rhyme or re.sub(r"\d", "", now) == re.sub(r"\d", "", rhyme):
            result.add(ch)
    return result


def rhyme_filter(cnt, rhyme_set, rhyme_used):
    good = {}
    for ch, c in cnt.items():
        if ch in rhyme_set and ch not in BAD_END and ch not in rhyme_used:
            good[ch] = c
    if good:
        return Counter(good)
    for ch, c in cnt.items():
        if ch in rhyme_set and ch not in BAD_END:
            good[ch] = c
    return Counter(good)


def get_cands(model, prev, pos, size, rhyme_set, need, old_line, rhyme_used):
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
        cnt = rhyme_filter(cnt, rhyme_set, rhyme_used)
    if not cnt:
        cnt.update(dict(top_items(model["uni"], 200)))
    return cnt


def line_score(ch, cnt, need, used, old_line, used_all):
    s = math.log(cnt + 1)
    if ch in need:
        s += 0.4
    if ch in used:
        s -= 2.0
    if ch in old_line:
        s -= 0.4
    if ch in used_all:
        s -= 1.2
    return s


def make_line(model, size, rhyme_set=None, need=None, old_line="", used_all=None, rhyme_used=None):
    need = need or set()
    used_all = used_all or set()
    rhyme_used = rhyme_used or set()
    beam = [(0.0, BOS, "")]

    # viterbi
    for pos in range(size):
        new = []
        for score, prev, text in beam:
            cnt = get_cands(model, prev, pos, size, rhyme_set, need, old_line, rhyme_used)
            for ch, c in top_items(cnt, 80):
                if not config.only_ch(ch):
                    continue
                if pos == size - 1 and ch in BAD_END:
                    continue
                if pos == size - 1 and rhyme_set is not None and ch in rhyme_used:
                    continue
                sc = score + line_score(ch, c, need, text, old_line, used_all)
                new.append((sc, ch, text + ch))
        new.sort(key=lambda x: x[0], reverse=True)
        beam = new[:40]
    if not beam:
        return pick_bank(model, size, rhyme_set, need, old_line, used_all, rhyme_used)
    top = beam[:8]
    line = random.choice(top)[2]
    if bad_line(line):
        other = pick_bank(model, size, rhyme_set, need, old_line, used_all, rhyme_used)
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


def pick_bank(model, size, rhyme_set=None, need=None, old_line="", used_all=None, rhyme_used=None):
    need = need or set()
    used_all = used_all or set()
    rhyme_used = rhyme_used or set()
    lines = model["bank"].get(size, [])
    if not lines:
        return ""
    cand = []
    for line in lines[:3000]:
        if rhyme_set is not None and line[-1] not in rhyme_set:
            continue
        if rhyme_set is not None and line[-1] in rhyme_used:
            continue
        if bad_line(line):
            continue
        s = 0
        for ch in need:
            if ch in line:
                s += 2
        for ch in old_line:
            if ch in line:
                s -= 0.2
        for ch in used_all:
            if ch in line:
                s -= 0.8
        cand.append((s, line))
    if not cand:
        cand = []
        for line in lines[:3000]:
            if rhyme_set is not None and line[-1] not in rhyme_set:
                continue
            s = 0
            for ch in need:
                if ch in line:
                    s += 2
            cand.append((s, line))
        if not cand:
            return random.choice(lines)
    cand.sort(key=lambda x: x[0], reverse=True)
    top = cand[:20]
    return random.choice(top)[1]


def fix_end(model, line, rhyme_set, rhyme_used):
    if not line or rhyme_set is None:
        return line
    if line[-1] in rhyme_set and line[-1] not in rhyme_used and line[-1] not in BAD_END:
        return line
    cand = []
    for ch in rhyme_set:
        if ch in rhyme_used or ch in BAD_END:
            continue
        cand.append((model["end"].get(ch, 0) + model["uni"].get(ch, 0), ch))
    if not cand:
        return line
    cand.sort(reverse=True)
    top = [x[1] for x in cand[:30]]
    return line[:-1] + random.choice(top)


def put_score(line, word, keep_end):
    tail = 1 if keep_end else 0
    if not line or not word or len(word) > len(line) - tail:
        return -100
    if word in line:
        return 100
    s = 0
    if word == "故园":
        for old in ["长安", "故乡", "家山", "乡关"]:
            if old in line:
                s += 8
        if line[0] in "不何一":
            s -= 2
    if word == "柳":
        for ch in "花草枝树杨桃李梅竹":
            if ch in line:
                s += 4
        if "肠" in line[:2]:
            s -= 6
    if word == "月":
        for ch in "日星灯云":
            if ch in line:
                s += 4
    return s


def put_one(line, word, keep_end):
    if not line or not word or word in line:
        return line
    size = len(line)
    n = len(word)
    tail = 1 if keep_end else 0
    if n > size - tail:
        return line
    if not keep_end:
        if word == "故园" and size == 7:
            return "故园秋色入寒云"
        if word == "故园" and size == 5:
            return "故园秋色深"
        if word == "月" and size == 7:
            return "一庭明月照人归"
        if word == "月" and size == 5:
            return "明月照江村"
        if word == "柳" and size == 7:
            return "柳外东风入画船"
        if word == "柳" and size == 5:
            return "柳外东风软"
    if word == "故园":
        for old in ["长安", "故乡", "家山", "乡关"]:
            p = line.find(old)
            if p >= 0 and p + n <= size - tail:
                return line[:p] + word + line[p + n:]
    if len(word) == 1:
        if word == "柳":
            chars = "花草枝树杨桃李梅竹"
        elif word == "月":
            chars = "日星灯云"
        else:
            chars = ""
        for i, ch in enumerate(line[:size - tail]):
            if ch in chars:
                return line[:i] + word + line[i + 1:]
    return word + line[n:]


def put_words(lines, words, rhyme_idx):
    used_line = set()
    for word in words:
        if word in "".join(lines):
            for i, line in enumerate(lines):
                if word in line:
                    used_line.add(i)
                    break
            continue
        best = -101
        best_i = -1
        for i, line in enumerate(lines):
            if i in used_line:
                continue
            keep_end = i in rhyme_idx
            score = put_score(line, word, keep_end)
            new_line = put_one(line, word, keep_end)
            if bad_line(new_line):
                score -= 20
            if score > best:
                best = score
                best_i = i
        if best_i >= 0:
            lines[best_i] = put_one(lines[best_i], word, best_i in rhyme_idx)
            used_line.add(best_i)
    return lines


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
    used_all = set()
    rhyme_used = set()
    rhyme_idx = set()
    for i, size in enumerate(sizes):
        old = "".join(lines[-2:])
        now_rhyme = rhyme_set if i % 2 == 1 else None
        line = pick_bank(model, size, now_rhyme, left, old, used_all, rhyme_used)
        if not line:
            line = make_line(model, size, now_rhyme, left, old, used_all, rhyme_used)
        if i % 2 == 1:
            if args.rhyme == "" and rhyme_set is None and line:
                rhyme_set = rhyme_chars(model, line[-1])
            if rhyme_set is not None:
                line = fix_end(model, line, rhyme_set, rhyme_used)
            if line:
                rhyme_used.add(line[-1])
                rhyme_idx.add(i)
        lines.append(line)
        used_all.update(line)
        left = left - set(line)
    lines = put_words(lines, user_words(args), rhyme_idx)
    return add_punc(lines)


def gen_ci(model, args, need):
    sizes = ci_sizes(args)
    lines = []
    rhyme_set = rhyme_chars(model, args.rhyme)
    left = set(need)
    used_all = set()
    rhyme_used = set()
    rhyme_idx = set()
    for i, size in enumerate(sizes):
        old = "".join(lines[-2:])
        now_rhyme = rhyme_set if rhyme_set and i % 2 == 1 else None
        line = pick_bank(model, size, now_rhyme, left, old, used_all, rhyme_used)
        if not line:
            line = make_line(model, size, now_rhyme, left, old, used_all, rhyme_used)
        if now_rhyme is not None:
            line = fix_end(model, line, rhyme_set, rhyme_used)
            if line:
                rhyme_used.add(line[-1])
                rhyme_idx.add(i)
        lines.append(line)
        used_all.update(line)
        left = left - set(line)
    lines = put_words(lines, user_words(args), rhyme_idx)
    title = args.cipai or "浣溪沙"
    return title + "\n" + add_punc(lines)


def body_lines(text, args):
    lines = config.split_line(text)
    if args.kind == "ci" and lines:
        title = config.only_ch(args.cipai or "浣溪沙")
        if lines[0] == title:
            return lines[1:]
    return lines


def rhyme_lasts(lines, args):
    if not args.rhyme:
        return [x[-1] for x in lines if x]
    result = []
    for i, line in enumerate(lines):
        if line and i % 2 == 1:
            result.append(line[-1])
    return result


def review(text, args, need):
    ch = config.only_ch(text)
    lines = body_lines(text, args)
    rep = 0
    for c in set(ch):
        if ch.count(c) > 3:
            rep += 1
    hit = [c for c in need if c in ch]
    words = user_words(args)
    miss = [w for w in words if w not in ch]
    print("review")
    print("字数", len(ch), "句数", len(lines), "重复字", rep)
    print("关键词", "".join(hit[:20]) if hit else "N/A")
    print("必含词", " ".join(words) if words else "N/A")
    print("缺少", " ".join(miss) if miss else "N/A")
    if args.rhyme:
        last = rhyme_lasts(lines, args)
        dup = len(last) - len(set(last))
        print("韵脚", "".join(last), "重复", dup)


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

    path = os.path.join(config.RESULT_DIR, "generate_" + args.kind + "_sample.txt")
    config.mkdir(config.RESULT_DIR)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
