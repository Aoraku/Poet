import argparse
import json
import os

import config
from label import guess_form, rhyme_label, tone_label


TODO_DIR = os.path.join(config.DATA_DIR, "llm_todo")


def save_jsonl(path, data):
    config.mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def empty_lab(x, tone_data):
    lines = config.split_line(x.get("content", ""))
    last = lines[-1][-1] if lines else ""
    return {
        "id": x.get("id", ""),
        "kind": x.get("kind", ""),
        "title": x.get("title", ""),
        "author": x.get("author", ""),
        "cipai": x.get("cipai", ""),
        "theme": "",
        "season": "",
        "festival": "",
        "emotion": "",
        "style": "",
        "ci_style": "",
        "form": guess_form(x),
        "rhyme": rhyme_label(last),
        "tone": tone_label(last, tone_data),
    }


def make_todo(split, kind, start, limit):
    poems = config.load_jsonl(os.path.join(config.POEM_DIR, split + ".jsonl"))
    tone_data = config.load_json(config.TONE_FILE)
    rows = []
    n = 0
    for i, x in enumerate(poems):
        if not config.same_kind(x, kind):
            continue
        if n < start:
            n += 1
            continue
        lab = empty_lab(x, tone_data)
        lab["index"] = i
        lab["content"] = x.get("content", "")
        rows.append(lab)
        if len(rows) >= limit:
            break
    path = os.path.join(TODO_DIR, kind + "_" + split + "_" + str(start) + ".jsonl")
    save_jsonl(path, rows)
    print(path, len(rows))


def check_one(split, kind):
    poem_path = os.path.join(config.POEM_DIR, split + ".jsonl")
    lab_path = config.label_path(split, config.LLM_DIR)
    need = 0
    for x in config.load_jsonl(poem_path):
        if config.same_kind(x, kind):
            need += 1
    if not os.path.exists(lab_path):
        print(split, kind, "need", need, "have", 0)
        return
    poems, labs = config.load_labeled(split, kind, config.LLM_DIR)
    have = len(labs)
    ok = 0
    for x in labs:
        miss = False
        for key in config.kind_std(kind):
            if x.get(key, "") == "":
                miss = True
        if not miss:
            ok += 1
    print(split, kind, "need", need, "have", have, "ok", ok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--kind", default="poem", choices=["poem", "ci"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.make:
        make_todo(args.split, args.kind, args.start, args.limit)
    if args.check:
        for split in ["train", "vaid", "test"]:
            check_one(split, args.kind)


if __name__ == "__main__":
    main()
