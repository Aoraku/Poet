import argparse
import json
import os
import time

import requests
from tqdm import tqdm

import config
from label import guess_form, rhyme_label, tone_label


BASE_URL = "https://litellm.nbdevenv.xiaoaojianghu.fun"
MODEL = "gemini-3-flash-preview"
SPLITS = ["train", "vaid", "test"]
KINDS = ["poem", "ci"]


def label_opts(kind):
    opts = {}
    keys = ["theme", "season", "festival", "emotion", "style"]
    if kind == "ci":
        keys.append("ci_style")
    for key in keys:
        opts[key] = list(config.CATEGORY[key].keys()) + ["N/A"]
    if kind == "ci":
        opts["form"] = list(config.CI_FORMS)
    else:
        opts["form"] = list(config.POEM_FORMS)
    return opts


def make_prompt(x, kind, idx):
    opts = label_opts(kind)
    item = {
        "index": idx,
        "kind": kind,
        "title": x.get("title", ""),
        "author": x.get("author", ""),
        "dynasty": x.get("dynasty", ""),
        "cipai": x.get("cipai", ""),
        "content": x.get("content", ""),
    }
    keys = list(opts.keys())
    text = []
    text.append("你是中国古典诗词标注员。请逐字阅读作品，不要用简单规则猜。")
    text.append("每个字段只能从候选标签里选一个。能归类就归类，只有真的看不出来才写 N/A。")
    text.append("season 没有季节就写 N/A，festival 没有节日就写 N/A。")
    text.append("只输出一个 JSON，不要解释，不要 markdown。")
    text.append("字段: " + ",".join(keys))
    text.append("候选标签:")
    text.append(json.dumps(opts, ensure_ascii=False))
    text.append("作品:")
    text.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(text)


def ask_api(prompt, args):
    key = os.environ.get("LITELLM_KEY", "")
    if not key:
        print("先设置 LITELLM_KEY")
        return {}
    url = args.base.rstrip("/") + "/v1/chat/completions"
    data = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    head = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    r = requests.post(url, headers=head, json=data, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]["content"]
    return read_json(msg)


def read_json(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    a = text.find("{")
    b = text.rfind("}")
    if a >= 0 and b >= 0:
        text = text[a:b + 1]
    return json.loads(text)


def fix_lab(raw, x, kind, idx, tone_data):
    lines = config.split_line(x.get("content", ""))
    last = lines[-1][-1] if lines else ""
    lab = {
        "index": idx,
        "id": x.get("id", ""),
        "kind": x.get("kind", ""),
        "title": x.get("title", ""),
        "author": x.get("author", ""),
        "cipai": x.get("cipai", ""),
        "theme": "N/A",
        "season": "N/A",
        "festival": "N/A",
        "emotion": "N/A",
        "style": "N/A",
        "ci_style": "N/A",
        "form": guess_form(x),
        "rhyme": rhyme_label(last),
        "tone": tone_label(last, tone_data),
    }
    opts = label_opts(kind)
    for key in opts:
        val = raw.get(key, "")
        if val in opts[key]:
            lab[key] = val
    if lab["form"] not in opts["form"]:
        lab["form"] = guess_form(x)
    if kind == "poem":
        lab["ci_style"] = "N/A"
    return lab


def save_rows(path, rows):
    config.mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")


def old_rows(path, kind, fresh):
    if not os.path.exists(path):
        return [], set(), set()
    rows = config.load_jsonl(path)
    if fresh:
        rows = [x for x in rows if not config.same_kind(x, kind)]
        save_rows(path, rows)
    ids = set()
    idxs = set()
    for x in rows:
        if config.same_kind(x, kind):
            ids.add(x.get("id", ""))
            if "index" in x:
                idxs.add(int(x["index"]))
    return rows, ids, idxs


def get_tasks(args):
    if args.split == "all":
        splits = SPLITS
    else:
        splits = [args.split]
    if args.kind == "all":
        kinds = KINDS
    else:
        kinds = [args.kind]
    tasks = []
    for split in splits:
        for kind in kinds:
            tasks.append((split, kind))
    return tasks


def clear_old(tasks):
    by_split = {}
    for split, kind in tasks:
        by_split.setdefault(split, set()).add(kind)
    for split, kinds in by_split.items():
        path = config.label_path(split, config.LLM_DIR)
        if not os.path.exists(path):
            continue
        if "poem" in kinds and "ci" in kinds:
            os.remove(path)
            print("clear", path)
            continue
        rows = config.load_jsonl(path)
        keep = []
        for x in rows:
            drop = False
            for kind in kinds:
                if config.same_kind(x, kind):
                    drop = True
            if not drop:
                keep.append(x)
        save_rows(path, keep)
        print("clear", path)


def todo_num(args, split, kind):
    poems = config.load_jsonl(os.path.join(config.POEM_DIR, split + ".jsonl"))
    path = config.label_path(split, config.LLM_DIR)
    rows, ids, idxs = old_rows(path, kind, False)
    n = 0
    pos = -1
    for idx, x in enumerate(poems):
        if not config.same_kind(x, kind):
            continue
        pos += 1
        if pos < args.start:
            continue
        if x.get("id", "") in ids or idx in idxs:
            continue
        n += 1
        if args.limit and n >= args.limit:
            break
    return n


def run_one(args, split, kind, bar):
    poems = config.load_jsonl(os.path.join(config.POEM_DIR, split + ".jsonl"))
    path = config.label_path(split, config.LLM_DIR)
    tone_data = config.load_json(config.TONE_FILE)
    rows, ids, idxs = old_rows(path, kind, False)
    config.mkdir(os.path.dirname(path))
    out = open(path, "a", encoding="utf-8")
    n = 0
    pos = -1
    for idx, x in enumerate(poems):
        if not config.same_kind(x, kind):
            continue
        pos += 1
        if pos < args.start:
            continue
        if x.get("id", "") in ids or idx in idxs:
            continue
        raw = ask_api(make_prompt(x, kind, idx), args)
        lab = fix_lab(raw, x, kind, idx, tone_data)
        out.write(json.dumps(lab, ensure_ascii=False, separators=(",", ":")) + "\n")
        out.flush()
        ids.add(lab["id"])
        idxs.add(idx)
        n += 1
        bar.set_description(split + " " + kind)
        bar.set_postfix_str(lab["title"][:12])
        bar.update(1)
        if args.limit and n >= args.limit:
            break
        if args.sleep:
            time.sleep(args.sleep)
    out.close()
    print("done", split, kind, n)


def run_many(args):
    tasks = get_tasks(args)
    if args.fresh:
        clear_old(tasks)
    total = 0
    for split, kind in tasks:
        total += todo_num(args, split, kind)
    bar = tqdm(total=total, ncols=100)
    for split, kind in tasks:
        run_one(args, split, kind, bar)
    bar.close()
    print("all done", total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "vaid", "test", "all"])
    parser.add_argument("--kind", default="poem", choices=["poem", "ci", "all"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base", default=BASE_URL)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    run_many(args)


if __name__ == "__main__":
    main()
