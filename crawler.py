import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from opencc import OpenCC
from pypinyin import Style, lazy_pinyin


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATA_DIR, "raw")
POEM_DIR = os.path.join(DATA_DIR, "poems")
RHYME_DIR = os.path.join(DATA_DIR, "rhyme")
TONE_DIR = os.path.join(DATA_DIR, "tone")
SITE_DIR = os.path.join(DATA_DIR, "site_page")

API = "https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/"
RAW = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"

RANDOM_SEED = 42

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
}

SITE_URLS = {
    "quantangshi": "http://www.quantangshi.cn/",
    "quansongshi": "https://www.diancang.xyz/shicixiqu/10604/",
    "quansongci": "https://www.diancang.xyz/shicixiqu/quansongci/",
    "songci_shiku": "https://www.shiku.org/shiku/gs/songci/index.htm",
}

RHYME_URLS = {
    "pingshui": "https://sou-yun.cn/QR.aspx",
    "xinyun": "https://www.51xueci.com/cyss/yj14.htm",
}

cc = OpenCC("t2s")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def simp(text):
    return cc.convert(text or "")


def clean(text):
    text = simp(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[（(](按|案|以上|见|原注).*", "", text)
    # 删夹注
    text = re.sub(r"[（(][^（）()]{0,100}[）)]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def only_ch(text):
    return "".join(re.findall(r"[\u4e00-\u9fff]", text or ""))


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.encoding is None or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text


def get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=40)
    return r.json()


def save_text(path, text):
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path, data):
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, data):
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def api_files(folder):
    url = API + quote(folder) + "?ref=master"
    data = get_json(url)
    files = []
    for x in data:
        if x.get("type") == "file" and x.get("download_url"):
            files.append(x)
    return files


def raw_url(folder, name):
    return RAW + quote(folder) + "/" + quote(name)


def poem_names(prefix):
    if prefix == "poet.tang":
        return [f"poet.tang.{i}.json" for i in range(0, 58000, 1000)]
    if prefix == "poet.song":
        return [f"poet.song.{i}.json" for i in range(0, 255000, 1000)]
    return []


def ci_names():
    names = [f"ci.song.{i}.json" for i in range(0, 22000, 1000)]
    names.insert(3, "ci.song.2019y.json")
    return names


def save_site():
    mkdir(SITE_DIR)
    for name, url in SITE_URLS.items():
        print("site", name, url)
        html = get(url)
        save_text(os.path.join(SITE_DIR, name + ".html"), html)
        time.sleep(0.5)


def norm_poem(x, kind, dynasty, source, tone=None):
    paras = x.get("paragraphs") or []
    paras = [clean(p) for p in paras if only_ch(p)]
    content = "\n".join(paras)
    title = clean(x.get("title") or x.get("rhythmic") or "")
    cipai = clean(x.get("rhythmic") or "")

    return {
        "id": x.get("id", ""),
        "kind": kind,
        "dynasty": dynasty,
        "title": title,
        "cipai": cipai,
        "author": clean(x.get("author", "")),
        "paragraphs": paras,
        "content": content,
        "tone": tone or [],
        "source": source,
    }


def load_strains(name):
    url = raw_url("strains/json", name)
    data = get_json(url)
    return data


def crawl_poem(prefix, kind, dynasty, max_file):
    files = []
    for name in poem_names(prefix):
        files.append({"name": name, "download_url": raw_url("全唐诗", name)})
    if max_file:
        files = files[:max_file]

    all_data = []
    tone_dict = {}

    for i, x in enumerate(files):
        name = x["name"]
        print(kind, i + 1, "/", len(files), name)
        data = get_json(x["download_url"])
        strains = load_strains(name)
        for j, item in enumerate(data):
            tone = strains[j].get("strains", []) if j < len(strains) else []
            one = norm_poem(item, kind, dynasty, x["download_url"], tone)
            all_data.append(one)
            add_tone(tone_dict, one)
        time.sleep(0.2)

    save_json(os.path.join(RAW_DIR, kind + ".json"), all_data)
    return all_data, tone_dict


def crawl_ci(max_file):
    files = []
    for name in ci_names():
        files.append({"name": name, "download_url": raw_url("宋词", name)})
    if max_file:
        files = files[:max_file]

    all_data = []
    for i, x in enumerate(files):
        print("song_ci", i + 1, "/", len(files), x["name"])
        data = get_json(x["download_url"])
        for item in data:
            one = norm_poem(item, "song_ci", "宋代", x["download_url"])
            all_data.append(one)
        time.sleep(0.2)

    save_json(os.path.join(RAW_DIR, "song_ci.json"), all_data)
    return all_data


def add_tone(tone_dict, poem):
    paras = poem.get("paragraphs", [])
    strains = poem.get("tone", [])
    for p, s in zip(paras, strains):
        chars = only_ch(p)
        tones = re.findall(r"[平仄中]", s)
        for ch, tone in zip(chars, tones):
            if ch not in tone_dict:
                tone_dict[ch] = Counter()
            tone_dict[ch][tone] += 1


def make_tone(tone_dict):
    result = {}
    for ch, cnt in tone_dict.items():
        py = lazy_pinyin(ch, style=Style.TONE3)
        main = cnt.most_common(1)[0][0]
        result[ch] = {
            "pinyin": py[0] if py else "",
            "tone": main,
            "count": dict(cnt),
        }
    save_json(os.path.join(TONE_DIR, "char_tone.json"), result)


def make_cipai(ci_list):
    data = {}
    for ci in ci_list:
        name = ci.get("cipai", "")
        if not name:
            continue
        lens = []
        for p in ci.get("paragraphs", []):
            lens.append(len(only_ch(p)))
        if name not in data:
            data[name] = Counter()
        data[name]["-".join(str(x) for x in lens)] += 1

    result = {}
    for name, cnt in data.items():
        pattern, num = cnt.most_common(1)[0]
        result[name] = {
            "pattern": [int(x) for x in pattern.split("-") if x],
            "count": sum(cnt.values()),
            "top": cnt.most_common(5),
        }
    save_json(os.path.join(RHYME_DIR, "cipai_len.json"), result)


def save_rhyme():
    for name, url in RHYME_URLS.items():
        print("rhyme", name, url)
        text = get(url)
        soup = BeautifulSoup(text, "html.parser")
        save_text(os.path.join(RHYME_DIR, name + ".txt"), soup.get_text("\n", strip=True))
        time.sleep(0.5)


def split_data(data):
    random.seed(RANDOM_SEED)
    data = list(data)
    random.shuffle(data)
    n = len(data)
    n_train = int(n * 0.8)
    n_vaid = int(n * 0.1)
    train = data[:n_train]
    vaid = data[n_train:n_train + n_vaid]
    test = data[n_train + n_vaid:]
    write_jsonl(os.path.join(POEM_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(POEM_DIR, "vaid.jsonl"), vaid)
    write_jsonl(os.path.join(POEM_DIR, "test.jsonl"), test)
    return train, vaid, test


def check_data(train, vaid, test):
    print("train", len(train))
    print("vaid", len(vaid))
    print("test", len(test))
    for data in [train, vaid, test]:
        for x in data[:3]:
            assert x["content"]
            assert x["kind"] in ["tang_poem", "song_poem", "song_ci"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_file", type=int, default=0)
    parser.add_argument("--no_site", action="store_true")
    args = parser.parse_args()

    mkdir(DATA_DIR)
    if not args.no_site:
        save_site()

    tang, tang_tone = crawl_poem("poet.tang", "tang_poem", "唐代", args.max_file)
    song, song_tone = crawl_poem("poet.song", "song_poem", "宋代", args.max_file)
    ci = crawl_ci(args.max_file)

    tone_dict = defaultdict(Counter)
    for d in [tang_tone, song_tone]:
        for ch, cnt in d.items():
            tone_dict[ch].update(cnt)
    make_tone(tone_dict)
    make_cipai(ci)
    save_rhyme()

    train, vaid, test = split_data(tang + song + ci)
    check_data(train, vaid, test)


if __name__ == "__main__":
    main()
