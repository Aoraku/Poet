import json
import os
from collections import Counter

from label import CI_STYLE, EMOTION, FESTIVAL, SEASON, STYLE, THEME
from label import only_ch, rhyme_label, split_line, tone_label


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
POEM_DIR = os.path.join(DATA_DIR, "poems")
LLM_DIR = os.path.join(DATA_DIR, "llm_labels")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULT_DIR = os.path.join(DATA_DIR, "results")
TONE_FILE = os.path.join(DATA_DIR, "tone", "char_tone.json")
CIPAI_FILE = os.path.join(DATA_DIR, "rhyme", "cipai_len.json")
W2V_FILE = os.path.join(MODEL_DIR, "word2vec.model")

RANDOM_SEED = 42

CATEGORY = {
    "theme": THEME,
    "season": SEASON,
    "festival": FESTIVAL,
    "emotion": EMOTION,
    "style": STYLE,
    "ci_style": CI_STYLE,
}

POEM_STD = ["theme", "season", "festival", "emotion", "style", "form"]
CI_STD = ["theme", "season", "festival", "emotion", "style", "ci_style", "form"]

# 人工权重
POEM_WEIGHT = {
    "theme": 2.2,
    "season": 1.4,
    "festival": 1.8,
    "emotion": 2.0,
    "style": 1.7,
    "form": 1.2,
    "rhyme": 0.8,
    "author": 0.6,
}

CI_WEIGHT = {
    "theme": 2.0,
    "season": 1.2,
    "festival": 1.6,
    "emotion": 2.0,
    "style": 1.3,
    "ci_style": 2.2,
    "cipai": 1.7,
    "form": 1.4,
    "rhyme": 0.7,
    "author": 0.8,
}

POEM_FORMS = [
    "五言绝句", "七言绝句", "六言绝句",
    "五言律诗", "七言律诗", "五言排律", "七言排律",
    "五言古诗", "七言古诗", "杂言古诗", "古体长篇",
    "杂言乐府", "其他",
]

CI_FORMS = ["小令", "中调", "长调"]

FORM_NAMES = POEM_FORMS + CI_FORMS

COMMON_CIPAI = [
    "水调歌头", "满江红", "念奴娇", "沁园春", "贺新郎", "鹧鸪天",
    "浣溪沙", "蝶恋花", "临江仙", "菩萨蛮", "西江月", "清平乐",
    "虞美人", "江城子", "卜算子", "如梦令", "声声慢", "雨霖铃",
    "永遇乐", "摸鱼儿", "踏莎行", "渔家傲", "南乡子", "点绛唇",
]


def norm_kind(kind):
    if kind in ["诗", "poem", "tang_poem", "song_poem"]:
        return "poem"
    if kind in ["词", "ci", "song_ci"]:
        return "ci"
    return "all"


def same_kind(x, kind):
    kind = norm_kind(kind)
    if kind == "all":
        return True
    if kind == "poem":
        return x.get("kind") != "song_ci"
    if kind == "ci":
        return x.get("kind") == "song_ci"
    return True


def kind_std(kind):
    kind = norm_kind(kind)
    if kind == "ci":
        return list(CI_STD)
    return list(POEM_STD)


def make_input(kind, text, cipai=""):
    kind = norm_kind(kind)
    if kind == "ci":
        real_kind = "song_ci"
    else:
        real_kind = "tang_poem"
    return {
        "id": "",
        "kind": real_kind,
        "title": "输入",
        "author": "",
        "cipai": cipai,
        "content": text,
    }

# 作者先验
AUTHOR_THEME = {
    "王维": "山水田园",
    "孟浩然": "山水田园",
    "岑参": "边塞征战",
    "高适": "边塞征战",
    "王昌龄": "边塞征战",
    "杜甫": "忧国民生",
    "白居易": "忧国民生",
    "陆游": "忧国民生",
    "辛弃疾": "忧国民生",
    "柳永": "爱情闺怨",
    "李清照": "爱情闺怨",
    "苏轼": "哲理咏怀",
}

AUTHOR_STYLE = {
    "李白": "飘逸旷达",
    "杜甫": "沉郁悲慨",
    "王维": "冲淡闲远",
    "孟浩然": "清新自然",
    "韩愈": "劲健洗炼",
    "李商隐": "含蓄委曲",
    "杜牧": "高古典雅",
    "苏轼": "豪放慷慨",
    "辛弃疾": "豪放慷慨",
    "柳永": "婉约清丽",
    "李清照": "婉约清丽",
    "姜夔": "清空骚雅",
    "周邦彦": "典雅工丽",
}


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    if not os.path.exists(path):
        return {}
    return json.load(open(path, "r", encoding="utf-8"))


def load_jsonl(path, limit=0):
    data = []
    for line in open(path, "r", encoding="utf-8"):
        data.append(json.loads(line))
        if limit and len(data) >= limit:
            break
    return data


def save_json(path, data):
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cut(num, max_num):
    if max_num == 0:
        return 0
    v = num / max_num
    if v > 1:
        return 1
    return v


def add_count(feat, text, name, words, weight):
    s = 0
    for w in words:
        s += text.count(w)
    feat[name] = cut(s, 8) * weight


def add_shape(feat, x, weight):
    lines = split_line(x.get("content", ""))
    lens = [len(a) for a in lines if a]
    total = len(only_ch(x.get("content", "")))
    if not lens:
        return
    avg = sum(lens) / len(lens)
    diff = sum(abs(n - avg) for n in lens) / len(lens)
    feat["shape_char"] = cut(total, 260) * weight
    feat["shape_line"] = cut(len(lines), 50) * weight
    feat["shape_avg"] = cut(avg, 15) * weight
    feat["shape_diff"] = cut(diff, 8) * weight
    feat["shape_short"] = len([n for n in lens if n <= 4]) / len(lens) * weight
    feat["shape_long"] = len([n for n in lens if n >= 8]) / len(lens) * weight


def add_image(feat, text, weight):
    add_count(feat, text, "img_time", ["春", "秋", "夕", "夜", "晓", "暮", "年", "岁"], weight)
    add_count(feat, text, "img_place", ["山", "水", "江", "湖", "楼", "亭", "关", "塞", "宫", "村"], weight)
    add_count(feat, text, "img_body", ["心", "泪", "眉", "鬓", "梦", "魂", "愁", "恨"], weight)
    add_count(feat, text, "img_color", ["红", "绿", "青", "白", "黄", "翠", "碧", "金", "玉"], weight)
    add_count(feat, text, "img_war", ["兵", "戈", "剑", "马", "胡", "虏", "烽", "战", "戍"], weight)
    add_count(feat, text, "img_wine", ["酒", "杯", "醉", "酌", "宴", "歌", "舞"], weight)


def add_tone_stat(feat, x, tone_data, weight):
    lines = split_line(x.get("content", ""))
    ping = 0
    ze = 0
    ends = []
    for line in lines:
        if not line:
            continue
        ends.append(line[-1])
        for ch in line:
            t = tone_label(ch, tone_data)
            if t == "平":
                ping += 1
            if t == "仄":
                ze += 1
    total = ping + ze
    if total:
        feat["tone_ping"] = ping / total * weight
        feat["tone_ze"] = ze / total * weight
    rhymes = [rhyme_label(ch) for ch in ends if rhyme_label(ch) != "N/A"]
    if rhymes:
        cnt = Counter(rhymes)
        feat["rhyme_same"] = cnt.most_common(1)[0][1] / len(rhymes) * weight


def add_author_name(feat, x, weight):
    author = x.get("author", "")
    for name in ["李白", "杜甫", "白居易", "王维", "苏轼", "陆游", "辛弃疾", "柳永", "李清照"]:
        feat["author_" + name] = weight if author == name else 0


def word_scores(text, table):
    raw = {}
    for name, words in table.items():
        s = 0.0
        for w in words:
            c = text.count(w)
            if c:
                if len(w) == 1:
                    s += c * 0.6
                else:
                    s += c * (1.3 + 0.2 * len(w))
        raw[name] = s
    m = max(raw.values()) if raw else 0
    if m == 0:
        return {k: 0 for k in raw}
    return {k: raw[k] / m for k in raw}


def add_words(feat, text, key, weight):
    table = CATEGORY[key]
    scores = word_scores(text, table)
    for name, val in scores.items():
        feat[key + "_" + name] = val * weight


def form_name(x):
    if x["kind"] == "song_ci":
        total = len(only_ch(x["content"]))
        if total <= 58:
            return "小令"
        if total <= 90:
            return "中调"
        return "长调"

    lines = split_line(x["content"])
    lens = [len(a) for a in lines if a]
    if not lens:
        return "其他"
    common = Counter(lens).most_common(1)[0][0]
    ratio = lens.count(common) / len(lens)
    if ratio < 0.65:
        return "杂言乐府"
    if len(lines) == 4 and common == 5:
        return "五言绝句"
    if len(lines) == 4 and common == 7:
        return "七言绝句"
    if len(lines) == 8 and common == 5:
        return "五言律诗"
    if len(lines) == 8 and common == 7:
        return "七言律诗"
    if len(lines) > 8:
        return "古体长篇"
    return "其他"


def add_form(feat, x, weight):
    name = form_name(x)
    for form in FORM_NAMES:
        feat["form_" + form] = weight if form == name else 0

    lines = split_line(x["content"])
    lens = [len(a) for a in lines if a]
    total = len(only_ch(x["content"]))
    avg_len = sum(lens) / len(lens) if lens else 0
    feat["num_char"] = cut(total, 220)
    feat["num_line"] = cut(len(lines), 40)
    feat["avg_len"] = cut(avg_len, 14)
    feat["five_ratio"] = lens.count(5) / len(lens) if lens else 0
    feat["seven_ratio"] = lens.count(7) / len(lens) if lens else 0


def add_rhyme(feat, x, tone_data, weight):
    lines = split_line(x["content"])
    last = lines[-1][-1] if lines else ""
    rhyme = rhyme_label(last)
    tone = tone_label(last, tone_data)
    if rhyme != "N/A":
        feat["rhyme_" + rhyme] = weight
    if tone != "N/A":
        feat["tone_" + tone] = weight


def add_author(feat, x, weight):
    author = x.get("author", "")
    if author in AUTHOR_THEME:
        feat["author_theme_" + AUTHOR_THEME[author]] = weight
    if author in AUTHOR_STYLE:
        feat["author_style_" + AUTHOR_STYLE[author]] = weight


def add_cipai(feat, x, weight):
    name = x.get("cipai", "") or x.get("title", "")
    for cipai in COMMON_CIPAI:
        feat["cipai_" + cipai] = weight if name == cipai else 0

    data = load_json(CIPAI_FILE)
    pattern = data.get(name, {}).get("pattern", [])
    feat["cipai_len"] = cut(sum(pattern), 120)
    feat["cipai_line"] = cut(len(pattern), 20)


def add_w2v(feat, x, w2v):
    if w2v is None:
        return
    chars = [ch for ch in only_ch(x.get("content", "")) if ch in w2v.wv]
    if not chars:
        return
    vec = None
    for ch in chars:
        if vec is None:
            vec = w2v.wv[ch].copy()
        else:
            vec += w2v.wv[ch]
    vec = vec / len(chars)
    for i, v in enumerate(vec):
        feat["w2v_" + str(i)] = float((v + 1) / 2)


def load_w2v(path=W2V_FILE):
    if not os.path.exists(path):
        return None
    from gensim.models import Word2Vec
    return Word2Vec.load(path)


def poem_feat(x, tone_data=None, w2v=None):
    tone_data = tone_data or {}
    text = x.get("title", "") + " " + x.get("author", "") + " " + x.get("content", "")
    feat = {}
    add_words(feat, text, "theme", POEM_WEIGHT["theme"])
    add_words(feat, text, "season", POEM_WEIGHT["season"])
    add_words(feat, text, "festival", POEM_WEIGHT["festival"])
    add_words(feat, text, "emotion", POEM_WEIGHT["emotion"])
    add_words(feat, text, "style", POEM_WEIGHT["style"])
    add_form(feat, x, POEM_WEIGHT["form"])
    add_rhyme(feat, x, tone_data, POEM_WEIGHT["rhyme"])
    add_author(feat, x, POEM_WEIGHT["author"])
    add_author_name(feat, x, 0.5)
    add_shape(feat, x, 1.2)
    add_image(feat, text, 1.4)
    add_tone_stat(feat, x, tone_data, 1.0)
    add_w2v(feat, x, w2v)
    return feat


def ci_feat(x, tone_data=None, w2v=None):
    tone_data = tone_data or {}
    text = x.get("title", "") + " " + x.get("cipai", "") + " " + x.get("author", "") + " " + x.get("content", "")
    feat = {}
    add_words(feat, text, "theme", CI_WEIGHT["theme"])
    add_words(feat, text, "season", CI_WEIGHT["season"])
    add_words(feat, text, "festival", CI_WEIGHT["festival"])
    add_words(feat, text, "emotion", CI_WEIGHT["emotion"])
    add_words(feat, text, "style", CI_WEIGHT["style"])
    add_words(feat, text, "ci_style", CI_WEIGHT["ci_style"])
    add_form(feat, x, CI_WEIGHT["form"])
    add_rhyme(feat, x, tone_data, CI_WEIGHT["rhyme"])
    add_author(feat, x, CI_WEIGHT["author"])
    add_cipai(feat, x, CI_WEIGHT["cipai"])
    add_author_name(feat, x, 0.7)
    add_shape(feat, x, 1.6)
    add_image(feat, text, 1.5)
    add_tone_stat(feat, x, tone_data, 0.9)
    add_w2v(feat, x, w2v)
    return feat


def build_feat(x, tone_data=None, w2v=None):
    if x.get("kind") == "song_ci":
        return ci_feat(x, tone_data, w2v)
    return poem_feat(x, tone_data, w2v)


def label_path(name, label_dir=None):
    label_dir = label_dir or LLM_DIR
    return os.path.join(label_dir, name + "_labels.jsonl")


def load_labels(name, label_dir=None, limit=0):
    path = label_path(name, label_dir)
    return load_jsonl(path, limit)


def label_map(name, label_dir=None):
    labels = load_labels(name, label_dir)
    by_id = {}
    by_index = {}
    for x in labels:
        if x.get("id", ""):
            by_id[x["id"]] = x
        if "index" in x:
            by_index[int(x["index"])] = x
    return by_id, by_index


def load_split(name, limit=0, label_dir=None):
    poem_path = os.path.join(POEM_DIR, name + ".jsonl")
    poems = load_jsonl(poem_path, limit)
    labels = load_labels(name, label_dir, limit)
    return poems, labels


def load_labeled(name, kind, label_dir=None, limit=0):
    poems = load_jsonl(os.path.join(POEM_DIR, name + ".jsonl"))
    by_id, by_index = label_map(name, label_dir)
    data = []
    labs = []
    for i, x in enumerate(poems):
        if not same_kind(x, kind):
            continue
        lab = by_id.get(x.get("id", ""))
        if lab is None:
            lab = by_index.get(i)
        if lab is None:
            continue
        data.append(x)
        labs.append(lab)
        if limit and len(data) >= limit:
            break
    return data, labs
