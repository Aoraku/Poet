import argparse
import json
import os
import re
from collections import Counter

from opencc import OpenCC
from pypinyin import Style, lazy_pinyin


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
POEM_DIR = os.path.join(DATA_DIR, "poems")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
TONE_FILE = os.path.join(DATA_DIR, "tone", "char_tone.json")

cc = OpenCC("t2s")

THEME = {
    "山水田园": ["山", "水", "溪", "泉", "林", "峰", "云", "松", "竹", "田", "园", "村", "渔", "樵", "归隐"],
    "边塞征战": ["边", "塞", "关", "胡", "羌", "烽", "戍", "将军", "战", "马", "楼兰", "玉门"],
    "送别留赠": ["送", "别", "离", "赠", "长亭", "折柳", "孤帆", "故人", "南浦", "饯"],
    "羁旅思乡": ["客", "旅", "乡", "故园", "归", "梦", "雁", "寒灯", "天涯", "驿"],
    "咏史怀古": ["古", "史", "怀古", "故垒", "废", "秦", "汉", "兴亡", "英雄", "前朝"],
    "咏物言志": ["梅", "兰", "竹", "菊", "松", "莲", "蝉", "鹤", "凌霜", "岁寒"],
    "爱情闺怨": ["郎", "妾", "闺", "相思", "鸳鸯", "红豆", "罗衣", "泪", "断肠"],
    "忧国民生": ["国", "社稷", "苍生", "兵", "乱", "饥", "田家", "悯农", "中原"],
    "酬唱宴游": ["酬", "和", "答", "宴", "酒", "歌", "舞", "楼", "亭", "友"],
    "哲理咏怀": ["人生", "天地", "浮生", "荣枯", "得失", "功名", "老", "空", "尘"],
    "祝颂应制": ["皇", "帝", "圣", "寿", "万岁", "郊祀", "导引", "庆", "瑞"],
    "悼亡祭奠": ["悼", "哭", "挽", "祭", "墓", "孤坟", "哀", "魂", "追思"],
}

SEASON = {
    "春": ["春", "东风", "桃", "李", "杏", "莺", "燕", "柳", "芳草", "落花", "新绿"],
    "夏": ["夏", "暑", "蝉", "荷", "莲", "芙蓉", "绿阴", "麦", "梅雨", "南风"],
    "秋": ["秋", "西风", "霜", "雁", "菊", "梧桐", "落叶", "秋月", "桂"],
    "冬": ["冬", "寒", "雪", "冰", "冻", "腊", "梅", "炉", "北风", "岁暮"],
    "节令": ["元日", "元宵", "上元", "寒食", "清明", "端午", "七夕", "中秋", "重阳", "除夕"],
}

FESTIVAL = {
    "春节元日": ["元日", "元旦", "新年", "爆竹", "屠苏", "桃符"],
    "元宵上元": ["元宵", "上元", "灯", "火树", "银花"],
    "寒食清明": ["寒食", "清明", "禁火", "扫墓", "梨花"],
    "端午": ["端午", "重午", "角黍", "粽", "龙舟", "艾"],
    "七夕": ["七夕", "乞巧", "牵牛", "织女", "河汉", "鹊桥"],
    "中秋": ["中秋", "团圆", "桂", "蟾", "玉兔", "嫦娥"],
    "重阳": ["重阳", "登高", "茱萸", "菊酒", "九日"],
    "除夕岁暮": ["除夕", "岁除", "守岁", "岁暮", "残年", "腊尽"],
}

EMOTION = {
    "离愁别绪": ["别", "离", "泪", "愁", "断肠", "惆怅", "长亭", "孤帆"],
    "思乡怀人": ["乡", "故园", "故人", "相思", "怀", "梦", "雁", "归心"],
    "悲慨忧愤": ["悲", "哀", "恨", "愤", "怨", "泣", "萧条", "苍凉"],
    "闲适淡泊": ["闲", "静", "淡", "卧", "渔", "樵", "归隐", "无事"],
    "豪迈昂扬": ["豪", "壮", "雄", "长啸", "万里", "剑", "酒", "丈夫"],
    "爱慕怨情": ["爱", "怜", "妾", "郎", "闺", "相思", "鸳鸯", "红豆"],
    "忧国伤时": ["国", "乱", "兵", "社稷", "苍生", "中原", "忧"],
    "喜悦赞美": ["喜", "笑", "乐", "欢", "佳", "新晴", "瑞", "赞"],
    "孤寂清冷": ["孤", "独", "寂", "冷", "寒", "空", "残灯", "夜半", "无人"],
}

STYLE = {
    "雄浑豪放": ["雄", "壮", "豪", "沧海", "万里", "长河", "剑", "高歌"],
    "冲淡闲远": ["淡", "闲", "远", "白云", "空山", "渔樵", "悠然"],
    "清新自然": ["清", "新", "花", "鸟", "泉", "雨", "竹", "明月"],
    "绮丽纤秾": ["绮", "丽", "红", "翠", "锦", "罗", "香", "艳", "珠", "玉"],
    "沉郁悲慨": ["沉", "郁", "悲", "慨", "泪", "恨", "苍凉", "风雨"],
    "高古典雅": ["古", "雅", "高", "典", "琴", "鼎", "礼", "清庙"],
    "含蓄委曲": ["含", "不尽", "欲言", "深意", "暗", "微", "曲"],
    "飘逸旷达": ["飘", "逸", "仙", "鹤", "云", "蓬莱", "旷", "达"],
    "劲健洗炼": ["劲", "健", "铁", "瘦", "峭", "简", "骨", "锋"],
}

CI_STYLE = {
    "婉约清丽": ["花", "月", "帘", "香", "泪", "相思", "闺", "柔", "小楼"],
    "豪放慷慨": ["江山", "万里", "剑", "酒", "风雷", "胡虏", "壮怀", "沙场"],
    "清空骚雅": ["清", "空", "烟水", "梅", "竹", "雅", "淡", "白石"],
    "沉郁悲慨": ["故国", "兵", "尘", "恨", "泪", "飘零", "中原", "南渡"],
    "典雅工丽": ["锦", "玉", "珠", "罗", "宫", "调", "华", "绮"],
    "自然疏淡": ["闲", "野", "溪", "渔", "樵", "归去", "淡", "草堂"],
    "俚俗谐趣": ["笑", "戏", "村", "俗", "俚", "儿童", "醉倒"],
}


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def simp(text):
    return cc.convert(text or "")


def only_ch(text):
    return "".join(re.findall(r"[\u4e00-\u9fff]", text or ""))


def split_line(text):
    parts = re.split(r"[，。！？；、\n\r\s]+", text or "")
    return [only_ch(x) for x in parts if only_ch(x)]


def score_label(text, table, low=1.0):
    best = "N/A"
    best_score = low
    for name, words in table.items():
        s = 0.0
        for w in words:
            c = text.count(w)
            if c:
                if len(w) == 1:
                    s += c * 0.6
                else:
                    s += c * (1.3 + 0.2 * len(w))
        if s > best_score:
            best = name
            best_score = s
    return best


def guess_form(x):
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
        return "N/A"
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


def get_last(x):
    lines = split_line(x["content"])
    if not lines:
        return ""
    return lines[-1][-1]


def rhyme_label(ch):
    if not ch:
        return "N/A"
    py = lazy_pinyin(ch, style=Style.FINALS_TONE3)
    if not py:
        return "N/A"
    return py[0]


def tone_label(ch, tone_data):
    if not ch:
        return "N/A"
    if ch in tone_data:
        return tone_data[ch]["tone"]
    py = lazy_pinyin(ch, style=Style.TONE3)
    if not py:
        return "N/A"
    if py[0][-1:] in ["1", "2"]:
        return "平"
    if py[0][-1:] in ["3", "4"]:
        return "仄"
    return "N/A"


def label_one(x, tone_data):
    text = simp(x.get("title", "") + " " + x.get("author", "") + " " + x.get("content", ""))
    last = get_last(x)
    item = {
        "id": x.get("id", ""),
        "kind": x.get("kind", ""),
        "title": x.get("title", ""),
        "author": x.get("author", ""),
        "theme": score_label(text, THEME),
        "season": score_label(text, SEASON),
        "festival": score_label(text, FESTIVAL),
        "emotion": score_label(text, EMOTION),
        "style": score_label(text, STYLE),
        "ci_style": "N/A",
        "form": guess_form(x),
        "rhyme": rhyme_label(last),
        "tone": tone_label(last, tone_data),
    }
    if x.get("kind") == "song_ci":
        item["ci_style"] = score_label(text, CI_STYLE)
        if item["theme"] == "N/A":
            item["theme"] = score_label(text + " " + x.get("cipai", ""), THEME)
    return item


def load_tone():
    if os.path.exists(TONE_FILE):
        return json.load(open(TONE_FILE, "r", encoding="utf-8"))
    return {}


def label_file(name, tone_data, limit):
    in_path = os.path.join(POEM_DIR, name + ".jsonl")
    out_path = os.path.join(LABEL_DIR, name + "_labels.jsonl")
    mkdir(LABEL_DIR)
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            x = json.loads(line)
            y = label_one(x, tone_data)
            fout.write(json.dumps(y, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    print(name, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tone_data = load_tone()
    for name in ["train", "vaid", "test"]:
        label_file(name, tone_data, args.limit)


if __name__ == "__main__":
    main()
