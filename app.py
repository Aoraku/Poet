import json
import os
from types import SimpleNamespace
from collections import Counter

import streamlit as st

import config
import generator
from label import label_one


st.set_page_config(page_title="Poet", layout="wide")


@st.cache_data
def load_tone():
    return config.load_json(config.TONE_FILE)


@st.cache_data
def load_some(limit=3000):
    poems, labels = config.load_split("train", limit=limit)
    return poems, labels


def same_score(a, b):
    aa = set(config.only_ch(a))
    bb = set(config.only_ch(b))
    if not aa or not bb:
        return 0
    return len(aa & bb) / len(aa | bb)


def find_sim(text, poems, top=5):
    arr = []
    for x in poems:
        s = same_score(text, x.get("content", ""))
        arr.append((s, x))
    arr.sort(key=lambda x: x[0], reverse=True)
    return arr[:top]


def label_text(text, cipai=""):
    kind = "song_ci" if cipai else "tang_poem"
    x = {
        "id": "",
        "kind": kind,
        "title": "输入",
        "author": "",
        "cipai": cipai,
        "content": text,
    }
    return label_one(x, load_tone())


def show_counts(labels, field):
    cnt = Counter()
    for x in labels:
        cnt[x.get(field, "N/A")] += 1
    data = [{"label": k, "count": v} for k, v in cnt.most_common(20)]
    st.dataframe(data, use_container_width=True)


def show_cluster():
    path = os.path.join(config.RESULT_DIR, "clusters", "summary.md")
    if os.path.exists(path):
        st.markdown(open(path, "r", encoding="utf-8").read())
    else:
        st.write("N/A")


def make_args(kind, form, theme, emotion, style, ci_style, cipai, rhyme, word, limit):
    return SimpleNamespace(
        kind=kind,
        form=form,
        theme=theme,
        emotion=emotion,
        style=style,
        ci_style=ci_style,
        cipai=cipai,
        rhyme=rhyme,
        word=word,
        limit=limit,
    )


def gen_text(args):
    data = generator.load_data(args)
    model = generator.train_model(data)
    need = generator.get_need(args)
    if args.kind == "ci":
        return generator.gen_ci(model, args, need)
    return generator.gen_poem(model, args, need)


st.title("Poet")

tab1, tab2, tab3 = st.tabs(["分类", "已有结果", "生成"])

with tab1:
    poems, labels = load_some()
    cipai = st.text_input("词牌", "")
    text = st.text_area("文本", height=180)
    if st.button("分类"):
        lab = label_text(text, cipai)
        st.json(lab)
        st.write("相似作品")
        rows = []
        for s, x in find_sim(text, poems):
            rows.append({
                "score": round(s, 4),
                "title": x.get("title", ""),
                "author": x.get("author", ""),
                "kind": x.get("kind", ""),
                "content": x.get("content", "")[:80],
            })
        st.dataframe(rows, use_container_width=True)

with tab2:
    poems, labels = load_some()
    field = st.selectbox("标签", ["theme", "season", "festival", "emotion", "style", "ci_style", "form"])
    show_counts(labels, field)
    st.divider()
    show_cluster()

with tab3:
    kind_name = st.radio("体裁", ["诗", "词"], horizontal=True)
    kind = "ci" if kind_name == "词" else "poem"
    form = st.selectbox("诗体", ["七言绝句", "五言绝句", "七言律诗", "五言律诗"])
    cipai = st.text_input("词牌 ", "浣溪沙" if kind == "ci" else "")
    theme = st.selectbox("题材", [""] + list(config.THEME.keys()))
    emotion = st.selectbox("情感", [""] + list(config.EMOTION.keys()))
    style = st.selectbox("诗风", [""] + list(config.STYLE.keys()))
    ci_style = st.selectbox("词风", [""] + list(config.CI_STYLE.keys()))
    rhyme = st.text_input("韵脚", "")
    word = st.text_input("关键词", "")
    limit = st.number_input("语料数", min_value=1000, max_value=30000, value=8000, step=1000)
    if st.button("生成"):
        args = make_args(kind, form, theme, emotion, style, ci_style, cipai, rhyme, word, int(limit))
        text = gen_text(args)
        st.text(text)
