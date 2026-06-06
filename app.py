import json
import os
from collections import Counter
from types import SimpleNamespace

import streamlit as st

import classifier
import config
import generator


st.set_page_config(page_title="Poet", layout="wide")

MODEL_NAMES = [
    "best", "KNN", "WKNN", "Bayes", "SVM", "Tree",
    "Bagging_KNN", "Bagging_WKNN", "Bagging_Bayes", "Bagging_SVM", "Bagging_Tree",
    "AdaBoost_Bayes", "AdaBoost_SVM", "AdaBoost_Tree",
]


@st.cache_data
def load_poems(split="train", limit=5000):
    return config.load_jsonl(os.path.join(config.POEM_DIR, split + ".jsonl"), limit)


@st.cache_data
def load_pair(split="train", kind="poem", limit=5000):
    if not has_llm():
        return [], []
    return config.load_labeled(split, kind, config.LLM_DIR, limit)


def has_llm():
    return os.path.exists(config.label_path("train", config.LLM_DIR))


def kind_val(name):
    if name == "词":
        return "ci"
    return "poem"


def same_score(a, b):
    aa = set(config.only_ch(a))
    bb = set(config.only_ch(b))
    if not aa or not bb:
        return 0
    return len(aa & bb) / len(aa | bb)


def find_sim(text, kind, top=5):
    poems = load_poems("train", 5000)
    arr = []
    for x in poems:
        if not config.same_kind(x, kind):
            continue
        s = same_score(text, x.get("content", ""))
        arr.append((s, x))
    arr.sort(key=lambda x: x[0], reverse=True)
    return arr[:top]


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


def label_opts(kind, field):
    if field == "form" and kind == "ci":
        return config.CI_FORMS
    if field == "form":
        return config.POEM_FORMS
    if field in config.CATEGORY:
        return list(config.CATEGORY[field].keys()) + ["N/A"]
    return ["N/A"]


def show_book(kind, field, val):
    poems, labs = load_pair("train", kind, 5000)
    rows = []
    for x, lab in zip(poems, labs):
        if val and lab.get(field, "") != val:
            continue
        rows.append({
            "title": x.get("title", ""),
            "author": x.get("author", ""),
            "cipai": x.get("cipai", ""),
            "theme": lab.get("theme", ""),
            "emotion": lab.get("emotion", ""),
            "style": lab.get("style", ""),
            "ci_style": lab.get("ci_style", ""),
            "form": lab.get("form", ""),
            "content": x.get("content", "")[:120],
        })
        if len(rows) >= 200:
            break
    st.dataframe(rows, use_container_width=True)


def show_count(kind, field):
    poems, labs = load_pair("train", kind, 5000)
    cnt = Counter()
    for x, lab in zip(poems, labs):
        cnt[lab.get(field, "N/A")] += 1
    rows = [{"label": k, "count": v} for k, v in cnt.most_common(30)]
    st.dataframe(rows, use_container_width=True)


def show_cls():
    for use_w2v in [False, True]:
        path = classifier.report_file(use_w2v)
        name = "word2vec" if use_w2v else "base"
        st.write(name)
        if os.path.exists(path):
            st.markdown(open(path, "r", encoding="utf-8").read())
        else:
            st.write("还没有分类报告")


def show_cluster(kind):
    folder = os.path.join(config.RESULT_DIR, "clusters")
    rows = []
    if os.path.exists(folder):
        for name in os.listdir(folder):
            if not name.startswith(kind + "_") or not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            x = json.load(open(path, "r", encoding="utf-8"))
            rows.append({
                "kind": x.get("kind", ""),
                "standard": x.get("standard", ""),
                "method": x.get("method", ""),
                "acc": round(x.get("acc", 0), 4),
                "f1": round(x.get("f1", 0), 4),
                "silhouette": round(x.get("silhouette", 0), 4),
                "labels": "、".join(x.get("cluster_names", [])),
            })
    st.dataframe(rows, use_container_width=True)


st.title("Poet")

tab1, tab2, tab3, tab4 = st.tabs(["分类", "诗词大全", "生成", "报告"])

with tab1:
    kind_name = st.radio("体裁", ["诗", "词"], horizontal=True)
    kind = kind_val(kind_name)
    cipai = ""
    if kind == "ci":
        cipai = st.text_input("词牌", "浣溪沙")
    model_name = st.selectbox("分类器", MODEL_NAMES)
    standards = st.multiselect("分类标准", config.kind_std(kind), default=config.kind_std(kind))
    text = st.text_area("文本", height=180)
    if st.button("分类"):
        if not has_llm():
            st.warning("缺少 dataset/llm_labels，不能用分类器预测。先逐条完成 LLM 真实标签。")
        else:
            ans = classifier.pred_text(kind, text, cipai, standards, model_name)
            rows = []
            for k, v in ans.items():
                rows.append({"standard": k, "model": v["model"], "label": v["label"]})
            st.dataframe(rows, use_container_width=True)
            st.write("相似作品")
            sim = []
            for s, x in find_sim(text, kind):
                sim.append({
                    "score": round(s, 4),
                    "title": x.get("title", ""),
                    "author": x.get("author", ""),
                    "cipai": x.get("cipai", ""),
                    "content": x.get("content", "")[:100],
                })
            st.dataframe(sim, use_container_width=True)

with tab2:
    kind_name = st.radio("查看体裁", ["诗", "词"], horizontal=True)
    kind = kind_val(kind_name)
    if not has_llm():
        st.warning("缺少 dataset/llm_labels，暂时不能按 LLM 标签筛选。")
    else:
        field = st.selectbox("分类标准", config.kind_std(kind))
        val = st.selectbox("标签", [""] + label_opts(kind, field))
        show_count(kind, field)
        show_book(kind, field, val)

with tab3:
    kind_name = st.radio("生成体裁", ["诗", "词"], horizontal=True)
    kind = kind_val(kind_name)
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

with tab4:
    kind_name = st.radio("报告体裁", ["诗", "词"], horizontal=True)
    kind = kind_val(kind_name)
    st.write("分类算法")
    show_cls()
    st.divider()
    st.write("聚类算法")
    show_cluster(kind)
