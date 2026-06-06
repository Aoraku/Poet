import argparse
import json
import os
import random
from collections import Counter

import kmedoids
import numpy as np
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances, silhouette_score

import config


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RESULT_DIR = os.path.join(config.RESULT_DIR, "clusters")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def label_need():
    path = config.label_path("train", config.LLM_DIR)
    if not os.path.exists(path):
        print("缺少 LLM 真实标签:", path)
        print("聚类评估必须和 dataset/llm_labels 比，不能用规则标签。")
        raise SystemExit(1)


def text_of(x):
    return x.get("title", "") + " " + x.get("author", "") + " " + x.get("cipai", "") + " " + x.get("content", "")


def k_num(kind, standard):
    if standard == "form":
        if kind == "ci":
            return len(config.CI_FORMS)
        return len(config.POEM_FORMS)
    if standard == "ci_style":
        return len(config.CATEGORY["ci_style"])
    if standard in config.CATEGORY:
        return len(config.CATEGORY[standard])
    return 8


def need_key(key, standard):
    if key.startswith(standard + "_"):
        return True
    if key.startswith("form_") or key.startswith("rhyme_") or key.startswith("tone_"):
        return True
    if key.startswith("author_"):
        return True
    if key.startswith("cipai_"):
        return True
    if key in ["num_char", "num_line", "avg_len", "five_ratio", "seven_ratio", "cipai_len", "cipai_line"]:
        return True
    return False


def load_data(split, kind, standard, limit):
    poems, labels = config.load_labeled(split, kind, config.LLM_DIR, 0)
    data = []
    y = []
    for x, lab in zip(poems, labels):
        yy = lab.get(standard, "")
        if yy == "" or yy == "N/A":
            continue
        data.append(x)
        y.append(yy)
        if limit and len(data) >= limit:
            break
    return data, y


def build_x(data, standard, use_w2v):
    tone_data = config.load_json(config.TONE_FILE)
    w2v = config.load_w2v() if use_w2v else None
    feat_list = []
    for x in data:
        raw = config.build_feat(x, tone_data, w2v)
        feat = {}
        for k, v in raw.items():
            if need_key(k, standard):
                feat[k] = v
        feat_list.append(feat)
    vec = DictVectorizer()
    x1 = vec.fit_transform(feat_list)
    tfidf = TfidfVectorizer(analyzer="char", max_features=300)
    x2 = tfidf.fit_transform([text_of(x) for x in data])
    return hstack([x1, x2])


def run_kmeans(x, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    return model.fit_predict(x)


def run_bisect(x, k):
    model = BisectingKMeans(n_clusters=k, random_state=RANDOM_SEED)
    return model.fit_predict(x)


def run_agglom(x, k):
    model = AgglomerativeClustering(n_clusters=k, linkage="average")
    return model.fit_predict(x.toarray())


def run_medoids(x, k, pam):
    arr = x.toarray()
    dist = pairwise_distances(arr, metric="euclidean")
    if pam:
        model = kmedoids.KMedoids(k, method="pam", init="build", random_state=RANDOM_SEED)
    else:
        model = kmedoids.KMedoids(k, method="fasterpam", init="random", random_state=RANDOM_SEED)
    model.fit(dist)
    return model.labels_


def name_cluster(labels, y, k):
    names = []
    for i in range(k):
        cnt = Counter()
        for lab, yy in zip(labels, y):
            if lab == i:
                cnt[yy] += 1
        if cnt:
            names.append(cnt.most_common(1)[0][0])
        else:
            names.append("N/A")
    return names


def eval_res(y, labels, names):
    pred = [names[x] for x in labels]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    return acc, f1, pred


def top_words(data, labels, k):
    result = []
    for i in range(k):
        cnt = Counter()
        examples = []
        for x, lab in zip(data, labels):
            if lab != i:
                continue
            for ch in config.only_ch(x.get("content", "")):
                cnt[ch] += 1
            if len(examples) < 3:
                examples.append(x.get("title", ""))
        result.append({
            "cluster": i,
            "top": cnt.most_common(20),
            "examples": examples,
        })
    return result


def run_one(kind, standard, method, split, limit, use_w2v):
    data, y = load_data(split, kind, standard, limit)
    x = build_x(data, standard, use_w2v)
    k = k_num(kind, standard)
    if k > len(data):
        k = len(data)

    if method == "kmeans":
        labels = run_kmeans(x, k)
    elif method == "kmedoids":
        labels = run_medoids(x, k, False)
    elif method == "pam":
        labels = run_medoids(x, k, True)
    elif method == "agglomerative":
        labels = run_agglom(x, k)
    else:
        labels = run_bisect(x, k)

    names = name_cluster(labels, y, k)
    acc, f1, pred = eval_res(y, labels, names)
    sil = silhouette_score(x, labels, sample_size=min(800, len(data)), random_state=RANDOM_SEED)

    out = {
        "kind": kind,
        "standard": standard,
        "method": method,
        "split": split,
        "limit": len(data),
        "use_w2v": use_w2v,
        "k": k,
        "acc": acc,
        "f1": f1,
        "silhouette": sil,
        "cluster_names": names,
        "clusters": top_words(data, labels, k),
    }
    mkdir(RESULT_DIR)
    mark = "_w2v" if use_w2v else ""
    path = os.path.join(RESULT_DIR, kind + "_" + standard + "_" + method + mark + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(kind, standard, method, "acc", round(acc, 4), "f1", round(f1, 4), "sil", round(sil, 4))
    return out


def write_summary(kind, results):
    path = os.path.join(RESULT_DIR, "summary_" + kind + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 聚类结果 " + kind + "\n\n")
        for x in results:
            f.write(f"## {x['standard']} / {x['method']}\n\n")
            f.write(f"- acc: {x['acc']:.4f}\n")
            f.write(f"- f1: {x['f1']:.4f}\n")
            f.write(f"- silhouette: {x['silhouette']:.4f}\n")
            f.write(f"- labels: {', '.join(x['cluster_names'])}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="poem", choices=["poem", "ci"])
    parser.add_argument("--standard", default="theme")
    parser.add_argument("--method", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--use_w2v", action="store_true")
    args = parser.parse_args()

    label_need()
    standards = [args.standard]
    if args.standard == "all":
        standards = config.kind_std(args.kind)

    methods = [args.method]
    if args.method == "all":
        methods = ["kmeans", "kmedoids", "pam", "agglomerative", "bisect"]

    results = []
    for standard in standards:
        for method in methods:
            res = run_one(args.kind, standard, method, args.split, args.limit, args.use_w2v)
            results.append(res)
    write_summary(args.kind, results)


if __name__ == "__main__":
    main()
