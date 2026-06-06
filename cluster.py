import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import kmedoids
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


def text_of(x):
    return x.get("title", "") + " " + x.get("author", "") + " " + x.get("content", "")


def k_num(standard):
    if standard == "form":
        return 10
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
    if key in ["num_char", "num_line", "avg_len", "five_ratio", "seven_ratio"]:
        return True
    if standard == "ci_style" and key.startswith("cipai_"):
        return True
    return False


def load_data(split, standard, kind, limit):
    poems, labels = config.load_split(split, limit=0)
    data = []
    y = []
    for x, lab in zip(poems, labels):
        if kind != "all" and x.get("kind") != kind:
            continue
        if standard == "ci_style" and x.get("kind") != "song_ci":
            continue
        data.append(x)
        y.append(lab.get(standard, "N/A"))
        if limit and len(data) >= limit:
            break
    return data, y


def build_x(data, standard):
    tone_data = config.load_json(config.TONE_FILE)
    feat_list = []
    for x in data:
        raw = config.build_feat(x, tone_data)
        feat = {}
        for k, v in raw.items():
            if need_key(k, standard):
                feat[k] = v
        feat_list.append(feat)

    vec = DictVectorizer()
    x1 = vec.fit_transform(feat_list)
    text = [text_of(x) for x in data]
    tfidf = TfidfVectorizer(analyzer="char", max_features=300)
    x2 = tfidf.fit_transform(text)
    return hstack([x1, x2]), vec, tfidf


def run_kmeans(x, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    return model.fit_predict(x)


def run_bisect(x, k):
    model = BisectingKMeans(n_clusters=k, random_state=RANDOM_SEED)
    return model.fit_predict(x)


def run_agglom(x, k):
    model = AgglomerativeClustering(n_clusters=k, linkage="average")
    return model.fit_predict(x.toarray())


def run_medoids(x, k, pam=False):
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
            if lab == i and yy != "N/A":
                cnt[yy] += 1
        if cnt:
            names.append(cnt.most_common(1)[0][0])
        else:
            names.append("N/A")
    return names


def pred_label(labels, names):
    return [names[x] for x in labels]


def eval_cluster(y, pred):
    y2 = []
    p2 = []
    for a, b in zip(y, pred):
        if a == "N/A":
            continue
        y2.append(a)
        p2.append(b)
    if not y2:
        return 0, 0
    acc = accuracy_score(y2, p2)
    f1 = f1_score(y2, p2, average="macro", zero_division=0)
    return acc, f1


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


def run_one(standard, method, split, kind, limit):
    data, y = load_data(split, standard, kind, limit)
    x, vec, tfidf = build_x(data, standard)
    k = k_num(standard)
    if k > len(data):
        k = len(data)

    if method == "kmeans":
        labels = run_kmeans(x, k)
    elif method == "kmedoids":
        labels = run_medoids(x, k, pam=False)
    elif method == "pam":
        labels = run_medoids(x, k, pam=True)
    elif method == "agglomerative":
        labels = run_agglom(x, k)
    else:
        labels = run_bisect(x, k)

    names = name_cluster(labels, y, k)
    pred = pred_label(labels, names)
    acc, f1 = eval_cluster(y, pred)
    sil = silhouette_score(x, labels, sample_size=min(800, len(data)), random_state=RANDOM_SEED)

    out = {
        "standard": standard,
        "method": method,
        "split": split,
        "kind": kind,
        "limit": len(data),
        "k": k,
        "acc": acc,
        "f1": f1,
        "silhouette": sil,
        "cluster_names": names,
        "clusters": top_words(data, labels, k),
    }

    mkdir(RESULT_DIR)
    path = os.path.join(RESULT_DIR, standard + "_" + method + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(standard, method, "acc", round(acc, 4), "f1", round(f1, 4), "sil", round(sil, 4))
    return out


def write_summary(results):
    path = os.path.join(RESULT_DIR, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 聚类结果\n\n")
        for x in results:
            f.write(f"## {x['standard']} / {x['method']}\n\n")
            f.write(f"- acc: {x['acc']:.4f}\n")
            f.write(f"- f1: {x['f1']:.4f}\n")
            f.write(f"- silhouette: {x['silhouette']:.4f}\n")
            f.write(f"- labels: {', '.join(x['cluster_names'])}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", default="theme")
    parser.add_argument("--method", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--kind", default="all")
    parser.add_argument("--limit", type=int, default=1200)
    args = parser.parse_args()

    standards = [args.standard]
    if args.standard == "all":
        standards = ["theme", "season", "emotion", "style", "ci_style"]

    methods = [args.method]
    if args.method == "all":
        methods = ["kmeans", "kmedoids", "pam", "agglomerative", "bisect"]

    results = []
    for standard in standards:
        for method in methods:
            res = run_one(standard, method, args.split, args.kind, args.limit)
            results.append(res)
    write_summary(results)


if __name__ == "__main__":
    main()
