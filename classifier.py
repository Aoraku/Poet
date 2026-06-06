import argparse
import os
import time

from scipy.sparse import hstack
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import config


RANDOM_SEED = 42
REPORT_FILE = os.path.join(config.BASE_DIR, "report.md")


def text_of(x):
    return x.get("title", "") + " " + x.get("author", "") + " " + x.get("content", "")


def get_xy(split, standard, limit, kind):
    poems, labels = config.load_split(split, limit=0)
    data = []
    y = []
    for x, lab in zip(poems, labels):
        if kind != "all" and x.get("kind") != kind:
            continue
        if standard == "ci_style" and x.get("kind") != "song_ci":
            continue
        yy = lab.get(standard, "N/A")
        if yy == "N/A":
            continue
        data.append(x)
        y.append(yy)
        if limit and len(data) >= limit:
            break
    return data, y


def build_x(train, test, use_w2v=False):
    tone_data = config.load_json(config.TONE_FILE)
    w2v = config.load_w2v() if use_w2v else None
    train_feat = [config.build_feat(x, tone_data, w2v) for x in train]
    test_feat = [config.build_feat(x, tone_data, w2v) for x in test]

    vec = DictVectorizer()
    x1 = vec.fit_transform(train_feat)
    x1_test = vec.transform(test_feat)

    tfidf = TfidfVectorizer(analyzer="char", max_features=1200)
    x2 = tfidf.fit_transform([text_of(x) for x in train])
    x2_test = tfidf.transform([text_of(x) for x in test])
    return hstack([x1, x2]), hstack([x1_test, x2_test])


def build_knn():
    return KNeighborsClassifier(n_neighbors=7, metric="cosine")


def build_wknn():
    return KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine")


def build_bayes():
    return MultinomialNB(alpha=0.5)


def build_svm():
    return LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_SEED)


def build_tree():
    return DecisionTreeClassifier(max_depth=30, min_samples_leaf=5, random_state=RANDOM_SEED)


def build_models():
    base = {
        "KNN": build_knn(),
        "WKNN": build_wknn(),
        "Bayes": build_bayes(),
        "SVM": build_svm(),
        "Tree": build_tree(),
    }
    models = dict(base)

    for name, model in base.items():
        models["Bagging_" + name] = BaggingClassifier(
            estimator=model,
            n_estimators=8,
            random_state=RANDOM_SEED,
        )

    models["AdaBoost_Bayes"] = AdaBoostClassifier(
        estimator=build_bayes(),
        n_estimators=12,
        algorithm="SAMME",
        random_state=RANDOM_SEED,
    )
    models["AdaBoost_SVM"] = AdaBoostClassifier(
        estimator=build_svm(),
        n_estimators=8,
        algorithm="SAMME",
        random_state=RANDOM_SEED,
    )
    models["AdaBoost_Tree"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
        n_estimators=20,
        algorithm="SAMME",
        random_state=RANDOM_SEED,
    )
    return models


def eval_one(y, pred):
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, average="macro", zero_division=0)
    recall = recall_score(y, pred, average="macro", zero_division=0)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    return acc, precision, recall, f1


def run_standard(standard, train_limit, test_limit, kind, use_w2v):
    if standard == "ci_style":
        kind = "song_ci"
    train, y_train = get_xy("train", standard, train_limit, kind)
    test, y_test = get_xy("test", standard, test_limit, kind)
    x_train, x_test = build_x(train, test, use_w2v)

    result = []
    models = build_models()
    for name, model in models.items():
        start = time.time()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc, precision, recall, f1 = eval_one(y_test, pred)
        cost = time.time() - start
        print(standard, name, round(acc, 4), round(f1, 4), f"{cost:.1f}s")
        result.append((name, acc, precision, recall, f1, cost))
    return result, len(train), len(test)


def write_report(all_result, train_limit, test_limit, use_w2v):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# 分类实验报告\n\n")
        f.write("true label 使用 label.py 生成的 LLM/人工规则辅助标签。\n\n")
        f.write(f"train_limit = {train_limit}, test_limit = {test_limit}\n\n")
        f.write(f"use_w2v = {use_w2v}\n\n")
        f.write("AdaBoost 没有套 KNN/WKNN，因为 KNN 不支持 AdaBoost 需要的样本权重更新。\n\n")

        for standard, info in all_result.items():
            rows, n_train, n_test = info
            f.write(f"## {standard}\n\n")
            f.write(f"train = {n_train}, test = {n_test}\n\n")
            f.write("| model | acc | precision | recall | f1 | time |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            for name, acc, precision, recall, f1, cost in rows:
                f.write(f"| {name} | {acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {cost:.1f}s |\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", default="all")
    parser.add_argument("--train_limit", type=int, default=3000)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--kind", default="all")
    parser.add_argument("--use_w2v", action="store_true")
    args = parser.parse_args()

    standards = [args.standard]
    if args.standard == "all":
        standards = ["theme", "season", "emotion", "style", "ci_style"]

    all_result = {}
    for standard in standards:
        rows, n_train, n_test = run_standard(standard, args.train_limit, args.test_limit, args.kind, args.use_w2v)
        all_result[standard] = (rows, n_train, n_test)
    write_report(all_result, args.train_limit, args.test_limit, args.use_w2v)


if __name__ == "__main__":
    main()
