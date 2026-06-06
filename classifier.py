import argparse
import os
import pickle
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
CLS_DIR = os.path.join(config.MODEL_DIR, "classifiers")

BEST = {
    "theme": "Tree",
    "season": "Tree",
    "festival": "Tree",
    "emotion": "Tree",
    "style": "Tree",
    "ci_style": "SVM",
    "form": "Tree",
}


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def text_of(x):
    return x.get("title", "") + " " + x.get("author", "") + " " + x.get("cipai", "") + " " + x.get("content", "")


def label_need():
    for split in ["train", "test"]:
        path = config.label_path(split, config.LLM_DIR)
        if not os.path.exists(path):
            print("缺少 LLM 真实标签:", path)
            print("先由 LLM 逐条阅读并填写 dataset/llm_labels。")
            raise SystemExit(1)


def get_xy(split, kind, standard, limit):
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


def make_feat(data, use_w2v):
    tone_data = config.load_json(config.TONE_FILE)
    w2v = config.load_w2v() if use_w2v else None
    feat = []
    text = []
    for x in data:
        feat.append(config.build_feat(x, tone_data, w2v))
        text.append(text_of(x))
    return feat, text


def fit_x(train, use_w2v):
    feat, text = make_feat(train, use_w2v)
    vec = DictVectorizer()
    x1 = vec.fit_transform(feat)
    tfidf = TfidfVectorizer(analyzer="char", max_features=1200)
    x2 = tfidf.fit_transform(text)
    return hstack([x1, x2]), vec, tfidf


def trans_x(data, vec, tfidf, use_w2v):
    feat, text = make_feat(data, use_w2v)
    x1 = vec.transform(feat)
    x2 = tfidf.transform(text)
    return hstack([x1, x2])


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


def base_models():
    return {
        "KNN": build_knn(),
        "WKNN": build_wknn(),
        "Bayes": build_bayes(),
        "SVM": build_svm(),
        "Tree": build_tree(),
    }


def build_models():
    base = base_models()
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


def pick_model(name, standard):
    if name == "best":
        name = BEST.get(standard, "Tree")
    return name


def model_path(kind, standard, model_name, use_w2v):
    mark = "_w2v" if use_w2v else ""
    name = kind + "_" + standard + "_" + model_name + mark + ".pkl"
    return os.path.join(CLS_DIR, name)


def eval_one(y, pred):
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, average="macro", zero_division=0)
    recall = recall_score(y, pred, average="macro", zero_division=0)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    return acc, precision, recall, f1


def train_pack(kind, standard, model_name, train_limit, use_w2v):
    model_name = pick_model(model_name, standard)
    train, y_train = get_xy("train", kind, standard, train_limit)
    x_train, vec, tfidf = fit_x(train, use_w2v)
    model = build_models()[model_name]
    model.fit(x_train, y_train)
    return {
        "kind": kind,
        "standard": standard,
        "model_name": model_name,
        "model": model,
        "vec": vec,
        "tfidf": tfidf,
        "use_w2v": use_w2v,
    }


def save_pack(pack):
    mkdir(CLS_DIR)
    path = model_path(pack["kind"], pack["standard"], pack["model_name"], pack["use_w2v"])
    with open(path, "wb") as f:
        pickle.dump(pack, f)
    return path


def load_pack(kind, standard, model_name, train_limit, use_w2v):
    model_name = pick_model(model_name, standard)
    path = model_path(kind, standard, model_name, use_w2v)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    pack = train_pack(kind, standard, model_name, train_limit, use_w2v)
    save_pack(pack)
    return pack


def run_one(kind, standard, model_name, train_limit, test_limit, use_w2v, save):
    train, y_train = get_xy("train", kind, standard, train_limit)
    test, y_test = get_xy("test", kind, standard, test_limit)
    x_train, vec, tfidf = fit_x(train, use_w2v)
    x_test = trans_x(test, vec, tfidf, use_w2v)
    models = build_models()
    names = list(models.keys()) if model_name == "all" else [pick_model(model_name, standard)]

    rows = []
    for name in names:
        model = models[name]
        start = time.time()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc, precision, recall, f1 = eval_one(y_test, pred)
        cost = time.time() - start
        print(kind, standard, name, round(acc, 4), round(f1, 4))
        rows.append((kind, standard, name, len(train), len(test), acc, precision, recall, f1, cost))
        if save:
            pack = {
                "kind": kind,
                "standard": standard,
                "model_name": name,
                "model": model,
                "vec": vec,
                "tfidf": tfidf,
                "use_w2v": use_w2v,
            }
            save_pack(pack)
    return rows


def write_report(rows, train_limit, test_limit, use_w2v):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# 分类实验报告\n\n")
        f.write("true label 只读取 dataset/llm_labels，由大模型逐条阅读后填写。\n\n")
        f.write(f"train_limit = {train_limit}, test_limit = {test_limit}\n\n")
        f.write(f"use_w2v = {use_w2v}\n\n")
        f.write("| kind | standard | model | train | test | acc | precision | recall | f1 | time |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for kind, standard, name, n_train, n_test, acc, precision, recall, f1, cost in rows:
            f.write(
                f"| {kind} | {standard} | {name} | {n_train} | {n_test} | "
                f"{acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {cost:.1f}s |\n"
            )


def pred_text(kind, text, cipai="", standards=None, model_name="best", train_limit=3000, use_w2v=False):
    label_need()
    kind = config.norm_kind(kind)
    standards = standards or config.kind_std(kind)
    x = config.make_input(kind, text, cipai)
    ans = {}
    for standard in standards:
        pack = load_pack(kind, standard, model_name, train_limit, use_w2v)
        xx = trans_x([x], pack["vec"], pack["tfidf"], pack["use_w2v"])
        ans[standard] = {
            "model": pack["model_name"],
            "label": pack["model"].predict(xx)[0],
        }
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="poem", choices=["poem", "ci"])
    parser.add_argument("--standard", default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--train_limit", type=int, default=3000)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--use_w2v", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--text", default="")
    parser.add_argument("--cipai", default="")
    args = parser.parse_args()

    label_need()
    if args.standard == "all":
        standards = config.kind_std(args.kind)
    else:
        standards = [args.standard]

    if args.predict:
        ans = pred_text(args.kind, args.text, args.cipai, standards, args.model, args.train_limit, args.use_w2v)
        for k, v in ans.items():
            print(k, v["model"], v["label"])
        return

    rows = []
    for standard in standards:
        rows += run_one(
            args.kind,
            standard,
            args.model,
            args.train_limit,
            args.test_limit,
            args.use_w2v,
            args.save,
        )
    write_report(rows, args.train_limit, args.test_limit, args.use_w2v)


if __name__ == "__main__":
    main()
