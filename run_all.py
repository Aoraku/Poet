import argparse
import os
import subprocess

import config


def need_num(split, kind):
    path = os.path.join(config.POEM_DIR, split + ".jsonl")
    n = 0
    for x in config.load_jsonl(path):
        if config.same_kind(x, kind):
            n += 1
    return n


def have_num(split, kind):
    path = config.label_path(split, config.LLM_DIR)
    if not os.path.exists(path):
        return 0
    data, labs = config.load_labeled(split, kind, config.LLM_DIR)
    ok = 0
    for x in labs:
        miss = False
        for key in config.kind_std(kind):
            if x.get(key, "") == "":
                miss = True
        if not miss:
            ok += 1
    return ok


def check_all():
    ok = True
    for split in ["train", "vaid", "test"]:
        for kind in ["poem", "ci"]:
            need = need_num(split, kind)
            have = have_num(split, kind)
            print(split, kind, "need", need, "have", have)
            if have < need:
                ok = False
    return ok


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_limit", type=int, default=1200)
    parser.add_argument("--no_w2v", action="store_true")
    args = parser.parse_args()

    if not check_all():
        print("LLM 标签还没齐，不跑分类和聚类。")
        return

    run(["python", "classifier.py", "--kind", "all", "--standard", "all", "--model", "all", "--train_limit", "0", "--test_limit", "0", "--save"])
    if not args.no_w2v:
        run(["python", "classifier.py", "--kind", "all", "--standard", "all", "--model", "all", "--train_limit", "0", "--test_limit", "0", "--save", "--use_w2v"])

    for kind in ["poem", "ci"]:
        run(["python", "cluster.py", "--kind", kind, "--standard", "all", "--method", "all", "--limit", str(args.cluster_limit)])
        if not args.no_w2v:
            run(["python", "cluster.py", "--kind", kind, "--standard", "all", "--method", "all", "--limit", str(args.cluster_limit), "--use_w2v"])


if __name__ == "__main__":
    main()
