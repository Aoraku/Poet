import argparse
import os

from gensim.models import Word2Vec

import config


def get_sents(limit):
    sents = []
    for split in ["train", "vaid", "test"]:
        path = os.path.join(config.POEM_DIR, split + ".jsonl")
        for x in config.load_jsonl(path):
            for line in config.split_line(x.get("content", "")):
                chars = list(config.only_ch(line))
                if len(chars) >= 2:
                    sents.append(chars)
            if limit and len(sents) >= limit:
                return sents
    return sents


def save_report(model):
    path = os.path.join(config.RESULT_DIR, "embedding_report.md")
    config.mkdir(config.RESULT_DIR)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Word2Vec\n\n")
        for ch in ["月", "山", "水", "花", "酒", "愁"]:
            if ch not in model.wv:
                continue
            sims = model.wv.most_similar(ch, topn=10)
            line = ch + ": " + ", ".join([a + "(" + f"{b:.2f}" + ")" for a, b in sims])
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120000)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    sents = get_sents(args.limit)
    print("sent", len(sents))
    model = Word2Vec(
        sentences=sents,
        vector_size=args.size,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=args.epochs,
        seed=config.RANDOM_SEED,
    )
    config.mkdir(config.MODEL_DIR)
    model.save(config.W2V_FILE)
    save_report(model)
    print(config.W2V_FILE)


if __name__ == "__main__":
    main()
