import evaluate
import json
import os
import pickle
from collections import OrderedDict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_categories():
    categories = []

    with open("../data/wiki_categories.json", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="categories"):
            data = json.loads(line)
            cate = data["categories"]

            if len(cate) == 1 and len(cate[0]) == 0:
                continue
            categories.extend(json.loads(line)["categories"])

    return list(OrderedDict.fromkeys(categories))


def get_embeddings(model, categories=None):
    file = "../embeddings/all-mpnet-base-v2/wiki_category_embeddings.pkl"

    if os.path.exists(file):
        with open(file, "rb") as f:
            return pickle.load(f)

    embeddings = model.encode(categories, convert_to_tensor=True)

    with open(file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def eval_agnews(model, agnews, categories, label_embeddings, category_embeddings):
    refs = []
    evals = []
    evals_2step = []

    for data in tqdm(agnews, total=len(agnews), desc="agnews"):
        query = data["text"]
        query_embedding = model.encode(query, convert_to_tensor=True)

        refs.append(data["label"])

        # region 直接取得 query 最接近的 label 作為 pred 結果
        index = util.semantic_search(
            query_embedding,
            label_embeddings,
            top_k=1)[0][0]["corpus_id"]

        if index == 4:
            index = 3

        evals.append(index)
        # endregion

        # region 兩階段取 label, step 1: 取得 query 最接近的 wiki category, step 2: 藉由此 category 取得 label
        index = util.semantic_search(
            model.encode(
                categories[util.semantic_search(
                    query_embedding,
                    category_embeddings,
                    top_k=1)[0][0]["corpus_id"]],
                convert_to_tensor=True),
            label_embeddings,
            top_k=1)[0][0]["corpus_id"]

        if index == 4:
            index = 3

        evals_2step.append(index)
        # endregion

    accuracy = evaluate.load("accuracy")

    print("1 step accu.:", accuracy.compute(predictions=evals, references=refs))
    print("2 step accu.:", accuracy.compute(predictions=evals_2step, references=refs))


def main():
    model = SentenceTransformer("../checkpoints/all-mpnet-base-v2", device="cuda")
    agnews = load_dataset("ag_news")["test"]

    labels = ["This topic is talk about World",
              "This topic is talk about Sports",
              "This topic is talk about Business",
              "This topic is talk about Science",
              "This topic is talk about Technology"]
    label_embeddings = model.encode(labels, convert_to_tensor=True)

    categories = get_categories()
    category_embeddings = get_embeddings(model)

    eval_agnews(model, agnews, categories, label_embeddings, category_embeddings)


if __name__ == "__main__":
    main()
