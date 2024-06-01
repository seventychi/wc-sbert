from datasets import load_dataset, load_metric
from sentence_transformers import (
    SentenceTransformer,
    util
)


def inference(eval_model_name, queries, labels, top_k=1):
    model = SentenceTransformer(eval_model_name)
    label_embeddings = model.encode(labels)
    query_embeddings = model.encode(queries)

    preds = util.semantic_search(query_embeddings, label_embeddings, top_k=top_k)
    evals = []

    for pred in preds:
        evals.append(pred[0]["corpus_id"])

    return evals


def eval_agnews(eval_model_name):
    labels = ["World", "Sports", "Business", "Science", "Technology"]

    statistics = {}

    for x_idx, x_label in enumerate(labels):
        statistics[x_idx] = {}
        for y_idx, y_label in enumerate(labels):
            statistics[x_idx][y_idx] = 0

    test_set = load_dataset("ag_news")["test"]
    refs = []
    queries = []

    for data in test_set:
        refs.append(data["label"])
        queries.append(data["text"])

    evals = inference(
        eval_model_name=eval_model_name,
        queries=queries,
        labels=labels)

    for index, value in enumerate(evals):
        if value == 4:
            evals[index] = 3

    for index, ref in enumerate(refs):
        statistics[ref][evals[index]] += 1

    accuracy_metric = load_metric("accuracy")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")

    accuracy = accuracy_metric.compute(predictions=evals, references=refs)
    precision = precision_metric.compute(predictions=evals, references=refs, average='macro')
    recall = recall_metric.compute(predictions=evals, references=refs, average='macro')
    f1 = f1_metric.compute(predictions=evals, references=refs, average='macro')

    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1']
    }


def main():
    print("agnews", "baseline", eval_agnews(eval_model_name="sentence-transformers/all-mpnet-base-v2"))
    print("agnews", "pre-trained with wiki", eval_agnews(eval_model_name="../checkpoints/all-mpnet-base-v2"))
    print("agnews, self-training", eval_agnews(eval_model_name="../checkpoints/agnews"))


if __name__ == "__main__":
    main()
