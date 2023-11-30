from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    util
)
import evaluate


def inference(eval_model_name, queries, labels, top_k=1):
    model = SentenceTransformer(eval_model_name)
    query_embeddings = model.encode(queries)
    label_embeddings = model.encode(labels)

    preds = util.semantic_search(query_embeddings, label_embeddings, top_k=top_k)
    evals = []

    for pred in preds:
        evals.append(pred[0]["corpus_id"])

    return evals


def main():
    labels = ["Company", "Education institution", "Artist", "Athlete", "Office holder",
              "Mean of transportation", "Building", "Nature place", "Village", "Animal",
              "Plant", "Album", "Film", "Written work"]

    labels[0] = "This topic is describing this company"
    labels[1] = "This school, university described in this content is an education institution"
    labels[2] = "This musician, painter, singer, writer, author described in this content is an artist"
    labels[3] = "This person who plays sport described in this content is an athlete"
    labels[4] = "This person who holds a position or office in a government described in this content is an " \
                "officeholder"
    labels[5] = "This vehicles, ridden, trains and other conveyances described in this content is transportation"
    labels[6] = "This man-made structure described in this content is a building"
    labels[7] = "This natural landforms, bodies of water, vegetation, rocks, forests, rivers, lakes, mountains, " \
                "oceans, grasslands described in this content is a natural place"
    labels[8] = "This town, small settlement or community described in this content is a village"
    labels[9] = "This organism described in this content is an animal"
    labels[10] = "This organism described in this content is a plant"
    labels[11] = "This music or recorded tracks described in this content is an album"
    labels[12] = "This movie described in this content is a film"
    labels[13] = "This books, essays, poems or literatures described in this content is a written work"

    test_set = load_dataset("dbpedia_14")["test"]
    refs = []
    queries = []

    for data in test_set:
        refs.append(data["label"])
        queries.append("{} {}".format(data["title"], data["content"]))

    evals = inference(
        eval_model_name='BAAI/bge-large-en-v1.5',
        queries=queries,
        labels=labels)

    accuracy = evaluate.load("accuracy")
    print(accuracy.compute(predictions=evals, references=refs))


if __name__ == "__main__":
    main()
