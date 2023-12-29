import evaluate
from pathlib import Path
from datasets import load_dataset
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
    labels = ["This topic is talk about {}".format(label) for label in labels]
    labels = [f"This article covers topics related to {label}" for label in labels]

    labels[0] += " not Business"
    labels[2] += " not World"

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

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=evals, references=refs)


def eval_yahoo(eval_model_name):
    labels = [
        "Society", "Culture",
        "Science", "Mathematics",
        "Health",
        "Education", "Reference",
        "Computers", "Internet",
        "Sports",
        "Business", "Finance",
        "Entertainment", "Music",
        "Family", "Relationships",
        "Politics", "Government"]
    labels = ["This topic is talk about {}".format(label) for label in labels]
    labels = [f"This article covers topics related to {label}" for label in labels]

    labels[0] += " not Family or Relationships"
    labels[5] += " not Science or Mathematics"

    statistics = {}

    for x_idx, x_label in enumerate(labels):
        statistics[x_idx] = {}
        for y_idx, y_label in enumerate(labels):
            statistics[x_idx][y_idx] = 0

    test_set = load_dataset("yahoo_answers_topics")["test"]
    refs = []
    queries = []

    for data in test_set:
        refs.append(data["topic"])
        queries.append("{} {} {}".format(data["question_title"], data["question_content"], data["best_answer"]))

    evals = inference(
        eval_model_name=eval_model_name,
        queries=queries,
        labels=labels)

    labels_map = {
        0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3,
        7: 4, 8: 4, 9: 5, 10: 6,
        11: 6, 12: 7, 13: 7,
        14: 8, 15: 8, 16: 9, 17: 9
    }

    for index, value in enumerate(evals):
        evals[index] = labels_map[value]

    for index, ref in enumerate(refs):
        statistics[ref][evals[index]] += 1

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=evals, references=refs)


def eval_dbpedia(eval_model_name):
    labels = ["Company", "Education institution", "Artist", "Athlete", "Office holder",
              "Mean of transportation", "Building", "Nature place", "Village", "Animal",
              "Plant", "Album", "Film", "Written work"]

    statistics = {}

    for x_idx, x_label in enumerate(labels):
        statistics[x_idx] = {}
        for y_idx, y_label in enumerate(labels):
            statistics[x_idx][y_idx] = 0

    # labels = [f"This topic is talk about {label}" for label in labels]
    labels = [f"This sentence is belong to {label}" for label in labels]
    labels = [f"This article covers topics related to {label}" for label in labels]

    labels[0] = "This topic is describing this company"
    labels[1] = "This school, university described in this content is an education institution"
    labels[2] = "This musician, painter, singer, writer, author described in this content is an artist"
    labels[3] = "This person who plays sport described in this content is an athlete"
    labels[4] = "This person who holds a position or office in a government described in this content is an officeholder"
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
        eval_model_name=eval_model_name,
        queries=queries,
        labels=labels)

    # for index, ref in enumerate(refs):
    #     statistics[ref][evals[index]] += 1
    #
    # # print confusion matrix
    # for value in statistics.values():
    #     print([str(v).rjust(4) for v in value.values()])

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=evals, references=refs)


def eval_20newsgroups(eval_model_name):
    labels = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
              "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
              "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
              "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
              "talk.religion.misc"]

    labels = [f"This topic is talk about {label}" for label in labels]

    test_set = load_dataset("SetFit/20_newsgroups")["test"]
    refs = []
    queries = []

    for data in test_set:
        refs.append(data["label"])
        queries.append(data["text"])

    evals = inference(
        eval_model_name=eval_model_name,
        queries=queries,
        labels=labels)

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=evals, references=refs)


def iterative_inference(task, inference_path):
    inference_path = Path(inference_path)
    inference_dirs = [x for x in inference_path.iterdir() if x.is_dir()]

    for inference_dir in inference_dirs:
        if task == "agnews":
            print(inference_dir.name, eval_agnews(inference_dir))
        elif task == "yahoo":
            print(inference_dir.name, eval_yahoo(inference_dir))
        elif task == "dbpedia":
            print(inference_dir.name, eval_dbpedia(inference_dir))
        elif task == "20newsgroups":
            print(inference_dir.name, eval_20newsgroups(inference_dir))


def main():
    # best case
    # iterative_inference(task="agnews", inference_path="../../checkpoints/v1_5/agnews/keep/threshold evaluation/12231537")
    iterative_inference(task="yahoo", inference_path="../../checkpoints/v1_5/yahoo/keep/12240859")
    # iterative_inference(task="dbpedia", inference_path="../../checkpoints/v1_5/dbpedia/12241052")

    # iterative_inference(task="agnews", inference_path="../../checkpoints/v1_5/agnews/12231624")
    # iterative_inference(task="agnews", inference_path="../../checkpoints/v1_5/agnews/12011127")
    # iterative_inference(task="agnews", inference_path="../../checkpoints/v1_5/agnews/12030954")
    # iterative_inference(task="agnews", inference_path="../../checkpoints/v1_5/agnews/12212111")
    # iterative_inference(task="yahoo", inference_path="../../checkpoints/v1_5/yahoo/12212213")
    # iterative_inference(task="yahoo", inference_path="../../checkpoints/v1_5/yahoo/12041014")
    # iterative_inference(task="dbpedia", inference_path="../checkpoints/v1_5/dbpedia/11271358")
    # iterative_inference(task="20newsgroups", inference_path="../checkpoints/v2/20newsgroups/11282259")


if __name__ == "__main__":
    main()
