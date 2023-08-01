import h5py
import itertools
import json
import math
import torch
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import (
    datasets,
    InputExample,
    losses,
    models,
    SentenceTransformer,
    util
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_wiki_sentences_embeddings():
    model = SentenceTransformer("../checkpoints/all-mpnet-base-v2", device="cuda")
    wiki_set = load_dataset("wikipedia", "20220301.en", split="train")

    sentences = []

    for text in wiki_set["text"]:
        sentences.append(" ".join(text.split(" ")[:200]))

    pool = model.start_multi_process_pool()

    start = datetime.now()
    embeddings = model.encode_multi_process(sentences=sentences, pool=pool)

    print("mult-process", datetime.now() - start, embeddings.shape)

    model.stop_multi_process_pool(pool)

    # 儲存嵌入到 HDF5 檔案
    with h5py.File("../embeddings/all-mpnet-base-v2/wiki_text_embeddings.h5", 'w') as hf:
        hf.create_dataset('embeddings', data=embeddings)


def get_wiki_categories_dict():
    dic = {}

    with open("../data/wiki_categories.json", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="wiki categories"):
            data = json.loads(line)
            dic[str(data["id"])] = data["categories"]

    return dic


def get_dev_samples():
    dic = get_wiki_categories_dict()
    dev_set = load_dataset("wikipedia", "20220301.en", split="train[:{}]".format(int(len(dic.items()) * 0.001)))

    ir_queries = {}
    ir_corpus = {}
    ir_relevant_docs = {}
    ir_corpus_reversed = {}
    corpus_id = 1

    for data in tqdm(dev_set, total=len(dev_set), desc="dev samples"):
        pid = data["id"]
        categories = dic[pid]

        if len(categories) == 3:
            ir_queries[pid] = " ".join(data["text"].split(" ")[:20])
            ir_relevant_docs[pid] = set()

            for category in categories:
                if category not in ir_corpus_reversed:
                    ir_corpus_reversed[category] = str(corpus_id)
                    corpus_id += 1
                ir_relevant_docs[pid].add(ir_corpus_reversed[category])

    for key, value in ir_corpus_reversed.items():
        ir_corpus[value] = key

    return ir_queries, ir_corpus, ir_relevant_docs


def batch(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def train(model_name, model_save_path, train_samples, use_no_duplicated_dataloader=False):
    train_batch_size = 128
    max_seq_length = 128
    num_epochs = 1
    ir_queries, ir_corpus, ir_relevant_docs = get_dev_samples()

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")

    # NoDuplicate 有 bug 可能會 fixed 住, 用在 wiki pre-trained model 上
    if use_no_duplicated_dataloader:
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
    else:
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    dev_evaluator = InformationRetrievalEvaluator(
        ir_queries,
        ir_corpus,
        ir_relevant_docs,
        score_functions={"cos_sim": util.cos_sim},
        map_at_k=[3]
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=int(len(train_dataloader) * 0.1),
        warmup_steps=warmup_steps,
        save_best_model=True,
        output_path=model_save_path,
        use_amp=True)


def finetune(labels, label_embeddings, finetune_model_name, model_save_path, threshold=0.8):
    dic = get_wiki_categories_dict()
    page_categories = list(dic.values())

    label_dic = {}

    for label in labels:
        label_dic[label] = set()

    train_samples = []

    with h5py.File("../embeddings/all-mpnet-base-v2/wiki_text_embeddings.h5", "r") as f:
        embeddings = f["embeddings"][:]

    batch_embeddings = batch(torch.from_numpy(embeddings), 100000)

    # page index
    index = 0

    for query_embeddings in tqdm(batch_embeddings, total=len(batch_embeddings), desc="training samples"):
        preds = util.semantic_search(
            query_embeddings,
            label_embeddings)

        for pred in preds:
            # pred[0]: 每個 query 最相似的 label
            if pred[0]["score"] >= threshold:
                label = labels[pred[0]["corpus_id"]]
                for category in page_categories[index]:
                    label_dic[label].add(category)
            index += 1

    for label, categories in label_dic.items():
        print(label, len(categories))

        for category in categories:
            train_samples.append(InputExample(texts=[label, category]))

    print(len(train_samples))

    train(model_name=finetune_model_name,
          model_save_path=model_save_path,
          train_samples=train_samples)


def train_wiki_pretrained_model():
    train_samples = []
    dic = get_wiki_categories_dict()

    for pid, categories in tqdm(dic.items(), total=len(dic), desc="train samples"):
        for pair in list(itertools.combinations(categories, 2)):
            train_samples.append(InputExample(texts=[pair[0], pair[1]]))

    train(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_save_path="../checkpoints/all-mpnet-base-v2",
        train_samples=train_samples,
        use_no_duplicated_dataloader=True
    )


def finetune_with_agnews(eval_model_name, finetune_model_name, model_save_path, threshold=0.8):
    model = SentenceTransformer(eval_model_name, device="cuda")

    labels = ["World", "Sports", "Business", "Science", "Technology"]
    label_prompts = [
        "This topic is talk about World",
        "This topic is talk about Sports",
        "This topic is talk about Business",
        "This topic is talk about Science",
        "This topic is talk about Technology"]

    # label_prompts = []
    #
    # for label in labels:
    #     label_prompts.append("This sentence is belong to {}".format(label))

    label_embeddings = model.encode(label_prompts, convert_to_tensor=True)

    finetune(
        labels=labels,
        label_embeddings=label_embeddings,
        finetune_model_name=finetune_model_name,
        model_save_path=model_save_path,
        threshold=threshold)


def finetune_with_yahoo(eval_model_name, finetune_model_name, model_save_path, threshold=0.8):
    model = SentenceTransformer(eval_model_name, device="cuda")

    labels = [
        "Society",
        "Culture",
        "Science",
        "Mathematics",
        "Health",
        "Education",
        "Reference",
        "Computers",
        "Internet",
        "Sports",
        "Business",
        "Finance",
        "Entertainment",
        "Music",
        "Family",
        "Relationships",
        "Politics",
        "Government"
    ]

    label_prompts = []

    for label in labels:
        label_prompts.append("This topic is talk about {}".format(label))
    label_embeddings = model.encode(label_prompts, convert_to_tensor=True)

    finetune(
        labels=labels,
        label_embeddings=label_embeddings,
        finetune_model_name=finetune_model_name,
        model_save_path=model_save_path,
        threshold=threshold)


def finetune_with_dbpedia(eval_model_name, finetune_model_name, model_save_path, threshold=0.8):
    model = SentenceTransformer(eval_model_name, device="cuda")
    labels = [
        "Company",
        "Education institution",
        "Artist",
        "Athlete",
        "Office holder",
        "Mean of transportation",
        "Building",
        "Nature place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written work"]

    label_prompts = []

    for label in labels:
        label_prompts.append("This sentence is belong to {}".format(label))
        # label_prompts.append("This topic is talk about {}".format(label))
    label_embeddings = model.encode(label_prompts, convert_to_tensor=True)

    finetune(
        labels=labels,
        label_embeddings=label_embeddings,
        finetune_model_name=finetune_model_name,
        model_save_path=model_save_path,
        threshold=threshold)


def main():
    # save wiki sentences embeddings to h5
    save_wiki_sentences_embeddings()

    # Train a pre-trained WC-SBERT based model
    train_wiki_pretrained_model()

    # region fine-tune agnews

    # iter 1
    finetune_with_agnews(
        eval_model_name="../checkpoints/all-mpnet-base-v2",
        finetune_model_name="../checkpoints/all-mpnet-base-v2",
        model_save_path="../checkpoints/agnews/all-mpnet-base-v2-iter1-base-0.8",
        threshold=0.8
    )

    # iter 2
    finetune_with_agnews(
        eval_model_name="../checkpoints/agnews/all-mpnet-base-v2-iter1-base-0.8",
        finetune_model_name="../checkpoints/all-mpnet-base-v2",
        model_save_path="../checkpoints/agnews/all-mpnet-base-v2-iter2-base-0.8",
        threshold=0.8
    )

    # endregion

    # region fine-tune yahoo

    # iter 1
    finetune_with_yahoo(
        eval_model_name="../checkpoints/all-mpnet-base-v2",
        finetune_model_name="../checkpoints/all-mpnet-v2",
        model_save_path="../checkpoints/yahoo/all-mpnet-base-v2-iter1-base-0.8"
    )

    # endregion

    # region fine-tune dbpedia

    # iter 1
    finetune_with_dbpedia(
        eval_model_name="../checkpoints/all-mpnet-base-v2",
        finetune_model_name="../checkpoints/all-mpnet-base-v2",
        model_save_path="../checkpoints/dbpedia/all-mpnet-base-v2-iter1-base-0.7",
        threshold=0.7
    )

    # endregion


if __name__ == "__main__":
    main()
