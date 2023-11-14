import itertools
import logging
import math
from datetime import datetime

import h5py
import torch
from datasets import load_dataset
from sentence_transformers import (
    datasets,
    losses,
    models,
    SentenceTransformer,
    util, InputExample
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from managers.wiki_manager import WikiManager


class TrainerService:
    def __init__(self):
        self.logger = logging.getLogger("wc_sbert_logger")

    @staticmethod
    def get_dev_samples():
        dic = WikiManager.get_wiki_category_dict()
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

    @staticmethod
    def save_wiki_categories_embeddings(model_name_or_path, embeddings_save_path):
        """
        Save wiki categories embeddings by SBERT
        :param model_name_or_path: SBERT model name
        :param embeddings_save_path: embedded save path
        """
        dic = WikiManager.get_wiki_category_dict()
        categories = WikiManager.get_wiki_categories(dic)

        model = SentenceTransformer(model_name_or_path, device="cuda")
        pool = model.start_multi_process_pool()
        embeddings = model.encode_multi_process(sentences=categories, pool=pool)

        model.stop_multi_process_pool(pool)

        print("categories: {}, shape: {}".format(len(categories), embeddings.shape))

        with h5py.File(embeddings_save_path, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings)

    @staticmethod
    def get_wiki_categories_embeddings(model, wiki_categories):
        """
        Get wiki categories embeddings by SBERT
        :param model: SBERT-based model
        :param wiki_categories: wiki categories
        :return: wiki categories embeddings
        """
        return model.encode(wiki_categories, convert_to_numpy=True)

    def train(self,
              model_name,
              model_save_path,
              train_samples,
              use_no_duplicated_dataloader=False,
              train_batch_size=128,
              max_seq_length=128,
              num_epochs=1):
        """
        Train a model with train samples by SBERT
        :param model_name: SBERT model name
        :param model_save_path: model save path
        :param train_samples: train samples for SBERT
        :param use_no_duplicated_dataloader: use NoDuplicatesDataLoader or not
        :param train_batch_size: train batch size
        :param max_seq_length: max sequence length
        :param num_epochs: number of epochs
        """
        ir_queries, ir_corpus, ir_relevant_docs = self.get_dev_samples()

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # NoDuplicate 有 bug 可能會 fixed 住, 用在 wiki pre-trained model 上
        if use_no_duplicated_dataloader:
            train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
        else:
            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

        dev_evaluator = InformationRetrievalEvaluator(
            ir_queries,
            ir_corpus,
            ir_relevant_docs,
            # score_functions={"cos_sim": util.cos_sim},
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

    def train_wiki_pretrained_model(
            self,
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_save_path="../checkpoints/all-mpnet-base-v2",
            train_batch_size=128,
            max_seq_length=128,
            num_epochs=1):
        """
        Train wiki pre-trained model with wiki categories by SBERT
        :param model_name: SBERT model name
        :param model_save_path: pre-trained model save path
        :param train_batch_size: batch size
        :param max_seq_length: max sequence length
        :param num_epochs: number of epochs
        """
        train_samples = []
        dic = WikiManager.get_wiki_category_dict()

        for pid, categories in tqdm(dic.items(), total=len(dic), desc="train samples"):
            for pair in list(itertools.combinations(categories, 2)):
                train_samples.append(InputExample(texts=[pair[0], pair[1]]))

        print("categories: {}, train samples: {}".format(len(dic), len(train_samples)))

        self.train(
            model_name=model_name,
            model_save_path=model_save_path,
            train_samples=train_samples,
            use_no_duplicated_dataloader=True,
            train_batch_size=train_batch_size,
            max_seq_length=max_seq_length,
            num_epochs=num_epochs)

    @staticmethod
    def _batch(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    def self_finetune(self,
                      pretrain_model_name_or_path,
                      model_name_or_path,
                      model_save_path,
                      labels,
                      descriptive_labels,
                      threshold,
                      train_samples: list,
                      train_batch_size=128,
                      max_seq_length=128,
                      num_epochs=1,
                      categories=None):
        """
        Finetune a model with descriptive labels by SBERT
        :param pretrain_model_name_or_path: pre-trained model name or path for self-training finetune phase
        :param model_name_or_path: inference model name or path for self-training inference phase
        :param model_save_path: model save path
        :param labels: target dataset's default labels
        :param descriptive_labels: labels with description for finetune
        :param threshold: threshold for semantic search
        :param train_samples: train samples for SBERT
        :param train_batch_size: train batch size
        :param max_seq_length: max sequence length
        :param num_epochs: number of epochs
        :param categories: wiki categories
        """
        if categories is None:
            dic = WikiManager.get_wiki_category_dict()
            categories = WikiManager.get_wiki_categories(dic)

        model = SentenceTransformer(model_name_or_path, device="cuda:0")
        pool = model.start_multi_process_pool()

        # region self-training inference phase (aim to get train samples)

        start = datetime.now()
        descriptive_labels_embeddings = model.encode(descriptive_labels, convert_to_tensor=True)
        self.logger.info("encode descriptive labels time: {}".format(datetime.now() - start))

        start = datetime.now()
        category_embeddings = self.get_wiki_categories_embeddings(model, categories)
        self.logger.info("encode wiki categories time: {}".format(datetime.now() - start))

        batch_category_embeddings = self._batch(torch.from_numpy(category_embeddings), 100000)

        category_index = 0

        for batch_embeddings in tqdm(batch_category_embeddings,
                                     total=len(batch_category_embeddings),
                                     desc="get train sample by inference wiki categories"):
            preds = util.semantic_search(
                batch_embeddings,
                descriptive_labels_embeddings)

            for pred in preds:
                if pred[0]["score"] >= threshold:
                    label = labels[pred[0]["corpus_id"]]
                    category = categories[category_index]
                    train_samples.append(InputExample(texts=[label, category]))
                category_index += 1

        model.stop_multi_process_pool(pool)

        # endregion

        # region self-training finetune phase (based on pre-trained model)

        self.train(
            model_name=pretrain_model_name_or_path,
            model_save_path=model_save_path,
            train_samples=train_samples,
            use_no_duplicated_dataloader=False,
            train_batch_size=train_batch_size,
            max_seq_length=max_seq_length,
            num_epochs=num_epochs)

        # endregion

    def self_training(self,
                      pretrain_model_name_or_path,
                      model_save_path,
                      labels,
                      descriptive_labels,
                      threshold,
                      num_iterations,
                      train_batch_size=128,
                      max_seq_length=128,
                      num_epochs=1):
        """
        Self-training by SBERT
        :param pretrain_model_name_or_path: pre-trained model name or path for self-training
        :param model_save_path: model save path
        :param labels: target dataset's default labels
        :param descriptive_labels: labels with description for self-training
        :param threshold: threshold for semantic search
        :param num_iterations: number of iterations
        :param train_batch_size: train batch size
        :param max_seq_length: max sequence length
        :param num_epochs: number of epochs
        """
        self.logger.info("---------args---------")
        self.logger.info("pretrain_model_name_or_path: {}".format(pretrain_model_name_or_path))
        self.logger.info("model_save_path: {}".format(model_save_path))
        self.logger.info("labels: {}".format(labels))
        self.logger.info("descriptive_labels: {}".format(descriptive_labels))
        self.logger.info("threshold: {}".format(threshold))
        self.logger.info("num_iterations: {}".format(num_iterations))
        self.logger.info("train_batch_size: {}".format(train_batch_size))
        self.logger.info("max_seq_length: {}".format(max_seq_length))
        self.logger.info("num_epochs: {}".format(num_epochs))
        self.logger.info("---------args---------")

        train_samples = []
        model_name_or_path = pretrain_model_name_or_path
        dic = WikiManager.get_wiki_category_dict()
        categories = WikiManager.get_wiki_categories(dic)

        for i in range(num_iterations):
            self.logger.info("----------------------")
            self.logger.info("iteration {}".format(i))
            self.logger.info("start")
            self.logger.info("input model_name_or_path: {}".format(model_name_or_path))
            self.logger.info("input train_samples: {}".format(len(train_samples)))

            start = datetime.now()
            model_path = "{}/{}".format(model_save_path, i)

            self.logger.info("output model_path: {}".format(model_path))

            self.self_finetune(
                pretrain_model_name_or_path=pretrain_model_name_or_path,
                model_name_or_path=model_name_or_path,
                model_save_path=model_path,
                labels=labels,
                descriptive_labels=descriptive_labels,
                threshold=threshold,
                train_samples=train_samples,
                train_batch_size=train_batch_size,
                max_seq_length=max_seq_length,
                num_epochs=num_epochs,
                categories=categories)

            self.logger.info("output train_samples: {}".format(len(train_samples)))
            self.logger.info("spent time: {}".format(datetime.now() - start))
            self.logger.info("end")
            self.logger.info("----------------------")

            model_name_or_path = model_path

            # 每次都清空 train samples 重新訓練，並且使用最新的 model 來 inference
            pretrain_model_name_or_path = model_path
            train_samples = []

