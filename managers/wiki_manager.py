import itertools
import json

from sentence_transformers import InputExample
from tqdm import tqdm


class WikiManager:
    @staticmethod
    def get_wiki_category_dict():
        """
        Get wiki category dictionary from data/wiki_categories.json
        :return: wiki categories dictionary with key: page id, value: categories
        """
        dic = {}

        with open("../data/wiki_categories.json", "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="wiki categories"):
                data = json.loads(line)
                dic[str(data["id"])] = data["categories"]

        return dic

    @staticmethod
    def get_wiki_categories(wiki_category_dict):
        """
        Get all categories from wiki categories dictionary
        :param wiki_category_dict: wiki categories dictionary with key: page id, value: categories
        :return: total categories without duplication
        """
        categories = []

        for key, value in wiki_category_dict.items():
            categories.extend(value)

        return list(set(categories))

    @staticmethod
    def get_wiki_train_samples(wiki_category_dict: dict):
        """
        Get SBERT train samples from wiki categories dictionary
        :param wiki_category_dict: wiki categories dictionary with key: page id, value: categories
        :return: train samples for SBERT
        """
        train_samples = []

        for pid, categories in tqdm(wiki_category_dict.items(), total=len(wiki_category_dict), desc="train samples"):
            for pair in list(itertools.combinations(categories, 2)):
                train_samples.append(InputExample(texts=[pair[0], pair[1]]))

        return train_samples

