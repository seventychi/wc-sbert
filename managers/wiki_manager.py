import itertools
import json

from sentence_transformers import InputExample
from tqdm import tqdm


class WikiManager:
    @staticmethod
    def get_wiki_category_dict(category_file_name: str = "../data/wiki_categories.json"):
        """
        Get wiki category dictionary from data/wiki_categories.json
        :param category_file_name: wiki categories file name
        :return: wiki categories dictionary with key: page id, value: categories
        """
        dic = {}

        with open(category_file_name, "r", encoding="utf-8") as f:
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

    @staticmethod
    def get_wiki_bge_train_samples(wiki_category_dict: dict, output_file_name: str):
        """
        Get BGE train samples from wiki categories dictionary
        :param wiki_category_dict: wiki categories dictionary with key: page id, value: categories
        :param output_file_name: output file name
        :return: train samples for BGE
        """
        with open(output_file_name, "w", encoding="utf-8") as f:
            for pid, categories in tqdm(wiki_category_dict.items(), total=len(wiki_category_dict), desc="train samples"):
                if len(categories) == 1:
                    continue

                sample = {
                    "query": categories[0],
                    "pos": categories[1:],
                    "neg": []
                }

                f.write(json.dumps(sample) + "\n")
                return
