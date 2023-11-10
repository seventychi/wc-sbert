from managers.wiki_manager import WikiManager
from datasets import load_dataset

import requests


def get_categories(wiki_categories_dict):
    """
    Get all categories from wiki_categories_dict
    :param wiki_categories_dict: wiki categories dictionary from data/wiki_categories.json
    :return: total categories without duplication
    """
    categories = []

    for key, value in wiki_categories_dict.items():
        categories.extend(value)

    return list(set(categories))


if __name__ == "__main__":
    dic = WikiManager.get_wiki_categories_dict()

    print("Total categories:", len(get_categories(dic)))
