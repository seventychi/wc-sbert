from managers.wiki_manager import WikiManager

wiki_category_dict = WikiManager.get_wiki_category_dict("../../data/wiki_categories.json")
WikiManager.get_wiki_bge_train_samples(wiki_category_dict, "../../data/bge_train_samples_test.jsonl")