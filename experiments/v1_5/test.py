import evaluate
import h5py
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from sentence_transformers import (
    SentenceTransformer,
    util
)
from tqdm import tqdm

from managers.wiki_manager import WikiManager

MIRLAB_OPENAI_KEY = "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"


def rephrase(labels):
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are currently performing a topic classification task using sentence embedding similarity. You need to rephrase the topic labels 
            into more meaningful and easily classifiable sentences. Please retain the original label in the sentence and incorporate the opposite 
            meaning of other labels to enhance differentiation if the labels is ambiguous. Please generate sentences for each label, 
            and the sentence should short as possible.
            """
        ),
        AIMessagePromptTemplate.from_template("{labels}"),
        HumanMessagePromptTemplate.from_template("{labels}"),
    ])

    llm = ChatOpenAI(openai_api_key=MIRLAB_OPENAI_KEY, model_name="gpt-4-1106-preview")
    result = llm(chat_template.format_messages(labels=labels), request_timeout=60)

    print(result.content)


def main():
    labels = ["Company", "Education institution", "Artist", "Athlete", "Office holder",
              "Mean of transportation", "Building", "Nature place", "Village", "Animal",
              "Plant", "Album", "Film", "Written work"]
    labels = [f"This article covers topics related to {label}" for label in labels]

    model = SentenceTransformer("../../checkpoints/v1_5/dbpedia/12241052/0")
    label_embeddings = model.encode(labels)
    query_embeddings = model.encode(labels)

    preds = util.semantic_search(label_embeddings, query_embeddings, top_k=14)

    cm = []

    for i1, pred_1 in enumerate(preds):
        l1 = [_ for _ in range(14)]
        for pred_2 in pred_1:
            l1[pred_2["corpus_id"]] = round(pred_2["score"], 2)
        cm.append(l1)

        # l1 = labels[i1]
        # for pred_2 in pred_1:
        #     print("{} x {}: {}".format(l1, labels[pred_2["corpus_id"]], pred_2["score"]))

    for row in cm:
        print(row)



if __name__ == "__main__":
    main()
