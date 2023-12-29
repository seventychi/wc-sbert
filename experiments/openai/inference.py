import evaluate
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from tqdm import tqdm

MIRLAB_OPENAI_KEY = "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"


def inference(labels, query):
    try:
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You will be provided with a topic content, and your task is to classify its topic as {labels}.
                Please respond with only one of these labels.            
                """
            ),
            HumanMessagePromptTemplate.from_template("{query}"),
        ])

        llm = ChatOpenAI(openai_api_key=MIRLAB_OPENAI_KEY, model_name="gpt-4-1106-preview")
        result = llm(chat_template.format_messages(
            labels=labels,
            query=query), request_timeout=10)

        print(result.content)

        return result.content
    except Exception as e:
        print(query, e)
        return ""


def eval_agnews():
    labels = ["World", "Sports", "Business", "Science", "Technology"]
    test_set = load_dataset("ag_news")["test"]
    refs = []
    evals = []

    with open("results/agnews_eval.txt", "w", encoding="utf-8") as f:
        for index, data in enumerate(tqdm(test_set, desc="agnews", total=len(test_set))):
            refs.append(data["label"])

            pred = inference(
                labels="World, Sports, Business, Science, Technology",
                query=data["text"])
            try:
                pred_idx = labels.index(pred)
            except:
                pred_idx = -1
            if pred_idx == 4:
                pred_idx = 3
            evals.append(pred_idx)

            f.write("{}\t{}\n".format(index, pred))
            f.flush()

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=evals, references=refs)


def main():
    labels = ["World", "Sports", "Business", "Science", "Technology"]
    test_set = load_dataset("ag_news")["test"]
    refs = []
    evals = []

    for index, data in enumerate(tqdm(test_set, desc="agnews", total=len(test_set))):
        refs.append(data["label"])

    with open("results/agnews_eval.txt", "r", encoding="utf-8") as f:
        for line in f:
            pred = line.strip().split("\t")[1]

            try:
                pred_idx = labels.index(pred)
            except:
                pred_idx = -1
            if pred_idx == 4:
                pred_idx = 3
            evals.append(pred_idx)

    accuracy = evaluate.load("accuracy")
    print(accuracy.compute(predictions=evals, references=refs))
    # eval_agnews()


if __name__ == "__main__":
    main()
