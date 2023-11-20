import logging
import os
from datetime import datetime

from services.trainer_service import TrainerService

PRETRAINED_MODEL_CHECKPOINT = "../checkpoints/all-mpnet-base-v2"
FINE_TUNE_TARGET = "agnews"  # agnews, yahoo, dbpedia
MODEL_DIR = "../checkpoints/v2/{}/{}".format(FINE_TUNE_TARGET, datetime.now().strftime("%m%d%H%M"))


def configure_logger():
    logger = logging.getLogger("wc_sbert_logger")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("{}/log.log".format(MODEL_DIR))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def fine_tune_agnews():
    trainer_service = TrainerService()

    labels = ["World", "Sports", "Business", "Science", "Technology"]
    descriptive_labels = [f"This topic is talk about {label}" for label in labels]

    # generate descriptive labels by gpt4
    # descriptive_labels = [
    #     "Covers global news, international affairs, and events happening around the world, focusing on politics, culture, and global trends.",
    #     "Dedicated to athletic activities, including coverage of sports events, athlete profiles, and analyses of various sports games and "
    #     "competitions.",
    #     "Focuses on the corporate sector, market trends, financial news, and economic policies impacting businesses and industries worldwide.",
    #     "Deals with scientific discoveries, research updates, and insights into various fields like biology, physics, and environmental studies.",
    #     "Centers on advancements in technology, including gadget reviews, tech industry developments, and the impact of new technologies on society."
    # ]

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.8,
        num_iterations=10,
        train_batch_size=256,
        max_seq_length=128,
        num_epochs=10,
        description="test batch size 256 and epoch 10"
    )


def fine_tune_yahoo():
    trainer_service = TrainerService()

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
    descriptive_labels = [f"This topic is talk about {label}" for label in labels]

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.8,
        num_iterations=10,
        train_batch_size=128,
        max_seq_length=128,
        num_epochs=1)


def fine_tune_dbpedia():
    trainer_service = TrainerService()

    labels = ["Company", "Education institution", "Artist", "Athlete", "Office holder",
              "Mean of transportation", "Building", "Nature place", "Village", "Animal",
              "Plant", "Album", "Film", "Written work"]
    descriptive_labels = [f"This topic is talk about {label}" for label in labels]

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.8,
        num_iterations=10,
        train_batch_size=128,
        max_seq_length=128,
        num_epochs=1)


def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    configure_logger()

    if FINE_TUNE_TARGET == "agnews":
        fine_tune_agnews()
    elif FINE_TUNE_TARGET == "yahoo":
        fine_tune_yahoo()
    elif FINE_TUNE_TARGET == "dbpedia":
        fine_tune_dbpedia()


if __name__ == "__main__":
    main()
