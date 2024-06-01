import logging
import os
from datetime import datetime


from services.trainer_service import TrainerService

PRETRAINED_MODEL_CHECKPOINT = "../checkpoints/all-mpnet-base-v2"
FINE_TUNE_TARGET = "agnews"  # agnews, yahoo, dbpedia 20newsgroups
MODEL_DIR = "../../checkpoints/{}/{}".format(FINE_TUNE_TARGET, datetime.now().strftime("%m%d%H%M"))


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
    descriptive_labels = [f"This article covers topics related to {label}" for label in labels]
    #
    # descriptive_labels[0] = "This article covers topics related to World, not Business"
    # descriptive_labels[3] = "This article covers topics related to Science, not World"

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.8,
        num_iterations=5,
        train_batch_size=256,
        max_seq_length=128,
        num_epochs=1,
        description=""
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
    # descriptive_labels = [f"This topic is talk about {label}" for label in labels]
    descriptive_labels = [f"This article covers topics related to {label}" for label in labels]

    # descriptive_labels[0] = "This article covers topics related to Society, not Relationships"
    # descriptive_labels[2] = "This article covers topics related to Science, not Education"

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.85,
        num_iterations=5,
        train_batch_size=256,
        max_seq_length=128,
        num_epochs=1,
        description="")


def fine_tune_dbpedia():
    trainer_service = TrainerService()

    labels = ["Company", "Education institution", "Artist", "Athlete", "Office holder",
              "Mean of transportation", "Building", "Nature place", "Village", "Animal",
              "Plant", "Album", "Film", "Written work"]
    # descriptive_labels = [f"This topic is talk about {label}" for label in labels]
    descriptive_labels = [f"This article covers topics related to {label}" for label in labels]

    trainer_service.self_training(
        pretrain_model_name_or_path=PRETRAINED_MODEL_CHECKPOINT,
        model_save_path=MODEL_DIR,
        labels=labels,
        descriptive_labels=descriptive_labels,
        threshold=0.85,
        num_iterations=5,
        train_batch_size=256,
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
