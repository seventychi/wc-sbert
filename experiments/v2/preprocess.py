import logging
from datetime import datetime

from services.trainer_service import TrainerService

CHECKPOINT = "../checkpoints/all-mpnet-base-v2"
TRAIN_PRETRAINED_MODEL = False
SAVE_WIKI_CATEGORIES_EMBEDDINGS = False


def main():
    logger = logging.getLogger("wc_sbert_logger")
    file_handler = logging.FileHandler("../logs/{}.log".format(datetime.now().strftime("%m%d%H%M")))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    trainer_service = TrainerService()

    if TRAIN_PRETRAINED_MODEL:
        trainer_service.train_wiki_pretrained_model(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_save_path=CHECKPOINT,
            train_batch_size=128,
            max_seq_length=128,
            num_epochs=1)

    if SAVE_WIKI_CATEGORIES_EMBEDDINGS:
        TrainerService.save_wiki_categories_embeddings(
            model_name_or_path=CHECKPOINT,
            embeddings_save_path="{}/wiki_category_embeddings.h5".format(CHECKPOINT))


if __name__ == "__main__":
    main()
