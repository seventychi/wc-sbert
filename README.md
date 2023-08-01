# wc-sbert
WC-SBERT: Zero-Shot Text Classification via SBERT with Self-Training for Wikipedia Categories

## data structure

You can download the required data for the system from [Google Drive](https://drive.google.com/drive/folders/1bPjBNcgjwayb8jBoyi2OVSARwJbiC_u0?usp=sharing)
Please place the relevant files following the structure below:
- checkpoints
  - agnews
    - all-mpnet-base-v2-iter2-base-0.8
  - all-mpnet-base-v2
  - dbpedia
    - all-mpnet-base-v2-iter1-base-0.7
  - yahoo
    - all-mpnet-base-v2-iter1-base-0.8
- data
  - wiki_categories.json
- embeddings
  - all-mpnet-base-v2
    - wiki_category_embeddings.pkl 
    - wiki_text_embeddings.h5

## experiments
All the related code is placed in the "experiments" directory.
- trainer.py: Pre-trained model training and fine-tuning.
- evaluator.py: Evaluates the model's predictions on different datasets, including the use of basic prompts and self-defined prompts for different labels.
- prompt_evaluation.py: Evaluates the use of prompts.