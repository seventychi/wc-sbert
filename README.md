# WC-SBERT: Zero-Shot Topic Classification Using SBERT and Self-Training with Wikipedia Categories

This repository contains programs related to the research "WC-SBERT: Zero-Shot Topic Classification Using SBERT and Self-Training with Wikipedia Categories".

---

**Journal**: Accepted by ACM Transactions on Intelligent Systems and Technology

**Authors**: TE-YU CHI, JYH-SHING ROGER JANG

**Affiliation**: Dept. of CSIE, National Taiwan University, Taiwan

**Paper URL**:

**Cite**:

---





## Getting Started

### Models and Data

You can download the required models data for the system from [Google Drive](https://drive.google.com/file/d/1TL4tHie9PJupK3NYBAJ271SEm7ywdqEH/view?usp=drive_link). Please place the relevant files into the project root directory. Below is the detailed directory structure:

**checkpoints**: Contains the WC-SBERT models for downstream tasks

* all-mpnet-base-v2: WC-SBERT pre-trained model

* agnews: fine-tuned WC-SBERT with AGNews target labels.

* dbpedia: fine-tuned WC-SBERT with DBPedia target labels.

* yahoo: fine-tuned WC-SBERT with Yahoo Answers target labels.

**data**: wc-category dataset, which you can also find and use on Hugging-Face at [seven-tychi/wikipedia-categories](https://huggingface.co/datasets/seven-tychi/wikipedia-categories) if needed.

**embeddings**: pre-stored Wikipedia text embeddings

## Experiment result

Please execute the following command to reproduce the experiment result.

```
python experiments/inference.py
```

For instructions on how to fine-tune WC-SBERT for different tasks, refer to finetune.py.