# FinBERT Fine-Tuning: Financial Sentiment Analysis

## Claim Statement

We claim that parameter-efficient fine-tuning using LoRA can improve FinBERT's financial sentiment classification performance while requiring fewer trainable parameters than full fine-tuning. We will test this by comparing zero-shot FinBERT, fully fine-tuned FinBERT, and FinBERT with LoRA adapters of different ranks (4, 8, and 12) on the Twitter Financial News Sentiment dataset and evaluating their ability to generalize to other financial sentiment datasets.

## Project Overview

**Track:** Model and Demo Review

**Research Question:** How do different fine-tuning approaches affect FinBERT's financial sentiment classification performance, and how well do the resulting models generalize to other financial sentiment datasets?

FinBERT is a BERT-based language model pretrained on financial text and fine-tuned for financial sentiment classification using the Financial PhraseBank dataset.

This project investigates whether additional training on the Twitter Financial News Sentiment dataset improves its performance and whether parameter-efficient fine-tuning methods, such as LoRA, can achieve results comparable to full fine-tuning while updating fewer parameters.

## Datasets

### Primary Dataset: Twitter Financial News Sentiment

The Twitter Financial News Sentiment dataset contains financial tweets labeled into three sentiment categories:

- Bullish
- Bearish
- Neutral

This dataset will be used to fine-tune and evaluate the models.

### External Evaluation Datasets

We will also evaluate the models on additional financial sentiment datasets to examine how well they generalize to financial text outside the original Twitter dataset.

One potential external evaluation dataset is the Financial PhraseBank dataset.

## Models and Experimental Setups

We will compare the following configurations:

### 1. FinBERT Zero-Shot (Baseline)
The original pretrained FinBERT model, evaluated without additional fine-tuning on the Twitter dataset.

### 2. FinBERT Full Fine-Tuning
FinBERT fine-tuned on the Twitter training dataset, updating as many model layers as computationally feasible.

### 3. FinBERT + LoRA (Rank 4)
FinBERT fine-tuned using Low-Rank Adaptation (LoRA) with a rank of 4.

### 4. FinBERT + LoRA (Rank 8)
FinBERT fine-tuned using LoRA with a rank of 8.

### 5. FinBERT + LoRA (Rank 12)
FinBERT fine-tuned using LoRA with a rank of 12.

### Optional Experiment: AdaLoRA

If time and computational resources permit, we may also explore AdaLoRA, an adaptive version of LoRA that dynamically allocates parameter budgets across model layers.

## Methodology

Our project will follow these steps:

1. Load and explore the Twitter Financial News Sentiment dataset.
2. Prepare consistent training, validation, and test splits.
3. Evaluate the original FinBERT model without additional training (zero-shot).
4. Fully fine-tune FinBERT on the Twitter training dataset.
5. Apply LoRA to FinBERT and train separate configurations using ranks 4, 8, and 12.
6. Evaluate all model configurations on the same held-out Twitter test set.
7. Compare performance, trainable parameters, and computational requirements.
8. Evaluate the models on additional financial sentiment datasets to examine generalization.

## Evaluation Metrics

Our primary evaluation metric will be **macro-F1**, which gives equal importance to all three sentiment classes: bullish, bearish, and neutral.

We will also examine:

- Precision and recall for each sentiment class.
- Number and percentage of trainable parameters.
- Training time and computational requirements.
- Performance differences across external financial sentiment datasets.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT (LoRA)
- Scikit-learn
- Pandas
- Google Colab
- NJIT Wulver HPC

## Team Members

- Ariana Daka
- Paridhi Bhardwaj
- Sahaj Singh

## Project Status

In progress. We are preparing the datasets, implementing the different fine-tuning approaches, and setting up the experimental workflow.
