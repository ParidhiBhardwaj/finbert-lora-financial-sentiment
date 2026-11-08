# FinBERT + LoRA: Financial Sentiment Analysis

## Claim Statement

We claim that applying LoRA fine-tuning to FinBERT will improve its financial sentiment classification performance compared to using the original, unmodified FinBERT model. We will test this by comparing their macro-F1 scores on the Twitter Financial News Sentiment dataset and evaluating whether the improvements transfer to real-world financial news headlines from Yahoo Finance.

## Project Overview

**Track:** Model and Demo Review

**Research Question:** Does additional financial sentiment training improve FinBERT's ability to classify financial text, and does that improvement transfer to other financial news?

FinBERT is a version of Google's BERT model that has been pretrained on financial text. This project investigates whether further fine-tuning an already financially pretrained model can improve its ability to classify financial sentiment.

We compare the original FinBERT model against a version fine-tuned using Low-Rank Adaptation (LoRA). LoRA is a parameter-efficient fine-tuning technique that updates a small number of additional parameters instead of updating all the model's weights.

## Dataset

**Primary Dataset:** Twitter Financial News Sentiment Dataset

The dataset contains financial tweets labeled into three sentiment categories:
- Bullish
- Bearish
- Neutral

We will use the dataset to fine-tune and evaluate our models, using a consistent training and evaluation split.

**Secondary Dataset:** Yahoo Finance News Headlines

We also plan to apply both models to real-world financial news headlines from Yahoo Finance to explore whether the improvements transfer beyond the original Twitter dataset.

## Models

We will evaluate two model configurations:

1. **FinBERT (Baseline):** The original pretrained FinBERT model, without additional fine-tuning on our Twitter dataset.
2. **FinBERT + LoRA:** The same pretrained FinBERT model, further fine-tuned on the Twitter Financial News Sentiment training data using LoRA.

Both models will use the same underlying FinBERT checkpoint and tokenizer to make the comparison consistent.

## Methodology

Our project will follow these steps:

1. Load and explore the Twitter Financial News Sentiment dataset.
2. Prepare the training and evaluation splits.
3. Evaluate the original FinBERT model on the held-out dataset.
4. Apply LoRA to FinBERT and fine-tune it using the training data.
5. Evaluate the fine-tuned model using the same held-out dataset.
6. Compare the models using macro-F1 scores.
7. Apply both models to Yahoo Finance news headlines and examine their predicted sentiment.

## Evaluation

Our primary evaluation metric will be **macro-F1**, which gives equal importance to all three sentiment classes: bullish, bearish, and neutral.

We will compare:
- Macro-F1 before and after LoRA fine-tuning.
- Performance across individual sentiment classes.
- Whether the fine-tuned model's predictions on external financial news differ from the original model's predictions.

Since Yahoo Finance headlines do not come with ground-truth sentiment labels, this external evaluation will focus on prediction patterns and transferability rather than classification accuracy.

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

In progress. We are currently exploring the dataset, setting up the models, and preparing our experimental workflow.
