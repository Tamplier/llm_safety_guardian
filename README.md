---
title: SBERT Suicide watch
emoji: 😢
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: true
short_description: Text classifier that looking for signs of suicidal behavior.
tags:
  - nlp
  - SBERT
  - text-classification
  - psychology
  - suicide
---


# nlp_suicide_watch

## Quick start
```
git clone https://github.com/Tamplier/nlp_suicide_watch.git
cd nlp_suicide_watch

# retrain models
invoke retrain-model

# Build and run Docker container
docker build -t nlp_suicide_watch .

# command line interface
docker run -it nlp_suicide_watch invoke cli

# run telegram bot
docker run -e TELEGRAM_TOKEN="YOUR_TOKEN_HERE" nlp_suicide_watch invoke start-telegram-bot
```

## Basic information

This project contains a machine learning model trained on a [dataset obtained from
Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
The dataset is balanced, which allowed us to use simple accuracy as the primary evaluation metric.

Before training,
a preliminary [exploratory analysis was conducted](https://www.kaggle.com/code/alexandrtinekov/sbert-suicide-watch).

## Preprocessing & Feature extraction
The dataset consists of natural text, which includes typos, elongated words like "soooooo sory", emoticons,
self-censorship such as "you s!ck", and word concatenations like "STOP|STOP|STOP". To handle this,
a preprocessing step was applied to clean the text
(
[here](https://github.com/Tamplier/nlp_suicide_watch/blob/main/src/transformers/sentece_splitter.py)
and
[here](https://github.com/Tamplier/nlp_suicide_watch/blob/main/src/util/typos_processor.py)
).

In addition,
[several features were engineered](https://github.com/Tamplier/nlp_suicide_watch/blob/main/src/transformers/features_extractor.py),
including the percentage of uppercase letters, ratio of exclamation and question marks to the number of sentences,
presence and count of self-censorship, text length, number of sentences, and number of individual emoticons.
After evaluating
[multiple approaches](https://github.com/Tamplier/nlp_suicide_watch/blob/main/src/transformers/feature_selector.py),
the most informative features were selected and integrated into the dataset.

As shown in the plot below, the two most important features for the classification task originate from the
custom feature extractor:
- Question mark rate
- Compression — a metric indicating how much a text is reduced due to repeated characters (e.g., “!!!!” or “sooooo”)

![Feature importance](https://raw.githubusercontent.com/Tamplier/nlp_suicide_watch/main/notebooks/feature_importance_plot.png)

## Training
[Training was performed](https://github.com/Tamplier/nlp_suicide_watch/tree/main/notebooks/kaggle-upload-to-github.ipynb)
on a GPU in Kaggle. SBERT sentence transformer was used to vectorize preprocessed text and ANN (Torch)
to create a classifier based on vectors and extra features from previous step.

Hyperparameter optimization was performed using Optuna and 3 fold cross validation.
The final model achieved approximately 95% accuracy on validation data.

![Loss plot](https://raw.githubusercontent.com/Tamplier/nlp_suicide_watch/main/notebooks/nn_loss_plot.png)

## CI/CD
A [Docker container was built](https://github.com/Tamplier/nlp_suicide_watch/blob/main/Dockerfile)
containing all necessary dependencies, and it is used for all subsequent steps. For quality assurance,
[pytest runs inside the Docker](https://github.com/Tamplier/nlp_suicide_watch/blob/main/.github/workflows/tests.yml)
container on every commit to the main branch, covering the core functionality of the project.
The unit tests are available [in tests folder](https://github.com/Tamplier/nlp_suicide_watch/tree/main/tests).

[Deployment to Hugging Face](https://github.com/Tamplier/nlp_suicide_watch/blob/main/.github/workflows/hugging_face_deploy.yml)
is handled via GitHub Actions. The deployment is manually triggered and blocked if any tests fail,
ensuring only verified versions are released.

![Tests](https://github.com/Tamplier/nlp_suicide_watch/actions/workflows/tests.yml/badge.svg)
![Deploy](https://github.com/Tamplier/nlp_suicide_watch/actions/workflows/hugging_face_deploy.yml/badge.svg)

## Production
The Docker image size is approximately 1.7 GB, and the average RAM usage under normal conditions is around 1.25 GB.
Since this is an educational, non-commercial project, one of the key goals was to minimize hosting costs.
However, due to the inclusion of large machine learning models and dependencies such as SciPy and Torch,
the memory requirements increased significantly.

Given these constraints, finding a free or low-cost hosting option was challenging.
As a result, Hugging Face Spaces was chosen as the production environment.
The project is currently deployed on this platform and can be accessed here:
https://huggingface.co/spaces/Tapocheck77/nlp_suicide_watch
