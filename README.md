# Recommendation System: Two-Tower + Item-Based Collaborative Filtering (ItemCF)

This project implements two types of recommendation models using telecom business data:
- A **Two-Tower Neural Network** using TensorFlow for deep learning-based recall.
- An **Item-Based Collaborative Filtering (ItemCF)** model for fast heuristic-based recommendation.

It supports full pipelines: data preprocessing, model training, inference, evaluation, and feature enhancement.


## 🚀 Features

### ✅ Two-Tower Model
- Dual-embedding architecture for users and items
- Deep neural network with dense layers and dropout
- Learns user-item relevance via dot-product similarity
- Encodes `member_id` and `goods_id` using `LabelEncoder`
- Supports feature augmentation (age, gender, quantity, etc.)

### ✅ Item-Based Collaborative Filtering (ItemCF)
- Computes similarity between items using co-occurrence or cosine similarity
- Recommends items similar to those the user has interacted with
- Efficient for quick prototyping and interpretability
- Lightweight, no neural network required

## 🧪 Requirements

- Python 3.11
- pandas
- numpy
- scikit-learn
- TensorFlow 2.x
- joblib


## 📊 Usage

- Prepare data
- Train two-tower model
- Inference with Two-Tower Model
- Evaluate Two-Tower Model

## 🔍 Evaluation Metrics

Precision@K: proportion of top-K recommendations that are relevant

Recall@K: proportion of relevant items captured in top-K recommendations

## 🧠 TODOs
✅ Two-Tower baseline with ID embeddings

✅ ItemCF model and evaluation

✅ Feature engineering for user/item profiles

⏳ Add user/item auxiliary features into tower input layers

⏳ Optimize ranking quality via sampling & loss tuning

⏳ Add diversity/novelty metrics
