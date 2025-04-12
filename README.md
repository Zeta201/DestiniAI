# DestiniGNN: GNN & SVDpp-Based User Recommendation Microservices

This repository contains **two key recommendation microservices** used in the broader *Destini* dating platform:

1. **Graph Neural Network (GNN)** service
2. **Collaborative Filtering using SVD++**

Both services provide personalized user match recommendations and are queried independently by the orchestrator via API.

---

## Overview

### 1. GNN-Based Recommendation

This service builds a user graph from profile features (e.g., interests, MBTI, goals, etc.), computes embeddings using a GNN (GraphSAGE), and generates top-N match recommendations.

### 2. SVD++ Collaborative Filtering

This service builds a collaborative filtering model using implicit feedback from user likes and negative samples, and computes similarity scores based on user interactions. It supports mutual interest recommendations.

---

## How This Fits Into the Bigger System

This repo contains **two of several microservices** in the Destini ecosystem:

- `DestiniColab` ✅ (this repo) – Collaborative filtering engine (SVD++)
- `DestiniGNN` ✅ (this repo) – GNN-based profile matching
- `DestiniRL`: Reinforcement Learning-based feedback refiner
- `DestiniVision`: Vision model for profile photo-based embedding
- `DestiniOrchestrator`: Central blackboard controller to merge multiple recommendation sources

Each microservice runs independently and is queried by the orchestrator.

---

## Tech Stack

- Python 3.10+
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (GNN model)
- [Surprise Library](http://surpriselib.com/) (SVD++ model)
- Sentence Transformers (text embeddings)
- FastAPI (API layer)
- NumPy, Scikit-learn, Pandas

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Zeta201/DestiniGNN.git
cd DestiniGNN
```

### 2. Setup environment

```bash
chmod +x install.sh
./install.sh
```

### 3. Run the service

```bash
python app.py
```

By default, it:

- Loads `gnn_model.pt` or `svd_model.pkl` if available
- Otherwise trains from scratch and saves the model

---

## 🔗 API Reference

### `POST /recommend` (GNN)

Returns GNN-based recommended user profiles.

**Request Body:**

```json
{
  "user_id": 12
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "age": 33,
      "name": "Brian Green",
      "similarity_score": 0.9845,
      "user_id": 734
    },
    {
      "age": 40,
      "name": "Willie Steele",
      "similarity_score": 0.9843,
      "user_id": 382
    }
  ]
}
```

---

### `POST /recommend/svd` (Collaborative Filtering)

Returns user recommendations based on implicit feedback using SVD++.

**Request Body:**

```json
{
  "user_id": 10
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "user_id": 10,
      "profile_id": 23,
      "similarity_score": 0.9112
    },
    {
      "user_id": 10,
      "profile_id": 41,
      "similarity_score": 0.8749
    }
  ]
}
```

---

## Model Persistence

- GNN model saved at: `saved_models/gnn_model.pt`
- SVD++ model saved at: `saved_models/svd_model.pkl`
- Both services automatically load saved models if available

---

## License

MIT © 2025

---

