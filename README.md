# DestiniGNN: GNN-Based User Recommendation Microservice

This is the **Graph Neural Network (GNN)**-based recommendation microservice used in the broader _Destini_ dating platform. It computes personalized user recommendations based on profile attributes and relationship compatibility using graph learning techniques.

---

## Overview

This microservice builds a user graph from profile features (e.g., interests, MBTI, goals, etc.), computes embeddings using a GNN (GraphSAGE), and generates top-N personalized match recommendations. It exposes a FastAPI-based REST API for other services (e.g., frontend, feedback engine) to consume.

---

## Tech Stack

- Python 3.10+
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Sentence Transformers (for text embeddings)
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
chmod +x setup.sh
./ setup.sh
```

### 3. Run the service

```bash
python app.py
```

By default, it:

- Loads `gnn_model.pt` if it exists
- Otherwise trains from scratch and saves the model

---

## 🔗 API Reference

### `POST /recommend`

Returns a list of recommended user profiles.

**Request Body:**

```json
{
  "user_id": 12
}
```

**Response:**

```json
  "recommendations": [
    {
      "age": 33,
      "name": "Brian Green",
      "similarity_score": 0.9845097064971924,
      "user_id": 734
    },
    {
      "age": 40,
      "name": "Willie Steele",
      "similarity_score": 0.9843288660049438,
      "user_id": 382
    },
    {
      "age": 31,
      "name": "Charles Martinez",
      "similarity_score": 0.9841940402984619,
      "user_id": 698
    },
    {
      "age": 33,
      "name": "Robert Knight",
      "similarity_score": 0.9841533899307251,
      "user_id": 412
    },
]
```
---

## Model Persistence

- Trained models are saved at: `saved_models/gnn_model.pt`
- On restart, the service automatically loads the saved model to avoid retraining

---

## How This Fits Into the Bigger System

This is **one of several microservices** in the Destini ecosystem. Other services (not in this repo):

- `DestiniColab`: Colaborative filtering engine
- `DestiniRL`: Reinforcement Learning based feedback refiner
- `DestiniVision`: Vision model for profile photo-based embedding
- `DestiniOrchestrator`: Central blackboard controller to merge multiple recommendation sources

This GNN service runs independently but is queried by the orchestrator via API.

---

---

## License

MIT © 2025

---
