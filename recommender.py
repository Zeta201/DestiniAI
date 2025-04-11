import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GNNRecommender:
    def __init__(self, model, data, df_users):
        self.model = model
        self.data = data
        self.df_users = df_users

    def recommend(self, user_id, top_n=5):
        user_idx = self.df_users[self.df_users['user_id'] == user_id].index[0]
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.data).cpu().numpy()

        user_emb = embeddings[user_idx]
        sims = cosine_similarity([user_emb], embeddings).flatten()
        top_k_idx = sims.argsort()[-(top_n + 1):-1][::-1]
        recs = self.df_users.iloc[top_k_idx].copy()
        recs['similarity_score'] = sims[top_k_idx]
        return recs
