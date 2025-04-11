import torch
import pandas as pd
from utils.feature_engineering import build_feature_matrices
from utils.graph_builder import build_graph
from model.trainer import GNNTrainer
from recommender import GNNRecommender


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_users = pd.read_csv('data/users.csv')  # or any data loading method

content_matrix, norm_numeric, scaler = build_feature_matrices(df_users)
data = build_graph(df_users, content_matrix, norm_numeric).to(device)

trainer = GNNTrainer(data, df_users, device=device)
model = trainer.train(epochs=100)
torch.save(model.state_dict(), 'saved_models/gnn_model.pt')
print('Model saved successfully!\n')
recommender = GNNRecommender(model, data, df_users)
results = recommender.recommend(user_id=1, top_n=20)
print(results[['user_id', 'name', 'age',
      'relationship_goal', 'similarity_score']])
