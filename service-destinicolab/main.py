from colab_utils.feature_extractor import extract_interactions_from_likes
from colab_utils.builder import build_model
from svd_recommender import SVDRecommender
import pandas as pd
from model.trainer import SVDTrainer


user_df = pd.read_csv('./data/users.csv')
user_id = 10

interactions_df = extract_interactions_from_likes(user_df)
user_ids = user_df['user_id'].unique()
profile_ids = user_df['user_id'].unique()

model, trainset, testset = build_model(interactions_df)
trainer = SVDTrainer(model, trainset, testset)
model, trainset = trainer.train_svd()
if __name__ == '__main__':
    recommender = SVDRecommender(
        model=model, trainset=trainset, interactions_df=interactions_df)

    mutuals = recommender.recommend(user_id)
    recommendations = mutuals
    print(f"\n✅ Mutual interest recommendations for User {user_id}: {mutuals}")
