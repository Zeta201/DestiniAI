import ast
import pandas as pd


def extract_interactions_from_likes(df):
    interactions = []
    for _, row in df.iterrows():
        user_id = row['user_id']
        try:
            liked_ids = ast.literal_eval(row['likes']) if isinstance(
                row['likes'], str) else row['likes']
            negative_ids = ast.literal_eval(row['negative_samples']) if isinstance(
                row['negative_samples'], str) else row['negative_samples']

            for liked_id in liked_ids:
                interactions.append((user_id, liked_id, 1)
                                    )  # implicit positive
            for nid in negative_ids:
                interactions.append((user_id, nid, 0))  # negatives
        except (ValueError, SyntaxError):
            continue
    return pd.DataFrame(interactions, columns=['user_id', 'profile_id', 'rating'])


def get_similarity_scores_for_user(user_id, profile_ids, model, trainset):
    scores = []
    for profile_id in profile_ids:
        try:
            _ = trainset.to_inner_uid(user_id)
            _ = trainset.to_inner_iid(profile_id)
            prediction = model.predict(user_id, profile_id)
            scores.append({
                "user_id": user_id,
                "profile_id": profile_id,
                "similarity_score": round(prediction.est, 4)
            })
        except ValueError:
            continue  # Skip if user or profile not in training set
    return scores
