from flask import Flask, request, jsonify
import pandas as pd
from colab_utils.feature_extractor import extract_interactions_from_likes
from colab_utils.builder import build_model
from svd_recommender import SVDRecommender
from model.trainer import SVDTrainer

app = Flask(__name__)


user_df = pd.read_csv('./data/users.csv')

interactions_df = extract_interactions_from_likes(user_df)
user_ids = user_df['user_id'].unique()
profile_ids = user_df['user_id'].unique()

model, trainset, testset = build_model(interactions_df)
trainer = SVDTrainer(model, trainset, testset)
model, trainset = trainer.train_svd()

# Endpoint for user recommendation


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.json['user_id'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Missing or invalid user_id'}), 400

    # Optional field: top_n
    top_n = request.json.get('top_n', 5)  # default to 5
    try:
        top_n = int(top_n)
    except ValueError:
        return jsonify({'error': 'top_n must be an integer'}), 400

    recommender = SVDRecommender(
        model=model, trainset=trainset, interactions_df=interactions_df)

    mutuals = recommender.recommend(user_id, top_n=top_n)
    return jsonify({'recommendations': mutuals})


if __name__ == '__main__':
    # Run the app on the default Flask server
    app.run(debug=True, host='0.0.0.0', port=3000)
