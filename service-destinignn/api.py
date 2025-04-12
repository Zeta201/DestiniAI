from flask import Flask, request, jsonify
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils.feature_engineering import build_feature_matrices
from utils.graph_builder import build_graph
from recommender import GNNRecommender
from model.trainer import GNNTrainer

# Assuming all necessary imports, model definitions, and helper functions from previous steps
# Initialize the Flask application
app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_users = pd.read_csv('data/users.csv')  # or any data loading method

content_matrix, norm_numeric, scaler = build_feature_matrices(df_users)
data = build_graph(df_users, content_matrix, norm_numeric).to(device)

trainer = GNNTrainer(data, df_users, device=device)
model = trainer.train(epochs=100)


# Endpoint for user recommendation


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user_id from the POST request body
    try:
        user_id = int(request.json['user_id'])
    except KeyError:
        return jsonify({'error': 'Missing user_id'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid user_id'}), 400

    recommender = GNNRecommender(model, data, df_users)

    results = recommender.recommend(user_id=user_id, top_n=5)
    recommendations = results[['user_id', 'similarity_score']
                              ].to_dict(orient='records')

    # Return recommendations as JSON
    return jsonify({'recommendations': recommendations})


# @app.route('/user_details', methods=['GET'])
# def user_details():
#     user_id = request.args.get('user_id')
#     if user_id is None:
#         return jsonify({'error': 'Missing user_id parameter'}), 400

#     # Find the user in your data
#     user_data = df_users[df_users['user_id'] == int(user_id)]
#     if user_data.empty:
#         return jsonify({'error': 'User not found'}), 404

#     # Convert the user details to a dictionary and return
#     user_info = user_data.iloc[0].to_dict()
#     return jsonify({'user_details': user_info})

if __name__ == '__main__':
    # Run the app on the default Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)
