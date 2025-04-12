import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import numpy as np
from torch_geometric.data import Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Haversine distance function


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_graph(df_users, content_matrix, normalized_numerical, top_k=10):
    num_users = len(df_users)
    content_similarity = cosine_similarity(content_matrix)
    num_similarity = 1 / (1 + cdist(normalized_numerical,
                          normalized_numerical, metric='euclidean'))

    latitudes = df_users['latitude'].values
    longitudes = df_users['longitude'].values
    lat1, lat2 = np.meshgrid(latitudes, latitudes)
    lon1, lon2 = np.meshgrid(longitudes, longitudes)
    distances = haversine(lat1, lon1, lat2, lon2)
    location_similarity = np.exp(-distances / 100)

    similarity_matrix = 0.33 * content_similarity + \
        0.33 * num_similarity + 0.33 * location_similarity

    top_k_neighbors = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    edge_list = [(i, j) for i in range(num_users)
                 for j in top_k_neighbors[i] if i != j]
    edge_index = torch.tensor(np.array(edge_list).T,
                              dtype=torch.long).to(device)

    node_features = np.hstack([content_matrix, normalized_numerical])
    x = torch.tensor(node_features, dtype=torch.float).to(device)

    return Data(x=x, edge_index=edge_index)


def create_contrastive_pairs(df_users, batch_size=512, device='cpu'):
    batch_size = min(batch_size, len(df_users))
    indices = np.random.choice(len(df_users), (batch_size, 2), replace=True)
    pairs = torch.tensor(indices, dtype=torch.long).to(device)

    # Compare relationship goal
    same_relationship = df_users.iloc[pairs[:, 0]
                                      ]['relationship_goal'].values == df_users.iloc[pairs[:, 1]]['relationship_goal'].values

    # Compare MBTI
    same_mbti = df_users.iloc[pairs[:, 0]
                              ]['MBTI'].values == df_users.iloc[pairs[:, 1]]['MBTI'].values

    # Compare interests using Jaccard similarity
    interests1 = df_users.iloc[pairs[:, 0]
                               ]['interests'].str.split(', ').apply(set)
    interests2 = df_users.iloc[pairs[:, 1]
                               ]['interests'].str.split(', ').apply(set)
    jaccard_sim = [(len(i1 & i2) / len(i1 | i2)) if len(i1 | i2)
                   > 0 else 0 for i1, i2 in zip(interests1, interests2)]

    # Compare age (normalized)
    age_diff = np.abs(
        df_users.iloc[pairs[:, 0]]['age'].values - df_users.iloc[pairs[:, 1]]['age'].values)
    max_age = df_users['age'].max()
    age_similarity = 1 - \
        (age_diff / max_age) if max_age > 0 else np.zeros_like(age_diff)

    # Compare location (Haversine distance)
    lat1, lon1 = df_users.iloc[pairs[:, 0]
                               ]['latitude'].values, df_users.iloc[pairs[:, 0]]['longitude'].values
    lat2, lon2 = df_users.iloc[pairs[:, 1]
                               ]['latitude'].values, df_users.iloc[pairs[:, 1]]['longitude'].values
    distances = haversine(lat1, lon1, lat2, lon2)
    location_similarity = np.exp(-distances / 100)

    # Compare zodiac sign
    zodiac_similarity = (df_users.iloc[pairs[:, 0]]['zodiac_sign'].values ==
                         df_users.iloc[pairs[:, 1]]['zodiac_sign'].values)

    # Compare height
    height_diff = np.abs(
        df_users.iloc[pairs[:, 0]]['height_cm'].values - df_users.iloc[pairs[:, 1]]['height_cm'].values)
    max_height = df_users['height_cm'].max()
    height_similarity = 1 - \
        (height_diff / max_height) if max_height > 0 else np.zeros_like(height_diff)

    # Compare body type
    same_body_type = df_users.iloc[pairs[:, 0]
                                   ]['body_type'].values == df_users.iloc[pairs[:, 1]]['body_type'].values

    # Compare languages spoken (Jaccard similarity)
    languages1 = df_users.iloc[pairs[:, 0]
                               ]['languages_spoken'].str.split(', ').apply(set)
    languages2 = df_users.iloc[pairs[:, 1]
                               ]['languages_spoken'].str.split(', ').apply(set)
    language_jaccard_sim = [(len(l1 & l2) / len(l1 | l2)) if len(l1 | l2)
                            > 0 else 0 for l1, l2 in zip(languages1, languages2)]

    # Compare dating preference
    same_dating_preference = df_users.iloc[pairs[:, 0]
                                           ]['dating_preference'].values == df_users.iloc[pairs[:, 1]]['dating_preference'].values

    # Compare photo count
    photo_count_diff = np.abs(
        df_users.iloc[pairs[:, 0]]['photo_count'].values - df_users.iloc[pairs[:, 1]]['photo_count'].values)
    max_photo_count = df_users['photo_count'].max()
    photo_count_similarity = 1 - \
        (photo_count_diff /
         max_photo_count) if max_photo_count > 0 else np.zeros_like(photo_count_diff)

    # Is user verified (binary)
    same_verified = df_users.iloc[pairs[:, 0]
                                  ]['is_verified'].values == df_users.iloc[pairs[:, 1]]['is_verified'].values

    # Is user premium (binary)
    same_premium = df_users.iloc[pairs[:, 0]
                                 ]['is_premium_user'].values == df_users.iloc[pairs[:, 1]]['is_premium_user'].values

    # Label calculation: Use weighted average of all similarities
    labels = (
        0.1 * same_relationship.astype(float) +
        0.05 * same_mbti.astype(float) +
        0.1 * np.array(jaccard_sim) +
        0.1 * age_similarity +
        0.1 * location_similarity +
        0.1 * zodiac_similarity.astype(float) +
        0.05 * height_similarity +
        0.05 * same_body_type.astype(float) +
        0.05 * np.array(language_jaccard_sim) +
        0.05 * same_dating_preference.astype(float) +
        0.05 * photo_count_similarity +
        0.05 * same_verified.astype(float) +
        0.05 * same_premium.astype(float)
    )

    # Convert to binary labels (similar or dissimilar)
    labels = (labels > 0.5).astype(float)
    return pairs.t(), torch.tensor(labels, dtype=torch.float).to(device)
