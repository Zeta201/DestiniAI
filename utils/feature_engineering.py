from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def build_feature_matrices(df_users):
    # Handle textual columns (fill missing, convert to strings)
    for col in ['interests', 'about_me', 'personality', 'MBTI', 'relationship_goal',
                'education_level', 'occupation', 'zodiac_sign', 'body_type', 'languages_spoken', 'dating_preference']:
        df_users[col] = df_users[col].fillna('').astype(str)

    # Encode textual features using SentenceTransformers (embedding)
    interests_emb = sentence_model.encode(
        df_users['interests'].tolist(), convert_to_numpy=True)
    about_me_emb = sentence_model.encode(
        df_users['about_me'].tolist(), convert_to_numpy=True)
    personality_emb = sentence_model.encode(
        df_users['personality'].tolist(), convert_to_numpy=True)
    mbti_emb = sentence_model.encode(
        df_users['MBTI'].tolist(), convert_to_numpy=True)
    relationship_emb = sentence_model.encode(
        df_users['relationship_goal'].tolist(), convert_to_numpy=True)
    education_emb = sentence_model.encode(
        df_users['education_level'].tolist(), convert_to_numpy=True)
    occupation_emb = sentence_model.encode(
        df_users['occupation'].tolist(), convert_to_numpy=True)
    zodiac_emb = sentence_model.encode(
        df_users['zodiac_sign'].tolist(), convert_to_numpy=True)
    body_emb = sentence_model.encode(
        df_users['body_type'].tolist(), convert_to_numpy=True)
    languages_emb = sentence_model.encode(
        df_users['languages_spoken'].tolist(), convert_to_numpy=True)
    dating_pref_emb = sentence_model.encode(
        df_users['dating_preference'].tolist(), convert_to_numpy=True)

    # Weighted fusion of embeddings
    combined = (
        0.12 * interests_emb +
        0.12 * about_me_emb +
        0.10 * personality_emb +
        0.10 * mbti_emb +
        0.10 * relationship_emb +
        0.10 * education_emb +
        0.10 * occupation_emb +
        0.10 * zodiac_emb +
        0.12 * body_emb +
        0.05 * languages_emb +
        0.05 * dating_pref_emb
    )

    # Include numerical features (age, height, photo_count, is_verified, is_premium_user)
    numerical_features = df_users[[
        'age', 'height_cm', 'photo_count', 'is_verified', 'is_premium_user']].astype(float)
    # Convert boolean columns (is_verified, is_premium_user) into numeric (0/1)
    numerical_features['is_verified'] = numerical_features['is_verified'].astype(
        int)
    numerical_features['is_premium_user'] = numerical_features['is_premium_user'].astype(
        int)

    # Normalize numerical features
    scaler = StandardScaler()
    normalized_numerical = scaler.fit_transform(numerical_features)

    return combined, normalized_numerical, scaler
