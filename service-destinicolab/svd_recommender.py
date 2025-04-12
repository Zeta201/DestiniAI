class SVDRecommender:
    def __init__(self, model, trainset, interactions_df):
        self.model = model
        self.trainset = trainset
        self.interactions_df = interactions_df

    def recommend(self, user_id, top_n=5):
        try:
            known_items = set(
                self.trainset.ur[self.trainset.to_inner_uid(user_id)])
            all_items = set(range(self.trainset.n_items))
            unknown_items = all_items - {iid for (iid, _) in known_items}
            predictions = [(iid, self.model.predict(user_id, self.trainset.to_raw_iid(iid)).est)
                           for iid in unknown_items]
            top_profiles = sorted(
                predictions, key=lambda x: x[1], reverse=True)

            liked_by = set(
                self.interactions_df[self.interactions_df['profile_id'] == user_id]['user_id'])

            mutual_matches = []
            for profile_id, score in top_profiles:
                raw_id = self.trainset.to_raw_iid(profile_id)
                if raw_id in liked_by:
                    mutual_matches.append({
                        "profile_id": raw_id,
                        "similarity_score": round(score, 4)
                    })
                if len(mutual_matches) >= top_n:
                    break
            return mutual_matches
        except ValueError:
            print(f"User {user_id} not in training set.")
            return []
