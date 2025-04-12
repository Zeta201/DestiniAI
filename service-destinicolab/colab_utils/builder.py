from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split


def build_model(df):
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(
        df[['user_id', 'profile_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVDpp(n_factors=20, n_epochs=10, lr_all=0.002, reg_all=0.1)

    return model, trainset, testset
