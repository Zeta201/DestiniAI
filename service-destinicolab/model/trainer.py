from surprise import accuracy


class SVDTrainer:
    def __init__(self, model, trainset, testset):
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def train_svd(self):
        self.model.fit(self.trainset)
        predictions = self.model.test(self.testset)
        print(f"Model RMSE (SVDpp): {accuracy.rmse(predictions)}")
        return self.model, self.trainset
