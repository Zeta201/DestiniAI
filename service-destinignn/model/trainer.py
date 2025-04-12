import torch
from model.loss import ContrastiveLoss
from model.gnn_model import GraphSAGE
from utils.graph_builder import create_contrastive_pairs


class GNNTrainer:
    def __init__(self, data, df_users, lr=0.001, margin=0.2, device='cpu'):
        self.device = device
        self.data = data
        self.df_users = df_users
        self.model = GraphSAGE(data.x.shape[1], 64).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = ContrastiveLoss(margin=margin).to(self.device)

    def train(self, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data)
            pair_indices, labels = create_contrastive_pairs(
                self.df_users, batch_size=512, device=self.device)
            loss = self.criterion(
                out[pair_indices[0]], out[pair_indices[1]], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        return self.model
