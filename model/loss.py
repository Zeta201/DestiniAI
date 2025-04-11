import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2, p=2)
        loss = 0.5 * (label * dist.pow(2) + (1 - label)
                      * F.relu(self.margin - dist).pow(2))
        return loss.mean()
