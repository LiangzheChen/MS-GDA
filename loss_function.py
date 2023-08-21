import torch
import torch.nn as nn
import torch.nn.functional as F

def get_link_prediction_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, full=False):
        num_data = x.shape[0]
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if full:
            return -1.0 * b.sum(1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b

class XeLoss(nn.Module):
    def __init__(self):
        super(XeLoss, self).__init__()

    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1) * F.log_softmax(x, dim=1) - F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b