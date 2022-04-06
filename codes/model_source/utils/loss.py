import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, logits = False, reduce = True, cls_weights = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

        if cls_weights is None:
            self.base_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        else:
            self.base_cross_entropy = nn.CrossEntropyLoss(resuction = 'none', weight=cls_weights)

    def forward(self, inputs, targets):
        ce_loss = self.base_cross_entropy(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss