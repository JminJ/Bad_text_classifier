from transformers import ElectraForSequenceClassification
import torch.nn as nn
import torch
import numpy as np

class ElectraBadClassifier(nn.Module):
    def __init__(self, ckpt = None, drop_p = 0.2):
        super(ElectraBadClassifier, self).__init__()
        self.drop_p = drop_p
        if ckpt is not None:
            self.electra_base = ElectraForSequenceClassification.from_pretrained(ckpt)
        else:
            self.electra_base = ElectraForSequenceClassification.from_pretrained('beomi/KcELECTRA-base')

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input_ids, attention_mask):
        base_output = self.electra_base(input_ids, attention_mask = attention_mask)
        model_result = base_output[0]

        softmax_result = self.softmax(model_result)

        return softmax_result