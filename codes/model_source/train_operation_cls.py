from transformers import ElectraForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from utils.loss import FocalLoss


class TrainOperation:
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.kwargs = kwargs

        self.loss_fn = FocalLoss(alpha = self.kwargs.loss_ALPHA)
        self.model = ElectraForSequenceClassification.from_pretrained(self.kwargs.model_name)
        
        self.not_same_data_dict = {'index' : [], 'text' : [], 'label' : []} # valid 중, 같지 않은 데이터 발견 시 expand함

    def calc_corrects(self, logits, labels, mode):
        logit_val, logit_label = torch.max(logits, dim = 1)
        n_corrects = 0
        not_same_data_list = []

        if mode == 'train':
            n_corrects = int(logit_label == labels).sum().item()

            return n_corrects, logit_label
        else:
            for i in range(len(labels)):
                if logits[i] == labels[i]:
                    n_corrects += 1
                    not_same_data_list.append(i)
            
            return n_corrects, not_same_data_list, logit_label

    def add_wrong_datas(self, batch, wrong_lists, logits):
        for i in wrong_lists:
            text = batch['text'][i]
            label = batch['label'][i]
            logit = logits[i]
            index = batch['index'][i]

            self.not_same_data_dict['text'] = list(self.not_same_data_dict['text']).append(text)
            self.not_same_data_dict['label'] = list(self.not_same_data_dict['label']).append(label)
            self.not_same_data_dict['logit'] = list(self.not_same_data_dict['logit']).append(int(logit.item()))
            self.not_same_data_dict['index'] = list(self.not_same_data_dict['index']).append(index)

    # fp16 적용해야 함
    def forward(self, batch, mode):
        input_ids = batch['input_ids'].to(self.kwargs.device)
        attention_mask = batch['attention_mask'].to(self.kwargs.device)
        label = torch.tensor(batch['label']).to(self.kwargs.device)

        model_output = self.model(input_ids, attention_mask = attention_mask)
        if mode == 'train':
            n_corrects, logit_label = self.calc_corrects(model_output, label, mode)
        else:
            n_corrects, not_same_data_list, logit_label = self.calc_corrects(model_output, label, mode)
            self.add_wrong_datas(batch, not_same_data_list, logit_label)
        
        step_loss = self.loss_fn(model_output, label)
        step_f1_score = f1_score(label.data.cpu(), logit_label.data.cpu())

        return step_loss, n_corrects, step_f1_score