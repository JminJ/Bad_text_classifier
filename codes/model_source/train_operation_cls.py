from model import ElectraBadClassifier
import torch
from sklearn.metrics import f1_score
from utils.loss import FocalLoss


class TrainOperation:
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.kwargs = kwargs

        self.loss_fn = FocalLoss(alpha = self.kwargs['loss_ALPHA'])

        if kwargs['mode'] == 'valid':
            model_name = self.parameters.base_save_ckpt_path
        else:
            if self.parameters.model_type == 0:
                model_name = 'beomi/KcELECTRA-base'
            elif self.parameters.model_type == 1:
                model_name = 'tunib/electra-ko-base'
            elif self.parameters.model_type == 2:
                model_name = 'monologg/koelectra-base-v3-discriminator'

        self.model = ElectraBadClassifier(model_name)
        self.model.to(self.kwargs['device'])
        
        self.not_same_data_dict = {'index' : [], 'text' : [], 'logit' : [], 'label' : []} # valid 중, 같지 않은 데이터 발견 시 expand함

    def calc_corrects(self, logits, labels, mode):
        logit_val, logit_label = torch.max(logits, dim = 1)
        n_corrects = 0
        not_same_data_list = []

        # if mode == 'train':
        #     n_corrects = (logit_label == labels).sum().item()

        #     return n_corrects, logit_label
        # else:
        #     for i in range(len(labels)):
        #         if logit_label[i].cpu().detach().numpy() == labels[i].cpu().detach().numpy():
        #             n_corrects += 1
        #         else:
        #             not_same_data_list.append(i)
            
        #     return n_corrects, not_same_data_list, logit_label
        
        n_corrects = (logit_label == labels).sum().item()

        return n_corrects, logit_label

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

    def save_model_checkpoint(self, path):
        self.model.electra_base.save_pretrained(path)

    def forward(self, batch, mode):
        input_ids = batch['input_ids'].to(self.kwargs['device'])
        attention_mask = batch['attention_mask'].to(self.kwargs['device'])
        label = torch.tensor(batch['label'], dtype=int).to(self.kwargs['device'])

        # fp16
        if self.parameters.use_float_16:
            with torch.cuda.amp.autocast():
                model_output = self.model(input_ids, attention_mask = attention_mask)
                step_loss = self.loss_fn(model_output, label)
        else:
            model_output = self.model(input_ids, attention_mask = attention_mask)
            step_loss = self.loss_fn(model_output, label)
        
        ## calc acc, f1_score
        # if mode == 'train':
        #     n_corrects, logit_label = self.calc_corrects(model_output, label, mode)
        # else:
        #     n_corrects, not_same_data_list, logit_label = self.calc_corrects(model_output, label, mode)
        #     self.add_wrong_datas(batch, not_same_data_list, logit_label)
        n_corrects, logit_label = self.calc_corrects(model_output, label, mode)
        
        step_f1_score = f1_score(label.data.cpu(), logit_label.data.cpu())


        return step_loss, n_corrects, step_f1_score