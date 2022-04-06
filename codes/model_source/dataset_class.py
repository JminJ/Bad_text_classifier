from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class BadSentenceDataset(Dataset):
    def __init__(self, parameters, target_dataset, mode):
        super(BadSentenceDataset, self).__init__()
        self.target_dataset = target_dataset

        if mode == 'valid':
            model_name = parameters.base_save_ckpt_path
        else:
            if parameters.model_type == 0:
                model_name = 'beomi/KcELECTRA-base'
            elif parameters.model_type == 1:
                model_name = 'tunib/electra-ko-base'
            elif parameters.model_type == 2:
                model_name = 'monologg/koelectra-base-v3-discriminator'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 

    def __len__(self):
        return len(self.target_dataset)
        
    def __getitem__(self, index):
        target_data = self.target_dataset.iloc[index]
        toked_result = self.tokenizer.encode_plus(str(target_data['text']), return_tensors='pt', max_length=128, padding = 'max_length', truncation=True)
        target_index = target_data['index']
        target_label = target_data['label']

        return {
            'index' : target_index,
            'text' : str(target_data['text']),
            'input_ids' : torch.squeeze(toked_result['input_ids']),
            'attention_mask' : torch.squeeze(toked_result['attention_mask']),
            'label' : target_label
        }