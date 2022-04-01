from torch.utils.data import Dataset
from transformers import ElectraTokenizer
import torch

class BadSentenceDataset(Dataset):
    def __init__(self, target_dataset):
        super(BadSentenceDataset, self).__init__()
        self.target_dataset = target_dataset
        self.tokenizer = ElectraTokenizer.from_pretrained('beomi/KcELECTRA-base') # 2022 ver도 test

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