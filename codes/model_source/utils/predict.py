from transformers import ElectraForSequenceClassification
import torch
from transformers import ElectraTokenizer
import argparse

import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def get_parameters():
    parser = argparse.ArgumentParser(description = 'parameters for predict user input.')
    parser.add_argument('-i', '--input_text', help = 'user input text.', default = '반갑습니다. JminJ입니다!', type = str)
    parser.add_argument('-b', '--base_ckpt', help = 'base path that saved trained checkpoints.', default = False, type = str)
    args = parser.parse_args()

    return args

Args = get_parameters()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if Args.base_ckpt:
    model = ElectraForSequenceClassification.from_pretrained(Args.base_ckpt)
    model.to(device).eval() 
else:
    # model = ElectraForSequenceClassification.from_pretrained('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')
    # model.to(device).eval()
    raise AttributeError()

tokenizer = ElectraTokenizer.from_pretrained(Args.base_ckpt)

def predict(toked_input, model):
    input_ids = toked_input['input_ids'].to(device)
    attention_mask = toked_input['attention_mask'].to(device)

    output = model(input_ids, attention_mask = attention_mask)[0]
    result = nn.Softmax(dim = 1)(output)

    return result


if __name__ == '__main__':
    label_list = ['bad', 'ok']
    
    tokenize_output = tokenizer.encode_plus(Args.input_text, max_length = 128, truncation=True, padding = 'max_length', return_tensors='pt')
    

    result = predict(tokenize_output, model)
    easy_result = np.argmax(result.data.cpu(), axis=1)
    
    print(Args.input_text)

    print(result)
    print(label_list[int(easy_result.item())])