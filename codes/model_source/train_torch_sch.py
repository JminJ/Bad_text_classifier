from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import wandb
import random
import torch.backends.cudnn as cudnn

from dataset_class import BadSentenceDataset
from model import ElectraBadClassifier
from train_operation_cls import TrainOperation

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def get_parameters():
    parser = argparse.ArgumentParser(description = 'arguments for train bad sentence classifier')
    parser.add_argument('-lr', '--learning_rate', help='decise learning rate for train', default = 5e-05, type = float)
    parser.add_argument('-fp16', '--use_float_16', help = 'decise to apply float 16 or not', default = False, type = bool)
    parser.add_argument('-wd', '--weight_decay', help = 'define weight decay lambda', default = None, type = float)
    parser.add_argument('-b', '--base_save_ckpt_path', help = 'base path that will be saved trained checkpoints', default = None, type = str)
    parser.add_argument('-e', '--epochs', help = 'full_train_epochs', default = 5, type = int)
    parser.add_argument('-bs', '--batch_size', help = 'batch size using in train time', default = 64, type = int)
    ## model_type
    ## * 0 : beomi/KcELECTRA-base
    ## * 1 : tunib/electra-ko-base
    ## * 2 : monologg/koelectra-base-v3-discriminator
    parser.add_argument('-m_t', '--model_type', help = 'used to choose what electra model using for training', default = 0, type = int)

    args = parser.parse_args()

    return args

def check_make_path(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

# parameter를 설정
args = get_parameters()
config = {
    'device' : 'cuda' if torch.cuda.is_available else 'cpu',
    'train_dataset' : pd.read_csv('...', sep = '\t'),
    'valid_dataset' : pd.read_csv('...', sep = '\t'),
    'warmup_step' : 500,
    'loss_ALPHA' : 0.8,
    'mode' : 'train'
}

class Trainer:
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.kwargs = kwargs

        self.train_operation = TrainOperation(parameters, **kwargs)
        self.optimizer = AdamW(params = self.train_operation.model.parameters(), lr = self.parameters.learning_rate)
        self.lr_schedular = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=kwargs['warmup_step'], num_training_steps=len(self.kwargs['train_dataset']) / self.parameters.batch_size * self.parameters.epochs)

        # self.train_pd_dataset = pd.read_csv('/workspace/jminj_bad_sentence_classifier/Bad_text_classifier/datasets/concat_public_ok_data/result/concated_ko_smilegate_train.tsv', sep = '\t')
        # self.valid_pd_dataset = pd.read_csv('/workspace/jminj_bad_sentence_classifier/Bad_text_classifier/datasets/unsmile_valid_v1.0.tsv', sep = '\t')

        self.train_Dataset = BadSentenceDataset(self.parameters, target_dataset = self.kwargs['train_dataset'], mode = kwargs['mode'])
        self.valid_Dataset = BadSentenceDataset(self.parameters, target_dataset = self.kwargs['valid_dataset'], mode = kwargs['mode'])

        self.train_dataloader = DataLoader(
            self.train_Dataset,
            batch_size = self.parameters.batch_size,
            shuffle = True,
            drop_last = True
        )
        self.valid_dataloader = DataLoader(
            self.valid_Dataset,
            batch_size = 64,
            shuffle = False,
            drop_last = False
        )
        # fp16을 위한 scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # wandb init, model 추적
        self.wandb_init()
        wandb.watch(self.train_operation.model)

    def wandb_init(self):
        if self.parameters.use_float_16:
            if self.parameters.weight_decay != None:
                wandb.init(project = 'smilegate_train_kcElectra',
                    config={
                        'epochs' : self.parameters.epochs,
                        'batch_size' : self.parameters.batch_size,
                        'learning_rate' : self.parameters.learning_rate,
                        'weight_decay' : self.parameters.weight_decay,
                        'ver' : 'FocalLoss_AdamW_sch_fp_16_wd_not_sampler'
                    }
                )
            else:
                wandb.init(project = 'smilegate_train_kcElectra',
                    config={
                        'epochs' : self.parameters.epochs,
                        'batch_size' : self.parameters.batch_size,
                        'learning_rate' : self.parameters.learning_rate,
                        'ver' : 'FocalLoss_AdamW_sch_fp_16_not_sampler'
                    }
                )
        else:
            if self.parameters.weight_decay != None:
                wandb.init(project = 'smilegate_train_kcElectra',
                    config={
                        'epochs' : self.parameters.epochs,
                        'batch_size' : self.parameters.batch_size,
                        'learning_rate' : self.parameters.learning_rate,
                        'weight_decay' : self.parameters.weight_decay,
                        'ver' : 'FocalLoss_AdamW_sch_wd_not_sampler'
                    }
                )
            else:
                wandb.init(project = 'smilegate_train_kcElectra',
                    config={
                        'epochs' : self.parameters.epochs,
                        'batch_size' : self.parameters.batch_size,
                        'learning_rate' : self.parameters.learning_rate,
                        'ver' : 'FocalLoss_AdamW_sch_not_sampler'
                    }
                )

    def train(self, epoch):
        self.train_operation.model.train()
        tr_loss = 0
        tr_corrects = 0
        tr_f1_score = 0

        tr_steps = 0
        tr_examples = 0

        for _, batch in enumerate(self.train_dataloader, 0):
            step_loss, step_n_corrects, step_f1_score = self.train_operation.forward(batch, 'train')
            
            tr_loss += step_loss.item()
            tr_corrects += step_n_corrects
            tr_f1_score += step_f1_score

            tr_steps += 1
            tr_examples += batch['label'].size(0)

            wandb.log({'train loss' : tr_loss / tr_steps, 'train acc' : tr_corrects / tr_examples, 'train f1 score' : tr_f1_score / tr_steps})

            self.optimizer.zero_grad()

            if self.parameters.use_float_16:
                self.scaler.scale(step_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                step_loss.backward()
                self.optimizer.step()

            self.lr_schedular.step()

        print(f'\n----- epoch{epoch} result -----')
        step_loss_mean = tr_loss / tr_steps
        step_acc = tr_corrects / tr_examples
        step_f1_score_mean = tr_f1_score / tr_examples

        print(f'epoch_loss : {step_loss_mean}')
        print(f'epoch_acc : {step_acc}')
        print(f'epoch_f1_score : {step_f1_score_mean}')

    def valid(self): 
        self.train_operation.model.eval()
        vl_loss = 0
        vl_corrects = 0
        vl_f1_score = 0

        vl_steps = 0
        vl_examples = 0

        for _, batch in enumerate(self.valid_dataloader, 0):
            step_loss, step_n_corrects, step_f1_score = self.train_operation.forward(batch, 'valid')
            
            vl_loss += step_loss.item()
            vl_corrects += step_n_corrects
            vl_f1_score += step_f1_score

            vl_steps += 1
            vl_examples += batch['label'].size(0)

        wandb.log({'valid loss' : vl_loss / vl_steps, 'valid acc' : vl_corrects / vl_examples, 'valid f1 score' : vl_f1_score / vl_steps})
        
        print(f'\n----- valid result -----')
        step_loss_mean = vl_loss / vl_steps
        step_acc = vl_corrects / vl_examples
        step_f1_score_mean = vl_f1_score / vl_examples

        print(f'valid_loss : {step_loss_mean}')
        print(f'valid_acc : {step_acc}')
        print(f'valid_f1_score : {step_f1_score_mean}')

    def forward(self, save_path):
        model_name = 'kcElectra'
        if self.parameters.model_type == 1:
            model_name = 'tunibElectra'
        elif self.parameters.model_type == 2:
            model_name = 'koElectraV3'
            
        each_save_path = save_path + f"/{model_name}_Focal_{self.kwargs['loss_ALPHA']}_AdamW_scheduler_{self.kwargs['warmup_step']}_lr_{self.parameters.learning_rate}_data"
        
        for epoch in range(self.parameters.epochs):
            self.train(epoch)
            # epoch 마다 저장한다
            temp_save_path = each_save_path + f'/{epoch}'
            if not os.path.exists(temp_save_path):
                print(f'make_directory...')
                os.makedirs(temp_save_path)
            self.train_operation.save_model_checkpoint(temp_save_path)
            self.train_Dataset.tokenizer.save_pretrained(temp_save_path)

            self.valid()
        print('\nDone...')

# path 관련 함수들
def check_make_path(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

def make_save_directory(args):
    if args.base_save_ckpt_path != None:
        if args.use_float_16:
            if args.weight_decay != None:
                save_path = args.base_save_ckpt_path + '/use_all'
                check_make_path(save_path)
            else:
                save_path = args.base_save_ckpt_path + '/use_amp'
                check_make_path(save_path)
        else:
            if args.weight_decay != None:
                save_path = args.base_save_ckpt_path + '/use_weight_decay'
                check_make_path(save_path)
            else:
                save_path = args.base_save_ckpt_path + '/base'
                check_make_path(save_path)
    else:
        raise Exception('-b 인자를 입력해주셔야 합니다.')
    
    return save_path

if __name__ == '__main__':
    train = Trainer(args, **config)
    # model이 저장될 path를 만든다.
    save_path = make_save_directory(args)

    train.forward(save_path)