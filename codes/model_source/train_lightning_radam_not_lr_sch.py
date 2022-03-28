from gc import callbacks
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import RAdam
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import wandb
import random
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_class import BadSentenceDataset
from utils.loss import FocalLoss

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
    parser.add_argument('-e', '--epochs', help = 'full_train_epochs', defualt = 10, type = int)
    parser.add_argument('-bs', '--batch_size', help = 'batch size using in train time', default = 64, type = int)

    args = parser.parse_args()

    return args

def check_make_path(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

args = get_parameters()
config = {
    'device' : 'cuda' if torch.cuda.is_available else 'cpu',
    'train_dataset' : pd.read_csv('', sep = '\t'),
    'valid_dataset' : pd.read_csv('', sep = '\t'),
    'warmup_step' : 500,
    'loss_ALPHA' : 0.8,
    'model_name' : 'beomi/KcELECTRA-base'
}

if args.weight_decay != None:
    if args.use_float_16:
        wandb_loggers = pl.loggers.WandbLogger(project = 'Bad Sentence Classifier Birary',
            config={
               'epochs' : args.epochs,
               'batch_size' : args.batch_size,
               'learning_rate' : args.learning_rate,
               'weight_decay': args.weight_decay,
               'ver' : 'FocalLoss_AdamW_scheduler_fp16_not_sampler'
            })
    else:
        wandb_loggers = pl.loggers.WandbLogger(project = 'Bad Sentence Classifier Birary',
            config={
               'epochs' : args.epochs,
               'batch_size' : args.batch_size,
               'learning_rate' : args.learning_rate,
               'weight_decay': args.weight_decay,
               'ver' : 'FocalLoss_AdamW_scheduler_not_sampler'
            })
else:
    if args.use_float_16:
        wandb_loggers = pl.loggers.WandbLogger(project = 'Bad Sentence Classifier Birary',
            config={
               'epochs' : args.epochs,
               'batch_size' : args.batch_size,
               'learning_rate' : args.learning_rate,
               'ver' : 'FocalLoss_AdamW_scheduler_fp16_not_sampler'
            })
    else:
        wandb_loggers = pl.loggers.WandbLogger(project = 'Bad Sentence Classifier Birary',
            config={
               'epochs' : args.epochs,
               'batch_size' : args.batch_size,
               'learning_rate' : args.learning_rate,
               'ver' : 'FocalLoss_AdamW_scheduler_not_sampler'
            })

if args.base_ckpt_save_path != None:
    if args.use_float_16:
        if args.weight_decay != None:
            save_path = args.base_ckpt_save_path + '/use_all'
            check_make_path(save_path)
        else:
            save_path = args.base_ckpt_save_path + '/use_float_16'
            check_make_path(save_path)
    else:
        if args.weight_decay != None:
            save_path = args.base_ckpt_save_path + '/use_weight_decay'
            check_make_path(save_path)
        else:
            save_path = args.base_ckpt_save_path + '/base'
            check_make_path(save_path)
else:
    raise Exception('-b 인자를 입력해주셔야 합니다.')


class BadSentenceClassifier(pl.LightningModule):
    def __init__(self, parameter, **kwargs):
        self.model = ElectraForSequenceClassification.from_pretrained(kwargs.model_name)
        self.loss_fn = FocalLoss(alpha=kwargs.loss_ALPHA)
        self.parameter = parameter
        self.kwargs = kwargs

    def configure_optimizers(self):
        if self.args.weight_decay != None:
            optimizer = RAdam(self.parameters(), lr = self.parameter.learning_rate, weight_decay=self.args.weight_decay)
        else:
            optimizer = RAdam(self.parameters(), lr = self.parameter.learning_rate)

        return optimizer

    def forward(self, batch):
        batch_input_ids = batch['input_ids']
        batch_attention_mask = batch['attention_mask']

        output = self.model(batch_input_ids, attention_mask = batch_attention_mask)

        return output
    
    def calc_correct(self, output, label):
        logit_val, logit_label = torch.max(output.data, dim = 1)

        n_correct = 0
        # for get wrong data of batch
        for i in range(len(logit_label)):
            if logit_label[i] == label[i]:
                n_correct += 1

        return logit_label, n_correct

    def step(self, batch):
        model_output = self(batch)
        logit_label, n_correct = self.calc_correct(model_output, batch['label'].data())
        output_loss = self.loss_fn(logit_label, batch['label'].data())
        output_f1_score = f1_score(batch.label.data.cpu(), logit_label.data.cpu())

        return output_loss, output_f1_score

    def training_step(self, batch, batch_idx):
        loss, output_f1_score = self.step(batch)

        ## logs
        self.log("train_loss", loss, prog_bar = True, logger = True)
        self.log("train_f1_score", output_f1_score, prog_bar = True, logger = True)

        return loss

    def validation_step(self, batch, batch_idx):
        print('----- VALID -----')
        loss, output_f1_score = self.step(batch)

        ## logs
        self.log("valid_loss", loss, prog_bar = True, logger = True)
        self.log("valid_f1_score", output_f1_score, prog_bar = True, logger = True)

        return loss

    def train_dataloader(self):
        train_Dataset = BadSentenceDataset(self.kwargs.train_dataset)
        return DataLoader(
            train_Dataset,
            batch_size = self.parameter.batch_size,
            shuffle = True,
            drop_last = True
        )
    
    def val_dataloader(self):
        valid_Dataset = BadSentenceDataset(self.kwargs.valid_dataset)
        return DataLoader(
            valid_Dataset,
            batch_size = 32,
            shuffle = True,
            drop_last = True
        )

checkpoint_callback = ModelCheckpoint(
    dirpath = save_path,
    filename = f'/Focal_{config.loss_ALPHA}_RAdam_{config.warmup_step}_lr_{args.learning_rate}_data',
    verbose = True,
    monitor = 'val_loss',
    mode = 'min'
)

if args.use_float_16:
    trainer = pl.Trainer(
        logger = wandb_loggers,
        callbacks = [checkpoint_callback],
        max_epochs = config.max_epochs,
        gpus = -1,
        progress_bar_refresh_rate = 30,
        precision = 16,
    )
else:
    trainer = pl.Trainer(
        logger = wandb_loggers,
        callbacks = [checkpoint_callback],
        max_epochs = config.max_epochs,
        gpus = -1,
        progress_bar_refresh_rate = 30,
    )

model = BadSentenceClassifier(args, **config)
trainer.fit(model)