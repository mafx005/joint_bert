# -*- coding:utf-8 -*-
# Author : M.
# Data : 2021/12/13

from trainer import Trainer
from config import Config
from model import Model
from dataset import MyDataset

config = Config(use_crf=False)
model = Model(config)
train_dataset = MyDataset(config, config.train_file)
test_dataset = MyDataset(config, config.test_file)

# trainer = Trainer(config=config,model=model,train_dataset=train_dataset,valid_dataset=test_dataset)
trainer = Trainer(config=config, model=model, train_dataset=train_dataset, valid_dataset=test_dataset)
for i in range(31):
    trainer.train(i)
