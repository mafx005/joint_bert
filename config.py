# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 18:01
# @Author  : M.
# @File    : config.py
# @Describe:

from transformers import BertModel, BertTokenizer
import random
import torch


class Config:
    def __init__(self, use_crf=True):
        self.seed_every(seed=42)
        self.use_crf = use_crf
        self.model_path = 'bert-base-chinese'
        self.train_file = 'data/train.txt'
        self.test_file = 'data/test.txt'
        self.slot_label_file = 'data/slot_label.txt'
        self.intent_label_file = 'data/intent_label.txt'

        self.lr = 8e-6
        self.batch_size = 32
        self.batch_split = 2  # ?
        self.eval_step = 40

        self.intent_lable_num = 47
        self.num_slot = 122

        self.pad_idx = 0
        self.max_len = 90

        self.slot_vocab = self.build_vocab(self.slot_label_file)
        self.intent_vocab = self.build_vocab(self.intent_label_file)

        self.id2slot = {v: k for k, v in self.slot_vocab.items()}
        self.id2intent = {v: k for k, v in self.intent_vocab.items()}

        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertModel.from_pretrained(self.model_path)

    def build_vocab(self, file_name):
        with open(file_name, 'r', encoding='utf8') as f:
            lines = f.readlines()
            vocab = {word.strip(): idx for idx, word in enumerate(lines)}
        return vocab

    def seed_every(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
