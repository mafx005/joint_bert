# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 18:35
# @Author  : M.
# @File    : model.py
# @Describe:

from transformers import BertModel
from torch import nn
from torchcrf import CRF
import transformers


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.model_path)
        self.drop = nn.Dropout(0.2)
        self.num_intent_label = len(self.config.intent_vocab)
        self.num_slot_label = len(self.config.slot_vocab)
        self.hidden_size = self.bert.config.hidden_size
        self.fc_intent = nn.Linear(self.hidden_size, self.num_intent_label)
        self.fc_slot = nn.Linear(self.hidden_size, self.num_slot_label)
        self.crf = CRF(self.config.num_slot, batch_first=True).cuda()


    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_id=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq_encoding, pool_out = out[0], out[1]
        seq_encoding  = self.drop(seq_encoding)
        pool_out = self.drop(pool_out)

        intent_logits = self.fc_intent(pool_out)
        slot_logits = self.fc_slot(seq_encoding)
        return intent_logits, slot_logits
