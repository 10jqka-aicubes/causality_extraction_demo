#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description: 利用表格填充形式进行因果关系抽取，抽取形式为<cause,influence,effect>,其中influence只有一种类型；
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


class CausalityModel(BertPreTrainedModel):
    """利用表格填充的方式进行因果关系抽取"""

    def __init__(self, config):
        super(CausalityModel, self).__init__(config)
        self.bert = BertModel(config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc_b = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_e = nn.Linear(config.hidden_size, config.hidden_size)
        self.elu = nn.ELU()
        self.fc_table = nn.Linear(config.hidden_size, config.num_type)

        torch.nn.init.orthogonal_(self.fc_b.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_e.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_table.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        # encoder
        H = self.get_embed(token_ids, mask_token_ids)  # embed:BLH
        L = H.shape[1]
        # predication table generate
        H_B = self.fc_b(H)  # BLH
        H_E = self.fc_e(H)
        h = self.elu(H_B.unsqueeze(2).repeat(1, 1, L, 1) * H_E.unsqueeze(1).repeat(1, L, 1, 1))  # BLLH
        pred_table = self.fc_table(h)  # BLLY
        return pred_table

    def get_embed(self, token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed = bert_out[0]
        embed = self.dropout(embed)
        return embed
