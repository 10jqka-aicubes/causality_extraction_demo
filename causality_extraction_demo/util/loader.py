#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description:   数据预处理模块
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#

import json
import numpy as np
import torch


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'cie_list': [(c, i, e)]}
    (cause,influence,effect)
    """
    D = []
    with open(filename, encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            text = l["text"]
            _id = l["id"]
            event = {}
            for item in l["entities"]:
                e_id = item["id"]
                e_txt = text[item["start_offset"] : item["end_offset"]]
                event[e_id] = e_txt
            cies = []
            for rs in l["relations"]:
                c = event[rs["from_id"]]
                e = event[rs["to_id"]]
                cies.append((c, rs["type"], e))
            D.append({"id": _id, "text": text, "cies": cies})
    return D


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    idx = -1
    for i in range(len(sequence)):
        if sequence[i : i + n] == pattern:
            return i
    return idx


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode="post"):
    """Numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, "__getitem__"):
        length = [length]

    slices = [np.s_[: length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == "post":
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == "pre":
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, "constant", constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class Collator(object):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_token, batch_token_mask = [], []
        batch_label, batch_label_mask = [], []
        for line in batch:
            text = line["text"].lower()
            tokens = self.tokenizer.tokenize(text, add_special_tokens=True, max_length=self.cfg.maxlen, truncation=True)
            res_tokenizer = self.tokenizer(text, max_length=self.cfg.maxlen)
            token_ids = res_tokenizer["input_ids"]
            mask = res_tokenizer["attention_mask"]
            # 解析三元组
            cies = set()
            for c, i, e in line["cies"]:
                c, e = c.lower(), e.lower()
                c_token, e_token = self.tokenizer.tokenize(c), self.tokenizer.tokenize(e)
                c_head, e_head = search(c_token, tokens), search(e_token, tokens)
                if c_head != -1 and e_head != -1:
                    c_tail, e_tail = c_head + len(c_token) - 1, e_head + len(e_token) - 1
                    if c_head <= c_tail and e_head <= e_tail:
                        cies.add((c_head, c_tail, self.cfg.predicate2id[i], e_head, e_tail))
            # 转化表格对应的label
            label = np.zeros([len(token_ids), len(token_ids), self.cfg.num_type])
            for ch, ct, i, eh, et in cies:
                label[ch, ct, 0] = 1  # 对应cause
                label[eh, et, 1] = 1  # 对应effect
                label[ch, eh, 2] = 1  # 对应relation
            # 考虑对称性，在识别cuase和effect将表格mask掉下三角部分
            mask_label = np.ones(label.shape)
            for i in range(len(token_ids)):
                for j in range(len(token_ids)):
                    if i > j:
                        mask_label[i, j, :2] = 0
            batch_token.append(token_ids)
            batch_token_mask.append(mask)
            batch_label.append(label)
            batch_label_mask.append(mask_label)
        batch_token = torch.tensor(sequence_padding(batch_token)).long()
        batch_token_mask = torch.tensor(sequence_padding(batch_token_mask)).float()
        batch_label = torch.tensor(sequence_padding(batch_label, seq_dims=2)).float()
        batch_label_mask = torch.tensor(sequence_padding(batch_label_mask, seq_dims=2)).float()
        return batch_token, batch_token_mask, batch_label, batch_label_mask
