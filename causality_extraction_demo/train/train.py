#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description: 利用表格填充形式进行因果关系抽取，抽取形式为<cause,influence,effect>,其中influence只有一种类型；对应训练模块；
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#
import os
import sys

sys.path.append("..")
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertConfig
from transformers import BertTokenizerFast

from lr_scheduler import ReduceLROnPlateau
from model import CausalityModel
from util.config import Config
from util.loader import *
from util.utils import *


class TrainImpl:
    """
    模型训练
    """

    def __init__(self, cfg):
        """训练初始化函数"""
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_path, do_lower_case=True, add_special_tokens=False)
        # bert config
        config = BertConfig.from_pretrained(cfg.config_path)
        config.num_type = cfg.num_type
        config.bert_path = cfg.bert_path
        self.model = CausalityModel.from_pretrained(pretrained_model_name_or_path=cfg.checkpoint_path, config=config)
        self.model.to(cfg.device)

    def do_train(self):
        """
        训练主函数
        """
        print("Train begin!!")
        train_data = load_data(cfg.train_path)
        dev_data = train_data[:500]
        train_data = train_data[500:]

        collator = Collator(cfg, self.tokenizer)
        data_loader = DataLoader(train_data, collate_fn=collator, batch_size=cfg.batch_size, num_workers=0)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr, eps=cfg.min_num)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8
        )
        best_f1 = -1.0
        step = 0
        crossentropy = nn.BCEWithLogitsLoss(reduction="none")

        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_loss = 0
            with tqdm(total=data_loader.__len__(), desc="train", ncols=80) as t:
                for i, batch in enumerate(data_loader):
                    batch = [d.to(cfg.device) for d in batch]
                    batch_token, batch_token_mask, batch_label, batch_label_mask = batch
                    table = self.model(batch_token, batch_token_mask)  # BLLY
                    table = table.reshape([-1, cfg.num_type])
                    batch_label = batch_label.reshape([-1, cfg.num_type])
                    loss = crossentropy(table, batch_label)
                    loss = (loss * batch_label_mask.reshape([-1, cfg.num_type])).sum()
                    loss.backward()
                    step += 1
                    epoch_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    self.model.zero_grad()
                    t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                    t.update(1)

            print("")
            mean_f1 = evaluate(dev_data, self.model, cfg, self.tokenizer)
            logs = {"f1_score": mean_f1}
            show_info = f"Epoch: {epoch} - " + "-".join([f" {key}: {value:.4f} " for key, value in logs.items()])
            # print(show_info)
            scheduler.epoch_step(logs["f1_score"], epoch)
            if logs["f1_score"] > best_f1:
                best_f1 = logs["f1_score"]
                if isinstance(self.model, nn.DataParallel):
                    model_stat_dict = self.model.module.state_dict()
                else:
                    model_stat_dict = self.model.state_dict()
                state = {"epoch": epoch, "state_dict": model_stat_dict}
                torch.save(state, cfg.model_save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_dir")
    parser.add_argument("--save_model_dir")
    args = parser.parse_args()
    cfg = Config()
    cfg.train_file_dir = args.input_file_dir
    cfg.save_model_path = args.save_model_dir
    train_object = TrainImpl(cfg)
    train_object.do_train()
