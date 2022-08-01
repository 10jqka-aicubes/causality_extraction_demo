#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description:   预测模块；
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#
import os
import sys

sys.path.append("..")
import json
import torch
import numpy as np
from transformers.models.bert.modeling_bert import BertConfig
from transformers import BertTokenizerFast
from util.config import Config
from model import CausalityModel


class PredictImpl:
    """模型预测"""

    def __init__(self, cfg):
        """预测初始化函数"""
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_path, do_lower_case=True, add_special_tokens=False)
        # bert config
        config = BertConfig.from_pretrained(cfg.config_path)
        config.num_type = cfg.num_type
        config.bert_path = cfg.bert_path
        self.model = CausalityModel.from_pretrained(pretrained_model_name_or_path=cfg.checkpoint_path, config=config)
        self.model.to(cfg.device)
        states = torch.load(cfg.model_save_path)
        state = states["state_dict"]
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)

    def causality_extraction(self, text, model, cfg, tokenizer, threshold=0):
        input_text = text.lower()
        res_tokenizer = tokenizer(input_text, return_offsets_mapping=True, max_length=cfg.maxlen)
        token_ids = res_tokenizer["input_ids"]
        mask = res_tokenizer["attention_mask"]
        mapping = res_tokenizer["offset_mapping"]
        token_ids = torch.LongTensor([token_ids]).to(cfg.device)
        mask = torch.Tensor([mask]).to(cfg.device)
        model.eval()
        with torch.no_grad():
            table = model(token_ids, mask)  # BLLY
            table = table.cpu().detach().numpy()[0]  # LLY
            # 识别事件
            causes, effects = set(), set()
            for h, t, r in zip(*np.where(table[:, :, :2] > threshold)):
                if r == 0:
                    if t > h:
                        causes.add((h, t))
                else:
                    if t > h:
                        effects.add((h, t))
            # 识别关系
            cies = set()
            for ch, ct in causes:
                for eh, et in effects:
                    if table[:, :, 2][ch, eh] > threshold:
                        cause = text[mapping[ch][0] : mapping[ct][1]]
                        effect = text[mapping[eh][0] : mapping[et][1]]
                        cies.add((cause, "Influence", effect))
            return list(cies)

    def do_predict(self):
        """
        预测主函数
        """
        print("Predict begin!!")
        with open(cfg.predict_path, "r", encoding="utf-8") as f, open(
            cfg.predict_result_path, "w", encoding="utf-8"
        ) as output:
            for line in f:
                line = json.loads(line)
                _id = line["id"]
                text = line["text"]
                res = self.causality_extraction(text, self.model, cfg, self.tokenizer)
                s = json.dumps({"id": line["id"], "text": line["text"], "pred_list": res}, ensure_ascii=False)
                output.write(s + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_dir")
    parser.add_argument("--load_model_dir")
    parser.add_argument("--predict_file_dir")
    args = parser.parse_args()

    cfg = Config()
    cfg.predict_file_dir = args.input_file_dir
    cfg.save_model_path = args.load_model_dir
    cfg.predict_result_file_dir = args.predict_file_dir
    predict_object = PredictImpl(cfg)
    predict_object.do_predict()
