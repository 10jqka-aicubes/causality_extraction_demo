#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description:   对应decode模块
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#
from tqdm import tqdm
import json
import torch
import numpy as np
import pylcs


class CIE(tuple):
    """用于判断三元组是否相同。"""

    def __init__(self, cie):
        self.spox = (
            tuple(list(cie[0])),
            cie[1],
            tuple(list(cie[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, cie):
        return self.spox == cie.spox


def extract_causality(text, model, cfg, tokenizer, threshold=0):
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


def evaluate_easy(pred_list, gold_list, threshold=0.9):
    """宽松式评估函数，计算f1、precision、recall"""

    def fsim(s1, s2):
        lcs = pylcs.lcs(s1, s2)
        return 2.0 * lcs / (len(s1) + len(s2))

    correct_num = 0
    if pred_list:
        for pitem in pred_list:
            pc, pe = pitem[0], pitem[2]
            for gitem in gold_list:
                gc, ge = gitem[0], gitem[2]
                c_score = fsim(pc, gc)
                e_score = fsim(pe, ge)
                if c_score >= threshold and e_score >= threshold:
                    correct_num += 1
    return correct_num


def evaluate(data, model, cfg, tokenizer, output_path=None):
    """评估函数，考虑到因果关系抽取的复杂性，在评估上采用三类评价维度，都是采用precision，recall，f1指标
    1）评价因果事件的识别情况
    2）评价因果事件三元组的识别情况；
    3）较为宽松的方式评价因果事件三元组的识别情况；因为可能存在预测的一个三元组跟标注的就相差一个token，这类我们也视为预测正确；
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    X_e = 1e-10
    if output_path:
        f = open(output_path, "w", encoding="utf-8")
    pbar = tqdm()
    for d in data:
        R = set([CIE(cie) for cie in extract_causality(d["text"], model, cfg, tokenizer)])
        T = set([CIE(cie) for cie in d["cies"]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()

        # 统计因果事件预测情况
        event_pred, event_gold = [], []
        for item in list(R):
            if item[0] not in event_pred:
                event_pred.append(item[0])
            if item[2] not in event_pred:
                event_pred.append(item[2])

        for dom in list(T):
            if dom[0] not in event_gold:
                event_gold.append(dom[0])
            if dom[2] not in event_gold:
                event_gold.append(dom[2])

        # 统计事件预测情况
        correct_num += len([t for t in event_pred if t in event_gold])
        predict_num += len(event_pred)
        gold_num += len(event_gold)

        # 宽松的因果三元组计算
        X_e += evaluate_easy(list(R), list(T))

        if output_path:
            s = json.dumps({"id": d["id"], "text": d["text"], "pred_list": list(R)}, ensure_ascii=False)
            f.write(s + "\n")
    pbar.close()
    if output_path:
        f.close()

    # 因果事件指标计算
    ep = correct_num / predict_num
    er = correct_num / gold_num
    ef = 2 * ep * er / (ep + er)

    # 因果三元组计算
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    # 宽松因果三元组计算
    f1_e, precision_e, recall_e = 2 * X_e / (Y + Z), X_e / Y, X_e / Z

    # 平均f1值
    mean_f1 = (f1 + ef + f1_e) / 3

    print("event metrics:                 f1: %.5f, precision: %.5f, recall: %.5f" % (ef, ep, er))
    print("hard causality tuple metrics:  f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall))
    print("easy causality tuple metrics:  f1: %.5f, precision: %.5f, recall: %.5f" % (f1_e, precision_e, recall_e))
    print("mean  metrics:                 f1: %.5f " % (mean_f1))
    return mean_f1
