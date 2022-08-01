#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Description:   评测模块；
# Version:       1.0
# Company:       www.10jqka.com.cn
# -------------------------------------------#

import json
import os
import pylcs


class CIE(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, cie):
        self.ciex = (
            tuple(list(cie[0])),
            cie[1],
            tuple(list(cie[2])),
        )

    def __hash__(self):
        return self.ciex.__hash__()

    def __eq__(self, cie):
        return self.ciex == cie.ciex


class EvalImpl:
    def load_data(self, input_file, mode="gold"):
        if mode == "gold":
            gold_data = []
            with open(os.path.join(input_file, "test.txt"), "r", encoding="utf-8") as f:
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
                    gold_data.append({"id": _id, "cies": cies})
            return gold_data
        else:
            pred_data = []
            with open(os.path.join(input_file, "predict.txt"), "r", encoding="utf-8") as f:
                for l in f:
                    l = json.loads(l)
                    _id = l["id"]
                    pred_data.append({"id": _id, "cies": l["pred_list"]})
            return pred_data

    def evaluate_easy(self, pred_list, gold_list, threshold=0.9):
        """评估函数，计算f1、precision、recall"""

        def fsim(s1, s2):
            """判断两个句子的相似度"""
            lcs = pylcs.lcs(s1, s2)
            return 2.0 * lcs / (len(s1) + len(s2))

        correct_num = 0
        if pred_list:
            for pitem in pred_list:
                pc, pe = pitem[0], pitem[2]
                for gitem in gold_list:
                    gc, ge = gitem[0], gitem[2]
                    c_jacc = fsim(pc, gc)
                    e_jacc = fsim(pe, ge)
                    if c_jacc >= threshold and e_jacc >= threshold:
                        correct_num += 1
        return correct_num

    def do_eval(self, predict_file_dir, groundtruth_file_dir, result_json_file, *args, **kargs):
        """评测主函数

        Args:
            predict_file_dir (Path): input, 模型预测结果的文件目录
            groundtruth_file_dir (Path): input, 真实结果的文件目录
            result_json_file (Path): output, 评测结果，json格式，{"f1": 0.99}
            result_detail_file (Path): output, 预测明细，可选
        """
        print("Eval begin!!")

        X, Y, Z = 1e-10, 1e-10, 1e-10
        correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
        X_e = 1e-10
        gold_data = self.load_data(groundtruth_file_dir, "gold")
        pred_data = self.load_data(predict_file_dir, "pred")
        assert len(gold_data) == len(pred_data)
        for i in range(len(pred_data)):
            if gold_data[i]["id"] != pred_data[i]["id"]:
                raise ValueError("预测的文件错误！")

            R = set([CIE(cie) for cie in pred_data[i]["cies"]])
            T = set([CIE(cie) for cie in gold_data[i]["cies"]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)

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

            # 统计事情预测情况
            correct_num += len([t for t in event_pred if t in event_gold])
            predict_num += len(event_pred)
            gold_num += len(event_gold)

            # easy的因果三元组计算
            X_e += self.evaluate_easy(list(R), list(T))

        # 因果事件指标计算
        ep = correct_num / predict_num
        er = correct_num / gold_num
        ef = 2 * ep * er / (ep + er)

        # 因果三元组计算
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        # 因果三元组宽松计算
        f1_e, precision_e, recall_e = 2 * X_e / (Y + Z), X_e / Y, X_e / Z

        # 平均f1值
        mean_f1 = (f1 + ef + f1_e) / 3

        print("event metrics:                 f1: %.5f, precision: %.5f, recall: %.5f" % (ef, ep, er))
        print("hard causality tuple metrics:  f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall))
        print("easy causality tuple metrics:  f1: %.5f, precision: %.5f, recall: %.5f" % (f1_e, precision_e, recall_e))
        print("mean  metrics:                 f1: %.5f" % (mean_f1))

        result = {
            "easy_f1": round(f1_e, 4),
            "hard_f1": round(f1, 4),
            "event_f1": round(ef, 4),
            "mean_f1": round(mean_f1, 4),
        }
        with open(result_json_file, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_dir")
    parser.add_argument("--groundtruth_file_dir")
    parser.add_argument("--result_json_file")
    parser.add_argument("--result_detail_file")
    args = parser.parse_args()

    eval_object = EvalImpl()
    eval_object.do_eval(args.predict_file_dir, args.groundtruth_file_dir, args.result_json_file)
