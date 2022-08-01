import torch
import os


class Config:

    # 训练时固定的参数
    maxlen = 256
    random_seed = 2008
    predicate2id = {"Influence": 0}
    num_type = 3
    weight_decay = 0.0
    max_grad_norm = 1.0
    warmup = 0.0
    min_num = 1e-7
    batch_size = 4
    epochs = 10
    lr = 2e-5
    device = torch.device("cuda")
    bert_path = "/read-only/common/pretrain_model/transformers/chinese-roberta-wwm-ext/"
    config_path = bert_path + "config.json"
    checkpoint_path = bert_path + "pytorch_model.bin"
    dict_path = bert_path + "vocab.txt"

    # 由setting.conf配置表传过来的参数
    train_file_dir = ""
    predict_file_dir = ""
    predict_result_file_dir = ""
    groundtruth_file_dir = ""
    save_model_path = ""

    @property
    def train_path(self):
        return os.path.join(self.train_file_dir, "train.txt")

    @property
    def predict_path(self):
        return os.path.join(self.predict_file_dir, "test.txt")

    @property
    def predict_result_path(self):
        return os.path.join(self.predict_result_file_dir, "predict.txt")

    @property
    def groundtruth_path(self):
        return os.path.join(self.groundtruth_file_dir, "test.txt")

    @property
    def model_save_path(self):
        return os.path.join(self.save_model_path, "pytorch_model.bin")
