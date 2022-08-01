# 【7-8暑期赛】面向金融领域的事件因果关系抽取

​&emsp;&emsp;抽取事件因果关系，构建事理图谱，是当前金融领域的一个研究应用热点。金融事理图谱中记录的大量金融事件的因果逻辑，有助于金融事件的影响因素探究以及因果逻辑推理，进而可对未来经济指标进行预测，指导投资建议。

​&emsp;&emsp;事件因果关系的识别，是构建事理图谱的核心环节，在实践中会面临以下挑战：1.因果关系数据标注难度大，很难积累到大规模高质量的可训练的数据，进而如何在有限的训练数据下，提高因果关系抽取的效果；​	2.在金融领域，文本中因果关系复杂，通常包含多个因果事件，而之间的因果影响范式多样，使得抽取难度加大。

- 本代码是该赛题的一个基础demo，仅供参考学习
- 比赛地址：<http://contest.aicubes.cn/>
- 时间：2022-07 ~ 2022-08
  ​

## 如何运行demo
- clone代码
- 预训练模型下载，https://github.com/brightmart/roberta_zh，版本为RoBERTa_zh_L12，torch版，将
下载的预训练模型放在项目的一个文件夹里，然后修改util/config.py中对应路径；
- 准备环境 
  - cuda11.0以上
  - python3.7以上
  - 安装python依赖 
  - python -m pip install -r requirements.txt
- 准备数据，从官网下载数据
   - 训练集存放在train/run.sh的--raw_data_dir对应的路径下
   - 测试集重命名为test.txt，存放在predict/run.sh的--raw_data_dir对应的路径下 
- 调整参数配置，参考模板项目的说明，主要配置文件causality_extraction_demo/setting.conf, 配置模型训练
涉及的超参数在util/config.py进行修改；
- 运行
   - 训练
   ```
   bash causality_extraction_demo/train/run.sh
   ```
   - 预测
   ```
   bash causality_extraction_demo/predict/run.sh
  ```
   - 计算结果指标
   ```
   bash causality_extraction_demo/metrics/run.sh
   ```
-  `metrics/eval.py`中有三类评价指标(event_f1,easy_f1, hard_f1)实现方式
   **为防止实现计算评测的公式有差异，可参考使用对应赛题提供的demo中的代码**


## 反作弊声明
1）参与者不允许使用多个小号，一经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码

## 赛事交流
![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)