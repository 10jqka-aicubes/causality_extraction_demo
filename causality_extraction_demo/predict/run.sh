#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


# 以下是样例，你可以自定义修改
python predict.py \
    --input_file_dir=$PREDICT_FILE_DIR \
    --load_model_dir=$SAVE_MODEL_DIR \
    --predict_file_dir=$PREDICT_RESULT_FILE_DIR