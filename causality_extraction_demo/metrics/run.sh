#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

# 以下是样例，你可以自定义修改
python eval.py \
    --predict_file_dir=$PREDICT_RESULT_FILE_DIR \
    --groundtruth_file_dir=$GROUNDTRUTH_FILE_DIR \
    --result_json_file=$RESULT_JSON_FILE \
    --result_detail_file=$RESULT_DETAIL_FILE
