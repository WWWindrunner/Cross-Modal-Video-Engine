#!/usr/bin/env bash

trainlist_dir=/data/shufan/shufan/mmaction2/data/kinetics400/trainlist_for_inference
vallist_dir=/data/shufan/shufan/mmaction2/data/kinetics400/vallist_for_inference
CONFIG=$1
CHECKPOINT=$2
MODEL_NAME=$3
MODEL_TYPE=$4
cd /data/shufan/shufan/mmaction2/tools/clip_inference
workdir=/data/shufan/shufan/mmaction2/data/kinetics400/cbm_data_grad_with_videoname/${MODEL_NAME}
mkdir /data/shufan/shufan/mmaction2/data/kinetics400/cbm_data_grad_with_videoname
mkdir ${workdir}
for trainlist in ${trainlist_dir}/*.txt
do
   echo ${trainlist}
   python inference_batch.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --cfg-options data.test.ann_file=${trainlist} work_dir=${workdir}
done
for vallist in ${vallist_dir}/*.txt
do
    echo ${vallist}
    python inference_grad_with_videoname.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --cfg-options data.test.ann_file=${vallist} work_dir=${workdir} --train_mode
done
