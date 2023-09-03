#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MODEL_NAME=$3
MODEL_TYPE=$4
DEVICE=$5
LEVEL=neural
train_list_dir=/data/shufan/shufan/mmaction2/data/kinetics400/trainlist_classwise_for_inference
RAWVIDEO_ROOT=/data/shufan/shufan/mmaction2/data/kinetics400/train_256
MASKVIDEO_ROOT=/data/shufan/shufan/mmaction2/data/kinetics400/train_256_mask

for trainlist in ${trainlist_dir}/*.txt
do
    echo get raw video embeddings
    action_name=$(basename ${trainlist})
    action_name=${action_name%.*}
    echo ${trainlist}
    echo ${action_name}
    CONCEPTEMB_ROOT=/data/shufan/shufan/MCT/data/kinetics400/model_embeddings/neural_level/${action_name}
    conceptlist=/data/shufan/shufan/mmaction2/data/kinetics400/conceptlist_classwise_for_inference/${action_name}.txt
    mkdir ${CONCEPTEMB_ROOT}
    echo "embedding inference..."
    mkdir ${CONCEPTEMB_ROOT}/${MODEL_NAME}
    python raw_video_inference.py ${CONFIG} ${CHECKPOINT} --level ${LEVEL} --eval top_k_accuracy --model_name ${MODEL_NAME}  --model_type ${MODEL_TYPE}  --cfg-options work_dir=${CONCEPTEMB_ROOT}/${MODEL_NAME} data.test.ann_file=${trainlist}

    echo get ctvs...
    echo ${conceptlist}
    python instance_level_ctv.py ${CONFIG} ${CHECKPOINT} --level ${LEVEL} --eval top_k_accuracy --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --ann_file_name "${conceptlist}" --raw_feats_dict ${CONCEPTEMB_ROOT}/${MODEL_NAME}/concept_embedding.pkl --cfg-options data.test.ann_file="${conceptlist}" work_dir=${CONCEPTEMB_ROOT}/${MODEL_NAME}
    rm ${CONCEPTEMB_ROOT}/${MODEL_NAME}/concept_embedding.pkl
done




#echo "probing..."
#mkdir /data/shufan/shufan/mmaction2/tools/clip_inference/probing_result
#python probing.py --data_root ${CONCEPTEMB_ROOT}/${MODEL_NAME}/train_256_mask --save_path /data/shufan/shufan/mmaction2/tools/clip_inference/probing_result/${MODEL_NAME}_entity_denoise_10.json --model_name ${MODEL_NAME}

#echo "get embedding data..."
#trainlist_dir=/data1/shufan/mmaction2/data/kinetics400/trainlist_for_inference
#vallist_dir=/data1/shufan/mmaction2/data/kinetics400/vallist_for_inference
#cd /data1/shufan/mmaction2/tools
#workdir=/data1/shufan/mmaction2/data/kinetics400/cbm_data/${MODEL_NAME}
#mkdir ${workdir}
#for trainlist in ${trainlist_dir}/*.txt
#do
#    echo ${trainlist}
#    python inference_batch.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --cfg-options data.test.ann_file=${trainlist} data.workers_per_gpu=1 work_dir=${workdir} --train_mode
#done
#for vallist in ${vallist_dir}/*.txt
#do
#    echo ${vallist}
    #python inference_batch.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --cfg-options data.test.ann_file=${vallist} data.workers_per_gpu=1 work_dir=${workdir}
#done

#echo 'train cbm model...'
#cd clip_inference
#python train_cbm.py --data_root ${CONCEPTEMB_ROOT}/${MODEL_NAME}/val_256_mask --grad_root ${workdir} --model_name ${MODEL_NAME}