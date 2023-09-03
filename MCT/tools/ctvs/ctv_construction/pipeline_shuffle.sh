#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MODEL_NAME=$3
MODEL_TYPE=$4
CLIP_LEN=$5
CONCEPTEMB_ROOT=/data/shufan/shufan/mmaction2/data/kinetics400/model_embeddings/entity_denoise_10
RAWVIDEO_ROOT=/data/shufan/shufan/mmaction2/data/kinetics400/val_256
MASKVIDEO_ROOT=/data/shufan/shufan/mmaction2/data/kinetics400/val_256_yolov5s_mask_denoise_10

trainlist_dir=/data/shufan/shufan/mmaction2/data/kinetics400/trainlist_for_inference
vallist_dir=/data/shufan/shufan/mmaction2/data/kinetics400/vallist_for_inference
conceptlist_dir=/data/shufan/shufan/mmaction2/data/kinetics400/conceptlist_for_inference_denoise_10

cd /data/shufan/shufan/mmaction2/tools/clip_inference
echo "embedding inference..."
mkdir ${CONCEPTEMB_ROOT}/${MODEL_NAME}

for vallist in ${vallist_dir}/*.txt
do
    echo ${vallist}
    python raw_video_inference_shuffle.py ${CONFIG} ${CHECKPOINT}  --eval top_k_accuracy --model_name ${MODEL_NAME}  --model_type ${MODEL_TYPE}  --cfg-options data.test.ann_file=${vallist} work_dir=${CONCEPTEMB_ROOT}/${MODEL_NAME}
done

#workdir=/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity_denoise_10/${MODEL_NAME}_shuffle
#mkdir ${workdir}
#for conceptlist in ${conceptlist_dir}/*.txt
#do
#    echo ${conceptlist}
#    python instance_level_ctv_shuffle.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --ann_file_name "${conceptlist}" --raw_feats_dict /data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity_denoise_10/${MODEL_NAME}/shuffle_concept_embedding.pkl --shuffle_idx_dict /data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity_denoise_10/${MODEL_NAME}/shuffle_concept_embedding_idx.pkl --cfg-options data.test.ann_file="${conceptlist}" work_dir=${workdir}
    #python inference_batch.py ${CONFIG} ${CHECKPOINT} --eval top_k_accuracy --model_name ${MODEL_NAME} --cfg-options data.test.ann_file=${vallist} work_dir=${workdir}
#done


#python embedding_inference_yolo.py --config ${CONFIG} --checkpoint ${CHECKPOINT} --model_name ${MODEL_NAME} --data_root ${MASKVIDEO_ROOT} --save_dir ${CONCEPTEMB_ROOT}/${MODEL_NAME}/val_256_mask --raw_feats_dict ${CONCEPTEMB_ROOT}/${MODEL_NAME}/concept_embedding.pkl --model_type ${MODEL_TYPE}  --device ${DEVICE}

